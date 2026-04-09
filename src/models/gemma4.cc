#include "ctranslate2/models/gemma4.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/models/model_factory.h"
#include "ctranslate2/ops/add.h"
#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/gather.h"
#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/mul.h"
#include "ctranslate2/ops/rotary.h"
#include "ctranslate2/ops/slide.h"
#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/topk.h"
#include "ctranslate2/ops/transpose.h"
#include "ctranslate2/models/model_reader.h"
#include "ctranslate2/sampling.h"
#include "ctranslate2/vocabulary.h"

namespace ctranslate2 {
  namespace models {

    // =========================================================================
    // Gemma4Decoder
    //
    // Thin subclass of TransformerDecoder that exposes decode_from_embeds()
    // as a public method.  The base class's protected method is called directly
    // for the multimodal KV-cache prefill path.
    // =========================================================================
    class Gemma4Decoder : public layers::TransformerDecoder {
    public:
      using layers::TransformerDecoder::TransformerDecoder;

      // Prefill the KV-cache with pre-computed embedding vectors and return
      // logits for every position in the sequence.
      // inputs_embeds: [batch, seq_len, text_hidden]  (already scaled)
      // lengths:       [batch]  int32 (sequence lengths)
      // state:         decoder KV-cache state (initialised before call)
      // logits (out):  [batch, seq_len, vocab_size]
      // input_ids: [batch, seq_len] int32 — used for token-side PLE lookup.
      void forward_with_embeds(const StorageView& inputs_embeds,
                               const StorageView& input_ids,
                               const StorageView& lengths,
                               layers::DecoderState& state,
                               StorageView& logits) {
        decode_from_embeds(inputs_embeds, &input_ids, &lengths, /*step=*/0, state, &logits);
      }
    };

    // =========================================================================
    // Gemma4VisionSelfAttention
    // =========================================================================

    Gemma4VisionSelfAttention::Gemma4VisionSelfAttention(const Model& model,
                                                         const std::string& scope,
                                                         dim_t num_heads,
                                                         dim_t head_dim)
      : _linear({layers::Dense(model, scope + "/linear_0"),
                 layers::Dense(model, scope + "/linear_1")})
      , _q_norm(layers::build_optional_layer<layers::LayerNorm>(model, scope + "/q_norm"))
      , _k_norm(layers::build_optional_layer<layers::LayerNorm>(model, scope + "/k_norm"))
      , _v_norm(layers::build_optional_layer<layers::LayerNorm>(model, scope + "/v_norm"))
      , _num_heads(num_heads)
      , _head_dim(head_dim)
    {
    }

    void Gemma4VisionSelfAttention::operator()(const StorageView& x,
                                               const StorageView& cos,
                                               const StorageView& sin,
                                               StorageView& output) const {
      const Device device = x.device();
      const DataType dtype = x.dtype();
      const dim_t batch_size = x.dim(0);
      const dim_t seq_len = x.dim(1);

      // Fused QKV projection: [B, T, D] → [B, T, 3*H*D_head] (when H==H_kv)
      StorageView fused_proj(dtype, device);
      _linear[0](x, fused_proj);

      // Split into Q, K, V: each [B, T, H*D_head]
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);
      ops::Split(2)(fused_proj, queries_proj, keys_proj, values_proj);

      // Reshape to [B, T, H, D_head]
      queries_proj.reshape({batch_size, seq_len, _num_heads, _head_dim});
      keys_proj.reshape({batch_size, seq_len, _num_heads, _head_dim});
      values_proj.reshape({batch_size, seq_len, _num_heads, _head_dim});

      // Apply per-head norms
      if (_q_norm) {
        StorageView tmp(dtype, device);
        (*_q_norm)(queries_proj, tmp);
        queries_proj = std::move(tmp);
      }
      if (_k_norm) {
        StorageView tmp(dtype, device);
        (*_k_norm)(keys_proj, tmp);
        keys_proj = std::move(tmp);
      }
      if (_v_norm) {
        StorageView tmp(dtype, device);
        (*_v_norm)(values_proj, tmp);
        values_proj = std::move(tmp);
      }

      // Apply 2D RoPE to Q and K.
      // cos/sin: [B*T, head_dim]. Reshape Q/K to [1, B*T, H, D_head] for Rotary.
      {
        const ops::Rotary rotary_op(_head_dim, /*interleave=*/false);
        const dim_t bt = batch_size * seq_len;

        queries_proj.reshape({1, bt, _num_heads, _head_dim});
        StorageView q_rotated(dtype, device);
        rotary_op(queries_proj, sin, cos, q_rotated, /*is_transposed=*/false);
        queries_proj = std::move(q_rotated);
        queries_proj.reshape({batch_size, seq_len, _num_heads, _head_dim});

        keys_proj.reshape({1, bt, _num_heads, _head_dim});
        StorageView k_rotated(dtype, device);
        rotary_op(keys_proj, sin, cos, k_rotated, /*is_transposed=*/false);
        keys_proj = std::move(k_rotated);
        keys_proj.reshape({batch_size, seq_len, _num_heads, _head_dim});
      }

      // Scaled dot-product attention (bidirectional, no mask).
      // Transpose to [B, H, T, D_head] for batched matmul.
      {
        // [B, T, H, D] ↔ [B, H, T, D]
        const ops::Transpose tr_bhsd(std::vector<dim_t>{0, 2, 1, 3});
        StorageView qt(dtype, device), kt(dtype, device), vt(dtype, device);
        tr_bhsd(queries_proj, qt);
        tr_bhsd(keys_proj, kt);
        tr_bhsd(values_proj, vt);

        // Attention weights: [B, H, T, T]
        StorageView attn_weights(dtype, device);
        ops::MatMul(/*trans_a=*/false, /*trans_b=*/true, /*alpha=*/1.0f)(qt, kt, attn_weights);

        // Softmax (in-place)
        ops::SoftMax()(attn_weights);

        // Context: [B, H, T, D]
        StorageView context(dtype, device);
        ops::MatMul(/*trans_a=*/false, /*trans_b=*/false)(attn_weights, vt, context);

        // Transpose back: [B, T, H, D]
        tr_bhsd(context, queries_proj);
      }

      // Reshape: [B, T, H, D] → [B, T, H*D]
      queries_proj.reshape({batch_size, seq_len, _num_heads * _head_dim});

      // Output projection
      _linear[1](queries_proj, output);
    }


    // =========================================================================
    // Gemma4VisionEncoderLayer
    // =========================================================================

    Gemma4VisionEncoderLayer::Gemma4VisionEncoderLayer(const Model& model,
                                                        const std::string& scope,
                                                        dim_t num_heads,
                                                        dim_t head_dim)
      : _input_layer_norm(model, scope + "/input_layer_norm")
      , _post_attention_layer_norm(model, scope + "/post_attention_layer_norm")
      , _pre_feedforward_layer_norm(model, scope + "/pre_feedforward_layer_norm")
      , _post_feedforward_layer_norm(model, scope + "/post_feedforward_layer_norm")
      , _self_attention(model, scope + "/self_attention", num_heads, head_dim)
      , _ff_gate(model, scope + "/ffn/linear_0")
      , _ff_up(model, scope + "/ffn/linear_0_noact")
      , _ff2(model, scope + "/ffn/linear_1")
    {
    }

    void Gemma4VisionEncoderLayer::operator()(const StorageView& input,
                                               const StorageView& cos,
                                               const StorageView& sin,
                                               StorageView& output) const {
      const Device device = input.device();
      const DataType dtype = input.dtype();

      // --- Attention block ---
      StorageView residual = input;  // save residual

      // Pre-attention norm
      StorageView normed(dtype, device);
      _input_layer_norm(input, normed);

      // Self-attention
      StorageView attn_out(dtype, device);
      _self_attention(normed, cos, sin, attn_out);

      // Post-attention norm + residual
      StorageView post_attn(dtype, device);
      _post_attention_layer_norm(attn_out, post_attn);
      ops::Add()(residual, post_attn, normed);  // normed = residual + post_attn

      residual = normed;  // update residual

      // --- Feed-forward block (GeGLU) ---
      // Pre-FFN norm
      StorageView ffn_in(dtype, device);
      _pre_feedforward_layer_norm(normed, ffn_in);

      // gate = act(linear_0(ffn_in)), up = linear_0_noact(ffn_in)
      StorageView gate(dtype, device), up(dtype, device), ffn_out(dtype, device);
      _ff_gate(ffn_in, gate);
      _ff_up(ffn_in, up);

      // GeGLU: element-wise product of gate (with activation) and up
      ops::Mul()(gate, up, ffn_in);

      // Down projection
      _ff2(ffn_in, ffn_out);

      // Post-FFN norm + residual
      StorageView post_ffn(dtype, device);
      _post_feedforward_layer_norm(ffn_out, post_ffn);
      ops::Add()(residual, post_ffn, output);
    }


    // =========================================================================
    // Gemma4VisionEncoder
    // =========================================================================

    static StorageView load_vision_weight(const Model& model, const std::string& name) {
      return model.get_variable(name).sync_copy();
    }

    Gemma4VisionEncoder::Gemma4VisionEncoder(const Model& model, const std::string& scope)
      : _input_proj(model, scope + "/patch_embedder/input_proj")
      , _position_embedding_table(load_vision_weight(model,
            scope + "/patch_embedder/position_embedding_table"))
      , _rope_cos(load_vision_weight(model, scope + "/rope_cos"))
      , _rope_sin(load_vision_weight(model, scope + "/rope_sin"))
      , _num_heads(static_cast<dim_t>(
            model.get_attribute_with_default<int16_t>(scope + "/num_heads", 12)))
      , _head_dim(model.get_attribute_with_default<int32_t>(
            scope + "/layer_0/self_attention/head_dim", 64))
      , _layers(layers::build_layers_list<Gemma4VisionEncoderLayer>(
            model, scope + "/layer", _num_heads, _head_dim))
    {
    }

    void Gemma4VisionEncoder::build_2d_rope(const StorageView& pixel_position_ids,
                                             StorageView& cos,
                                             StorageView& sin) const {
      const Device device = pixel_position_ids.device();
      // pixel_position_ids: [B, T, 2]
      const dim_t B = pixel_position_ids.dim(0);
      const dim_t T = pixel_position_ids.dim(1);

      // Extract x-positions and y-positions.
      // ops::Slice along axis 2: [B, T, 0:1] and [B, T, 1:2]
      StorageView x_pos_3d(DataType::INT32, device);
      StorageView y_pos_3d(DataType::INT32, device);
      ops::Slide(2, 0, 1)(pixel_position_ids, x_pos_3d);  // [B, T, 1]
      ops::Slide(2, 1, 1)(pixel_position_ids, y_pos_3d);  // [B, T, 1]

      // Reshape to [B*T] for gather
      StorageView x_pos = x_pos_3d;
      StorageView y_pos = y_pos_3d;
      x_pos.reshape({B * T});
      y_pos.reshape({B * T});

      // Clamp negative indices (e.g. -1 tile-separator markers) to 0
      // so they don't cause out-of-bounds gather on CPU/GPU.
      {
        StorageView x_cpu = x_pos.to(Device::CPU);
        StorageView y_cpu = y_pos.to(Device::CPU);
        int32_t* xd = x_cpu.data<int32_t>();
        int32_t* yd = y_cpu.data<int32_t>();
        for (dim_t i = 0; i < B * T; ++i) {
          if (xd[i] < 0) xd[i] = 0;
          if (yd[i] < 0) yd[i] = 0;
        }
        x_pos = x_cpu.to(device);
        y_pos = y_cpu.to(device);
      }

      // Gather: rope_cos [max_pos, spatial_dim] indexed by x/y positions
      const DataType float_dtype = _rope_cos.dtype();
      StorageView x_cos(float_dtype, device), y_cos(float_dtype, device);
      StorageView x_sin(float_dtype, device), y_sin(float_dtype, device);

      // Move rope tables to device if needed
      StorageView rope_cos = _rope_cos.to(device);
      StorageView rope_sin = _rope_sin.to(device);

      ops::Gather(/*axis=*/0)(rope_cos, x_pos, x_cos);  // [B*T, spatial_dim]
      ops::Gather(/*axis=*/0)(rope_cos, y_pos, y_cos);  // [B*T, spatial_dim]
      ops::Gather(/*axis=*/0)(rope_sin, x_pos, x_sin);  // [B*T, spatial_dim]
      ops::Gather(/*axis=*/0)(rope_sin, y_pos, y_sin);  // [B*T, spatial_dim]

      // Concatenate along last dim: [B*T, head_dim]
      ops::Concat(1)({&x_cos, &y_cos}, cos);
      ops::Concat(1)({&x_sin, &y_sin}, sin);
    }

    void Gemma4VisionEncoder::operator()(const StorageView& pixel_values,
                                          const StorageView& pixel_position_ids,
                                          StorageView& output) const {
      const Device device = pixel_values.device();
      const DataType dtype = pixel_values.dtype();
      const dim_t B = pixel_values.dim(0);
      const dim_t T = pixel_values.dim(1);

      // 1. Patch embedding: linear projection
      StorageView patch_embeds(dtype, device);
      _input_proj(pixel_values, patch_embeds);  // [B, T, hidden]

      // 2. Add 2D position embeddings.
      //    position_embedding_table: [2, max_pos, hidden]
      //    pixel_position_ids: [B, T, 2] (int32 x and y indices)
      //    For each patch (b, t): embed = embed + table[0][x] + table[1][y]
      {
        // Extract x and y position indices
        StorageView x_pos_3d(DataType::INT32, device);
        StorageView y_pos_3d(DataType::INT32, device);
        ops::Slide(2, 0, 1)(pixel_position_ids, x_pos_3d);  // [B, T, 1]
        ops::Slide(2, 1, 1)(pixel_position_ids, y_pos_3d);  // [B, T, 1]
        StorageView x_pos = x_pos_3d;
        StorageView y_pos = y_pos_3d;
        x_pos.reshape({B * T});
        y_pos.reshape({B * T});

        // Clamp negative indices (tile-separator markers = -1) to 0.
        {
          StorageView x_cpu = x_pos.to(Device::CPU);
          StorageView y_cpu = y_pos.to(Device::CPU);
          int32_t* xd = x_cpu.data<int32_t>();
          int32_t* yd = y_cpu.data<int32_t>();
          for (dim_t i = 0; i < B * T; ++i) {
            if (xd[i] < 0) xd[i] = 0;
            if (yd[i] < 0) yd[i] = 0;
          }
          x_pos = x_cpu.to(device);
          y_pos = y_cpu.to(device);
        }

        // Extract x and y embedding tables from position_embedding_table [2, max_pos, hidden]
        StorageView tbl = _position_embedding_table.to(device);
        StorageView x_tbl(dtype, device), y_tbl(dtype, device);
        ops::Slide(0, 0, 1)(tbl, x_tbl);  // [1, max_pos, hidden]
        x_tbl.reshape({x_tbl.dim(1), x_tbl.dim(2)});  // [max_pos, hidden]
        ops::Slide(0, 1, 1)(tbl, y_tbl);  // [1, max_pos, hidden]
        y_tbl.reshape({y_tbl.dim(1), y_tbl.dim(2)});  // [max_pos, hidden]

        // Gather position embeddings: [B*T, hidden]
        StorageView x_pe(dtype, device), y_pe(dtype, device);
        ops::Gather(0)(x_tbl, x_pos, x_pe);
        ops::Gather(0)(y_tbl, y_pos, y_pe);

        // Reshape patch_embeds to [B*T, hidden] for add
        patch_embeds.reshape({B * T, patch_embeds.dim(2)});
        ops::Add()(patch_embeds, x_pe, patch_embeds);
        ops::Add()(patch_embeds, y_pe, patch_embeds);
        patch_embeds.reshape({B, T, patch_embeds.dim(1)});
      }

      // 3. Build 2D RoPE cos/sin tables for this batch.
      StorageView rope_cos(dtype, device), rope_sin(dtype, device);
      build_2d_rope(pixel_position_ids, rope_cos, rope_sin);

      // 4. Run encoder layers.
      StorageView hidden = std::move(patch_embeds);
      for (const auto& layer : _layers) {
        StorageView layer_out(dtype, device);
        (*layer)(hidden, rope_cos, rope_sin, layer_out);
        hidden = std::move(layer_out);
      }

      output = std::move(hidden);
    }


    // =========================================================================
    // Gemma4MultimodalProjector
    // =========================================================================

    Gemma4MultimodalProjector::Gemma4MultimodalProjector(const Model& model,
                                                          const std::string& scope)
      : _pre_projection_norm(model, scope + "/pre_projection_norm")
      , _projection(model, scope + "/projection")
    {
    }

    void Gemma4MultimodalProjector::operator()(const StorageView& input,
                                                StorageView& output) const {
      StorageView normed(input.dtype(), input.device());
      _pre_projection_norm(input, normed);
      _projection(normed, output);
    }


    // =========================================================================
    // Gemma4Model
    // =========================================================================

    void Gemma4Model::initialize(ModelReader& model_reader) {
      Model::initialize(model_reader);

      VocabularyInfo vocab_info;
      vocab_info.unk_token = config["unk_token"];
      vocab_info.bos_token = config["bos_token"];
      vocab_info.eos_token = config["eos_token"];

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error("Cannot load the vocabulary from the model directory");

      if (config.contains("image_token_id"))
        _image_token_id = config["image_token_id"].get<size_t>();
    }

    const Vocabulary& Gemma4Model::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t Gemma4Model::get_image_token_id() const {
      return _image_token_id;
    }

    size_t Gemma4Model::current_spec_revision() const {
      return 1;
    }

    bool Gemma4Model::is_quantizable(const std::string& variable_name) const {
      // Variable must end in exactly "weight" (not "weight_scale", "weight_zero", etc.)
      // to be eligible for int8/int16 quantization.
      // Note: "per_layer_token_embedding" and "position_embedding_table" end in "embedding"
      // and "table" respectively, so they are naturally excluded by this check.
      if (!ends_with(variable_name, "weight"))
        return false;
      // Exclude precomputed RoPE tables (read-only, non-learnable).
      return (variable_name.find("rope_cos") == std::string::npos
              && variable_name.find("rope_sin") == std::string::npos);
    }

    bool Gemma4Model::is_linear_weight(const std::string& variable_name) const {
      // GEMM pre-packing applies to quantizable weights that are used in matrix
      // multiplications.  Embedding lookup tables (Gather, not MatMul) are excluded
      // to match the pattern used by WhisperModel::is_linear_weight.
      return is_quantizable(variable_name)
             && variable_name.find("embedding") == std::string::npos;
    }

    std::unique_ptr<Model> Gemma4Model::clone() const {
      return std::make_unique<Gemma4Model>(*this);
    }


    // =========================================================================
    // Gemma4Replica
    // =========================================================================

    std::unique_ptr<Gemma4Replica>
    Gemma4Replica::create_from_model(const Model& model) {
      const auto* gemma4_model = dynamic_cast<const Gemma4Model*>(&model);
      if (!gemma4_model)
        throw std::invalid_argument("Expected a Gemma4Model");
      return std::make_unique<Gemma4Replica>(
          std::static_pointer_cast<const Gemma4Model>(model.shared_from_this()));
    }

    Gemma4Replica::Gemma4Replica(const std::shared_ptr<const Gemma4Model>& model)
      : ModelReplica(model)
      , _model(model)
      , _vision_encoder(*model, "vision_encoder")
      , _multimodal_projector(*model, "multimodal_embedder")
      , _decoder(std::make_unique<Gemma4Decoder>(*model, "decoder"))
    {
    }

    StorageView Gemma4Replica::encode_vision(const StorageView& pixel_values,
                                              const StorageView& pixel_position_ids) {
      const Device device = _model->device();

      StorageView pixel_values_d = pixel_values.to(device);
      StorageView pixel_position_ids_d = pixel_position_ids.to(device);

      // 1. Vision encoder: patches → [B, T, vision_hidden]
      StorageView vision_hidden(_vision_encoder.output_type(), device);
      _vision_encoder(pixel_values_d, pixel_position_ids_d, vision_hidden);

      // 2. Pooler: scale by sqrt(hidden_size).
      //    (Spatial average pooling is expected to be performed on the Python side
      //    before passing pixel_values; here we apply only the scalar scaling.)
      {
        const float sqrt_hidden = std::sqrt(static_cast<float>(_vision_encoder.output_size()));
        const StorageView scale_factor = StorageView(sqrt_hidden).to(_vision_encoder.output_type());
        ops::Mul()(vision_hidden, scale_factor, vision_hidden);
      }

      // 3. Multimodal projector: [B, T, vision_hidden] → [B, T, text_hidden]
      StorageView text_embeds(_multimodal_projector.output_type(), device);
      _multimodal_projector(vision_hidden, text_embeds);

      return text_embeds;
    }


    // =========================================================================
    // Gemma4Replica::_build_inputs_embeds
    //
    // Builds the merged text+vision embedding tensor:
    //   1. Look up text token embeddings via the decoder embedding table.
    //   2. Apply Gemma 4 embedding scale to text tokens: *= sqrt(text_hidden_size).
    //   3. Overwrite image placeholder positions with vision_embeds (unscaled).
    //
    // The scale is applied BEFORE injecting vision embeddings so that vision
    // features are inserted at their native magnitude, matching HF behaviour
    // (HF scales text embeddings in get_input_embeddings() then does
    //  inputs_embeds.masked_scatter(..., image_features) on top).
    //
    // The result is ready to be passed directly to Gemma4Decoder::forward_with_embeds(),
    // which skips the embedding lookup and scale internally.
    // =========================================================================
    void Gemma4Replica::_build_inputs_embeds(
        const StorageView& input_ids,        // [batch, seq_len] int32 (CPU)
        const StorageView& vision_embeds,    // [batch, num_img_tokens, text_hidden] float
        size_t image_token_id,
        StorageView& inputs_embeds) const {

      const dim_t batch     = input_ids.dim(0);
      const dim_t seq_len   = input_ids.dim(1);
      const dim_t text_hidden = static_cast<dim_t>(_multimodal_projector.output_size());

      // 1. Token embedding lookup (raw, unscaled).
      layers::Embeddings token_embeddings(*_model, "decoder/embeddings");
      // Initialize with the embedding table dtype so that Gather's data<T>() doesn't
      // ASSERT_DTYPE-fail when the model is float16 (the default StorageView is float32).
      StorageView raw_embeds(token_embeddings.output_type(), _model->device());
      token_embeddings(input_ids.to(_model->device()), raw_embeds);

      // Work in float32 on CPU for safe element-wise injection.
      StorageView inputs_embeds_cpu = raw_embeds.to(DataType::FLOAT32).to(Device::CPU);

      // 2. Apply Gemma 4 embedding scale to text token embeddings: *= sqrt(text_hidden_size).
      // This must be done BEFORE injecting vision embeddings so that the scale is only
      // applied to text tokens, not to vision features.  HF applies the embedding scale
      // inside get_input_embeddings() and then does masked_scatter(image_features) on top,
      // so vision embeddings are inserted already in the correct numerical range.
      {
        const float scale = std::sqrt(static_cast<float>(text_hidden));
        float*      data  = inputs_embeds_cpu.data<float>();
        const dim_t total = inputs_embeds_cpu.size();
        for (dim_t i = 0; i < total; ++i)
          data[i] *= scale;
      }

      // 3. Overwrite image placeholder positions with vision_embeds (unscaled).
      const StorageView vis_cpu = vision_embeds.to(DataType::FLOAT32).to(Device::CPU);
      const StorageView input_ids_cpu = input_ids.to(Device::CPU);

      const int32_t* ids_data = input_ids_cpu.data<int32_t>();
      float*         emb_data = inputs_embeds_cpu.data<float>();
      const float*   vis_data = vis_cpu.data<float>();
      const dim_t    num_img  = vis_cpu.dim(1);

      for (dim_t b = 0; b < batch; ++b) {
        dim_t vis_pos = 0;
        for (dim_t s = 0; s < seq_len; ++s) {
          if (static_cast<size_t>(ids_data[b * seq_len + s]) == image_token_id) {
            if (vis_pos >= num_img)
              throw std::runtime_error(
                "Gemma4: more image placeholder positions in input_ids than"
                " vision_embeds tokens for batch item "
                + std::to_string(b));
            const float* src = vis_data + (b * num_img + vis_pos) * text_hidden;
            float*       dst = emb_data + (b * seq_len + s) * text_hidden;
            std::memcpy(dst, src, static_cast<size_t>(text_hidden) * sizeof(float));
            ++vis_pos;
          }
        }
      }

      // 4. Move to model device and cast to compute dtype.
      const DataType compute_dt = _decoder->output_type();
      inputs_embeds = inputs_embeds_cpu.to(_model->device()).to(compute_dt);
    }


    // =========================================================================
    // Gemma4Replica::generate
    //
    // Two-phase multimodal generation:
    //   Phase 1 – prefill KV-cache with merged text+vision embeddings.
    //   Phase 2 – autoregressive decode (greedy / sampling / beam search).
    //
    // Follows the same design as VibeVoiceAsrReplica::generate.
    // =========================================================================
    std::vector<Gemma4GenerationResult>
    Gemma4Replica::generate(const StorageView& input_ids,
                             const StorageView& vision_embeds,
                             const Gemma4GenerationOptions& options) {

      const auto scoped_device_setter = _model->get_scoped_device_setter();

      const dim_t batch      = input_ids.dim(0);
      const dim_t prefix_len = input_ids.dim(1);

      const Vocabulary& vocab  = _model->get_vocabulary();
      const size_t      eos_id = vocab.eos_id();

      const dim_t beam_size = static_cast<dim_t>(std::max<size_t>(1, options.beam_size));
      const dim_t num_hyp   = static_cast<dim_t>(std::max<size_t>(1, options.num_hypotheses));
      const size_t max_candidates = std::max(
        static_cast<size_t>(num_hyp),
        static_cast<size_t>(std::round(static_cast<float>(beam_size) * options.patience)));

      // Resolve image_token_id: use the caller-supplied value or fall back to
      // the value stored in the model's config.json.
      const size_t image_token_id = options.image_token_id != 0
                                      ? options.image_token_id
                                      : _model->get_image_token_id();

      // -----------------------------------------------------------------
      // Phase 1: build merged inputs_embeds and KV-cache prefill
      // -----------------------------------------------------------------
      StorageView inputs_embeds(_model->device());
      _build_inputs_embeds(input_ids, vision_embeds, image_token_id, inputs_embeds);

      layers::DecoderState state = _decoder->initial_state();

      StorageView prefix_lengths({batch}, DataType::INT32, Device::CPU);
      for (dim_t b = 0; b < batch; ++b)
        prefix_lengths.data<int32_t>()[b] = static_cast<int32_t>(prefix_len);
      prefix_lengths = prefix_lengths.to(_model->device());

      const DataType compute_dt = _decoder->output_type();
      StorageView prefix_logits(compute_dt, _model->device());
      // Build PLE-safe input_ids for the token-side per-layer embedding lookup.
      // HF replaces image placeholder positions with pad_token_id (0) before
      // calling embed_tokens_per_layer, so that vision positions contribute
      // the PAD token's per-layer embedding (not the out-of-range image token ID).
      // See Gemma4ForConditionalGeneration.forward(): llm_input_ids[multimodal_mask] = pad_token_id.
      StorageView ple_input_ids(Device::CPU, DataType::INT32);
      {
        StorageView ids_cpu = input_ids.to(Device::CPU);
        ple_input_ids = ids_cpu;  // copy
        int32_t* ids_data = ple_input_ids.data<int32_t>();
        const dim_t seq = ple_input_ids.dim(1);
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t s = 0; s < seq; ++s)
            if (static_cast<size_t>(ids_data[b * seq + s]) == image_token_id)
              ids_data[b * seq + s] = 0;  // replace with PAD token ID (0)
      }
      const StorageView ple_input_ids_dev = ple_input_ids.to(_model->device());
      _decoder->forward_with_embeds(inputs_embeds, ple_input_ids_dev, prefix_lengths, state, prefix_logits);
      inputs_embeds.release();

      const dim_t vocab_size = prefix_logits.dim(2);

      // Slice logit at the last prefix position → (batch, vocab_size)
      StorageView last_logits(compute_dt, _model->device());
      {
        StorageView tmp(compute_dt, _model->device());
        ops::Slide(1, prefix_len - 1, 1)(prefix_logits, tmp);
        tmp.reshape({batch, vocab_size});
        last_logits = std::move(tmp);
      }
      prefix_logits.release();

      // -----------------------------------------------------------------
      // Helper: repetition penalty applied in-place on CPU float32 logits.
      // -----------------------------------------------------------------
      auto apply_rep_penalty = [&](StorageView& logits,
                                   const std::vector<std::vector<size_t>>& token_lists) {
        if (options.repetition_penalty == 1.0f || token_lists.empty())
          return;
        const Device   dev = logits.device();
        const DataType dt  = logits.dtype();
        StorageView cpu_f32 = logits.to(Device::CPU);
        if (dt != DataType::FLOAT32)
          cpu_f32 = cpu_f32.to(DataType::FLOAT32);
        float* d = cpu_f32.data<float>();
        const dim_t n = cpu_f32.dim(0);
        const dim_t v = cpu_f32.dim(1);
        for (dim_t i = 0; i < n; ++i) {
          const auto& ids = token_lists[
            static_cast<size_t>(i) < token_lists.size()
              ? static_cast<size_t>(i)
              : token_lists.size() - 1];
          for (size_t id : ids) {
            if (static_cast<dim_t>(id) < v) {
              float& logit = d[i * v + id];
              if (logit > 0.0f) logit /= options.repetition_penalty;
              else               logit *= options.repetition_penalty;
            }
          }
        }
        if (dt != DataType::FLOAT32)
          cpu_f32 = cpu_f32.to(dt);
        logits = cpu_f32.to(dev);
      };

      // =================================================================
      // GREEDY / SAMPLING  (beam_size == 1)
      // =================================================================
      if (beam_size == 1) {
        const bool use_random = (options.sampling_temperature != 1.0f
                                 || options.sampling_topk != 1);
        std::unique_ptr<Sampler> sampler_ptr;
        if (use_random)
          sampler_ptr = std::make_unique<RandomSampler>(
            static_cast<dim_t>(options.sampling_topk),
            /*topp=*/1.0f,
            options.sampling_temperature);
        else
          sampler_ptr = std::make_unique<BestSampler>();

        std::vector<std::vector<size_t>> generated(static_cast<size_t>(batch));
        std::vector<float> cum_log_probs(static_cast<size_t>(batch), 0.0f);
        std::vector<bool>  done(static_cast<size_t>(batch), false);

        auto do_sample = [&](StorageView logits) {
          apply_rep_penalty(logits, generated);

          StorageView log_probs(_model->device());
          if (options.return_scores)
            ops::LogSoftMax()(logits, log_probs);

          StorageView sampled_ids(DataType::INT32, Device::CPU);
          StorageView sampled_scores(logits.dtype(), Device::CPU);
          (*sampler_ptr)(logits, sampled_ids, sampled_scores, /*num_samples=*/1);

          StorageView lp_cpu_f32;
          if (options.return_scores && log_probs) {
            lp_cpu_f32 = log_probs.to(Device::CPU);
            if (lp_cpu_f32.dtype() != DataType::FLOAT32)
              lp_cpu_f32 = lp_cpu_f32.to(DataType::FLOAT32);
          }

          for (dim_t b = 0; b < batch; ++b) {
            if (done[static_cast<size_t>(b)]) continue;
            const size_t id =
              static_cast<size_t>(sampled_ids.data<int32_t>()[b]);
            generated[static_cast<size_t>(b)].push_back(id);
            if (options.return_scores && lp_cpu_f32)
              cum_log_probs[static_cast<size_t>(b)] +=
                lp_cpu_f32.data<float>()[b * vocab_size + static_cast<dim_t>(id)];
            if (id == eos_id)
              done[static_cast<size_t>(b)] = true;
          }
        };

        do_sample(std::move(last_logits));

        StorageView step_logits(compute_dt, _model->device());
        for (size_t t = 1; t < options.max_new_tokens; ++t) {
          if (std::all_of(done.begin(), done.end(), [](bool d) { return d; }))
            break;

          StorageView step_ids({batch}, DataType::INT32, Device::CPU);
          for (dim_t b = 0; b < batch; ++b)
            step_ids.data<int32_t>()[b] = done[static_cast<size_t>(b)]
              ? static_cast<int32_t>(eos_id)
              : static_cast<int32_t>(generated[static_cast<size_t>(b)].back());
          step_ids = step_ids.to(_model->device());

          (*_decoder)(
            static_cast<dim_t>(prefix_len + t - 1),
            step_ids, state, &step_logits);

          do_sample(step_logits);
        }

        std::vector<Gemma4GenerationResult> results(static_cast<size_t>(batch));
        for (dim_t b = 0; b < batch; ++b) {
          Gemma4GenerationResult res;
          std::vector<std::string> tokens;
          tokens.reserve(generated[static_cast<size_t>(b)].size());
          for (size_t id : generated[static_cast<size_t>(b)])
            tokens.push_back(vocab.to_token(id));
          res.sequences     = {std::move(tokens)};
          res.sequences_ids = {generated[static_cast<size_t>(b)]};
          if (options.return_scores)
            res.scores = {cum_log_probs[static_cast<size_t>(b)]};
          results[static_cast<size_t>(b)] = std::move(res);
        }
        return results;
      }

      // =================================================================
      // BEAM SEARCH  (beam_size > 1)
      // =================================================================
      const dim_t num_candidates = beam_size * 2;

      StorageView init_log_probs(compute_dt, _model->device());
      ops::LogSoftMax()(last_logits, init_log_probs);
      last_logits.release();

      StorageView init_scores_dev(DataType::FLOAT32, _model->device()),
                  init_ids_dev(DataType::INT32, _model->device());
      {
        const ops::TopK topk_init(beam_size);
        topk_init(init_log_probs, init_scores_dev, init_ids_dev);
      }
      init_log_probs.release();

      StorageView init_scores = init_scores_dev.to(Device::CPU);
      if (init_scores.dtype() != DataType::FLOAT32)
        init_scores = init_scores.to(DataType::FLOAT32);
      StorageView init_ids = init_ids_dev.to(Device::CPU);
      init_scores_dev.release();
      init_ids_dev.release();

      struct Hypothesis { std::vector<size_t> tokens; float score; };

      std::vector<std::vector<std::vector<size_t>>> beam_tokens(
        static_cast<size_t>(batch),
        std::vector<std::vector<size_t>>(static_cast<size_t>(beam_size)));
      std::vector<float> beam_scores(
        static_cast<size_t>(batch * beam_size),
        std::numeric_limits<float>::lowest());
      std::vector<std::vector<Hypothesis>> completed(static_cast<size_t>(batch));
      std::vector<bool> batch_done(static_cast<size_t>(batch), false);
      const float neg_inf = std::numeric_limits<float>::lowest();

      for (dim_t b = 0; b < batch; ++b) {
        for (dim_t k = 0; k < beam_size; ++k) {
          const size_t tok = static_cast<size_t>(
            init_ids.data<int32_t>()[b * beam_size + k]);
          const float  sc  = init_scores.data<float>()[b * beam_size + k];
          beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)] = {tok};
          if (tok == eos_id) {
            const float lp = (options.length_penalty != 0.0f)
              ? std::pow(1.0f, options.length_penalty) : 1.0f;
            completed[static_cast<size_t>(b)].push_back({{tok}, sc / lp});
          } else {
            beam_scores[static_cast<size_t>(b * beam_size + k)] = sc;
          }
        }
        if (completed[static_cast<size_t>(b)].size() >= max_candidates) {
          std::sort(completed[static_cast<size_t>(b)].begin(),
                    completed[static_cast<size_t>(b)].end(),
                    [](const auto& a, const auto& b) { return a.score > b.score; });
          float best_active = neg_inf;
          for (dim_t k = 0; k < beam_size; ++k)
            best_active = std::max(best_active,
              beam_scores[static_cast<size_t>(b * beam_size + k)]);
          if (best_active < completed[static_cast<size_t>(b)][num_hyp - 1].score)
            batch_done[static_cast<size_t>(b)] = true;
        }
      }

      // Expand KV cache: (batch,) → (batch * beam_size,)
      static_cast<layers::Decoder*>(_decoder.get())->replicate_state(state, beam_size);

      StorageView step_logits(compute_dt, _model->device());

      for (size_t t = 1; t < options.max_new_tokens; ++t) {
        if (std::all_of(batch_done.begin(), batch_done.end(),
                        [](bool d) { return d; }))
          break;

        StorageView step_ids({batch * beam_size}, DataType::INT32, Device::CPU);
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t k = 0; k < beam_size; ++k) {
            const auto& toks = beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)];
            const size_t last_tok = toks.empty() ? eos_id : toks.back();
            step_ids.data<int32_t>()[b * beam_size + k] =
              static_cast<int32_t>(last_tok);
          }
        step_ids = step_ids.to(_model->device());

        (*_decoder)(
          static_cast<dim_t>(prefix_len + t - 1),
          step_ids, state, &step_logits);

        if (options.repetition_penalty != 1.0f) {
          std::vector<std::vector<size_t>> flat_tokens(
            static_cast<size_t>(batch * beam_size));
          for (dim_t b = 0; b < batch; ++b)
            for (dim_t k = 0; k < beam_size; ++k)
              flat_tokens[static_cast<size_t>(b * beam_size + k)] =
                beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)];
          apply_rep_penalty(step_logits, flat_tokens);
        }

        ops::LogSoftMax()(step_logits);

        StorageView lp_cpu = step_logits.to(Device::CPU);
        if (lp_cpu.dtype() != DataType::FLOAT32)
          lp_cpu = lp_cpu.to(DataType::FLOAT32);
        float* lp_data = lp_cpu.data<float>();

        for (dim_t i = 0; i < batch * beam_size; ++i) {
          const float bs = beam_scores[static_cast<size_t>(i)];
          if (bs <= neg_inf / 2.0f) {
            for (dim_t v = 0; v < vocab_size; ++v)
              lp_data[i * vocab_size + v] = neg_inf;
          } else {
            for (dim_t v = 0; v < vocab_size; ++v)
              lp_data[i * vocab_size + v] += bs;
          }
        }

        lp_cpu.reshape({batch, beam_size * vocab_size});
        StorageView topk_vals(DataType::FLOAT32, Device::CPU);
        StorageView topk_flat_ids(DataType::INT32, Device::CPU);
        {
          const ops::TopK topk_cand(num_candidates);
          topk_cand(lp_cpu, topk_vals, topk_flat_ids);
        }

        const float*   tv_data = topk_vals.data<float>();
        const int32_t* ti_data = topk_flat_ids.data<int32_t>();

        std::vector<std::vector<std::vector<size_t>>> new_beam_tokens(
          static_cast<size_t>(batch),
          std::vector<std::vector<size_t>>(static_cast<size_t>(beam_size)));
        std::vector<float>   new_beam_scores(
          static_cast<size_t>(batch * beam_size), neg_inf);
        std::vector<int32_t> gather_idx(
          static_cast<size_t>(batch * beam_size), 0);

        for (dim_t b = 0; b < batch; ++b) {
          if (batch_done[static_cast<size_t>(b)]) {
            for (dim_t k = 0; k < beam_size; ++k) {
              new_beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)] =
                beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)];
              gather_idx[static_cast<size_t>(b * beam_size + k)] =
                static_cast<int32_t>(b * beam_size + k);
            }
            continue;
          }

          dim_t next_k = 0;
          for (dim_t ci = 0; ci < num_candidates && next_k < beam_size; ++ci) {
            const float   cand_score  = tv_data[b * num_candidates + ci];
            const int32_t flat_id     = ti_data[b * num_candidates + ci];
            const dim_t   beam_origin = flat_id / vocab_size;
            const dim_t   token_id    = flat_id % vocab_size;

            if (cand_score <= neg_inf / 2.0f)
              break;

            if (token_id == static_cast<dim_t>(eos_id)) {
              const float seq_len =
                static_cast<float>(
                  beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(beam_origin)].size() + 1);
              const float lp = (options.length_penalty != 0.0f)
                ? std::pow(seq_len, options.length_penalty) : 1.0f;
              completed[static_cast<size_t>(b)].push_back(
                {beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(beam_origin)],
                 cand_score / lp});
            } else {
              new_beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(next_k)] =
                beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(beam_origin)];
              new_beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(next_k)].push_back(
                static_cast<size_t>(token_id));
              new_beam_scores[static_cast<size_t>(b * beam_size + next_k)] = cand_score;
              gather_idx[static_cast<size_t>(b * beam_size + next_k)] =
                static_cast<int32_t>(b * beam_size + beam_origin);
              ++next_k;
            }
          }

          for (; next_k < beam_size; ++next_k) {
            new_beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(next_k)] = {};
            gather_idx[static_cast<size_t>(b * beam_size + next_k)] =
              static_cast<int32_t>(b * beam_size);
          }

          if (completed[static_cast<size_t>(b)].size() >= max_candidates) {
            std::sort(completed[static_cast<size_t>(b)].begin(),
                      completed[static_cast<size_t>(b)].end(),
                      [](const auto& a, const auto& b) { return a.score > b.score; });
            const float best_active =
              new_beam_scores[static_cast<size_t>(b * beam_size)];
            if (best_active <= neg_inf / 2.0f ||
                best_active < completed[static_cast<size_t>(b)][num_hyp - 1].score)
              batch_done[static_cast<size_t>(b)] = true;
          }
        }

        StorageView gather_indices({batch * beam_size}, DataType::INT32, Device::CPU);
        std::copy(gather_idx.begin(), gather_idx.end(),
                  gather_indices.data<int32_t>());
        gather_indices = gather_indices.to(_model->device());
        _decoder->update_state(state, std::move(gather_indices), beam_size,
                               /*alive_batches=*/nullptr);

        beam_tokens  = std::move(new_beam_tokens);
        beam_scores  = std::move(new_beam_scores);
      }

      // Force-finish any active beams.
      for (dim_t b = 0; b < batch; ++b) {
        if (batch_done[static_cast<size_t>(b)]) continue;
        for (dim_t k = 0; k < beam_size; ++k) {
          const float sc = beam_scores[static_cast<size_t>(b * beam_size + k)];
          if (sc <= neg_inf / 2.0f ||
              beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)].empty())
            continue;
          const float seq_len =
            static_cast<float>(
              beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)].size());
          const float lp = (options.length_penalty != 0.0f)
            ? std::pow(seq_len, options.length_penalty) : 1.0f;
          completed[static_cast<size_t>(b)].push_back(
            {beam_tokens[static_cast<size_t>(b)][static_cast<size_t>(k)], sc / lp});
        }
      }

      std::vector<Gemma4GenerationResult> results(static_cast<size_t>(batch));
      for (dim_t b = 0; b < batch; ++b) {
        std::sort(completed[static_cast<size_t>(b)].begin(),
                  completed[static_cast<size_t>(b)].end(),
                  [](const auto& a, const auto& b) { return a.score > b.score; });
        Gemma4GenerationResult res;
        const size_t n =
          std::min(completed[static_cast<size_t>(b)].size(),
                   static_cast<size_t>(num_hyp));
        res.sequences.reserve(n);
        res.sequences_ids.reserve(n);
        if (options.return_scores) res.scores.reserve(n);
        for (size_t h = 0; h < n; ++h) {
          std::vector<std::string> tokens;
          tokens.reserve(completed[static_cast<size_t>(b)][h].tokens.size());
          for (size_t id : completed[static_cast<size_t>(b)][h].tokens)
            tokens.push_back(vocab.to_token(id));
          res.sequences.push_back(std::move(tokens));
          res.sequences_ids.push_back(completed[static_cast<size_t>(b)][h].tokens);
          if (options.return_scores)
            res.scores.push_back(completed[static_cast<size_t>(b)][h].score);
        }
        results[static_cast<size_t>(b)] = std::move(res);
      }
      return results;
    }


    // =========================================================================
    // Gemma4  (public API)
    // =========================================================================

    std::future<StorageView>
    Gemma4::encode_vision(const StorageView& pixel_values,
                           const StorageView& pixel_position_ids) {
      return post<StorageView>([pixel_values, pixel_position_ids](Gemma4Replica& replica) {
        return replica.encode_vision(pixel_values, pixel_position_ids);
      });
    }

    std::vector<std::future<Gemma4GenerationResult>>
    Gemma4::generate(const StorageView& input_ids,
                     const StorageView& vision_embeds,
                     Gemma4GenerationOptions options) {
      const size_t batch_size = static_cast<size_t>(input_ids.dim(0));
      return post_batch<Gemma4GenerationResult>(
        [input_ids    = input_ids.sync_copy(),
         vision_embeds = vision_embeds.sync_copy(),
         options      = std::move(options)]
        (Gemma4Replica& replica) mutable {
          return replica.generate(input_ids, vision_embeds, options);
        },
        batch_size);
    }


  }  // namespace models
}  // namespace ctranslate2
