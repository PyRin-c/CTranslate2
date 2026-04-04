// Qwen3-ASR CTranslate2 backend implementation.
//
// Architecture (Qwen3ASRForConditionalGeneration):
//
//   Audio Encoder:
//     3× Conv2D (3×3, stride=2, pad=1, GELU) → [B, 480, 16, T/8]
//     Linear (7680 → d_model, no bias) = conv_out
//     N× TransformerEncoderLayer (Whisper-style, Q/K/V bias, GELU FFN)
//     Final LayerNorm (ln_post)
//     Projector: proj1 (Linear+GELU) → proj2 (Linear)
//
//   LLM Decoder (Qwen3):
//     28-layer decoder-only Transformer
//     GQA: 16Q / 8KV heads, head_dim=128
//     QK Normalization (per-head RMSNorm on Q and K)
//     Interleaved MRoPE: sections=[24,20,20], stride-3, theta=1,000,000
//     SwiGLU FFN
//
//   Audio-Text Fusion:
//     Token embeddings built from text tokens + audio_token_id placeholders
//     Placeholders replaced with audio encoder output (in-place)
//     Decoder operates on the fused embeddings via decode_from_embeds()
//
// Weight name convention (CT2 model binary):
//   audio_encoder/conv_stem_0/{weight,bias}
//   audio_encoder/conv_stem_1/{weight,bias}
//   audio_encoder/conv_stem_2/{weight,bias}
//   audio_encoder/conv_out/weight
//   audio_encoder/layer_N/self_attention/...
//   audio_encoder/layer_N/ffn/...
//   audio_encoder/layer_norm/{gamma,beta}
//   audio_encoder/proj1/{weight,bias}
//   audio_encoder/proj2/{weight,bias}
//   decoder/...  (standard TransformerDecoder layout)

#include "ctranslate2/models/qwen3_asr.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "ctranslate2/decoding.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/transformer.h"
#include "ctranslate2/models/model_reader.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/sampling.h"
#include "../dispatch.h"

#ifdef CT2_WITH_CUDA
#  include "cuda/utils.h"
#endif

namespace ctranslate2 {
  namespace models {

    // -----------------------------------------------------------------------
    // Qwen3AudioConvStem
    //
    // 3× Conv2D (3×3, stride=2, pad=1, GELU):
    //   [B, 128, T] → add channel → [B, 1, 128, T]
    //                → conv2d×3  → [B, 480, 16, T/8]
    //                → permute   → [B, T/8, 480, 16]
    //                → reshape   → [B, T/8, 7680]
    //                → linear    → [B, T/8, d_model]
    // -----------------------------------------------------------------------
    class Qwen3AudioConvStem {
    public:
      Qwen3AudioConvStem(const Model& model, const std::string& scope)
        : _conv0_op(2, 2, 1, 1)
        , _conv1_op(2, 2, 1, 1)
        , _conv2_op(2, 2, 1, 1)
        , _conv_out(model, scope + "/conv_out")
      {
        _w0 = &model.get_variable(scope + "/conv_stem_0/weight");
        _b0 = &model.get_variable(scope + "/conv_stem_0/bias");
        _w1 = &model.get_variable(scope + "/conv_stem_1/weight");
        _b1 = &model.get_variable(scope + "/conv_stem_1/bias");
        _w2 = &model.get_variable(scope + "/conv_stem_2/weight");
        _b2 = &model.get_variable(scope + "/conv_stem_2/bias");
      }

      // features: [B, 128, T]
      // output:   [B, N_tokens, d_model]
      void operator()(const StorageView& features, StorageView& output) const {
        const Device dev = features.device();
        const DataType dt = features.dtype();

        const dim_t B = features.dim(0);
        const dim_t F = features.dim(1);  // 128
        const dim_t T = features.dim(2);

        // [B, 128, T] → [B, 1, 128, T]
        StorageView x = features;
        x.reshape({B, 1, F, T});

        // Weights must be on device and in correct dtype
        StorageView w0 = _w0->to(dt).to(dev);
        StorageView b0 = _b0->to(dt).to(dev);
        StorageView w1 = _w1->to(dt).to(dev);
        StorageView b1 = _b1->to(dt).to(dev);
        StorageView w2 = _w2->to(dt).to(dev);
        StorageView b2 = _b2->to(dt).to(dev);

        // Conv2D × 3 with GELU activation
        {
          StorageView tmp(dt, dev);
          _conv0_op(x, w0, b0, tmp);
          ops::GELU()(tmp, x);
        }
        {
          StorageView tmp(dt, dev);
          _conv1_op(x, w1, b1, tmp);
          ops::GELU()(tmp, x);
        }
        {
          StorageView tmp(dt, dev);
          _conv2_op(x, w2, b2, tmp);
          ops::GELU()(tmp, x);
        }

        // x: [B, 480, H, W]  where H≈16, W≈T/8
        // Permute to [B, W, 480, H] then reshape to [B, W, 480*H]
        StorageView x_perm(dt, dev);
        ops::Transpose({0, 3, 1, 2})(x, x_perm);

        const dim_t N_tokens = x_perm.dim(1);
        const dim_t flat_dim = x_perm.dim(2) * x_perm.dim(3);
        x_perm.reshape({B, N_tokens, flat_dim});

        // Linear projection
        _conv_out(x_perm, output);
      }

    private:
      const ops::Conv2D _conv0_op;
      const ops::Conv2D _conv1_op;
      const ops::Conv2D _conv2_op;

      const StorageView* _w0;
      const StorageView* _b0;
      const StorageView* _w1;
      const StorageView* _b1;
      const StorageView* _w2;
      const StorageView* _b2;

      const layers::Dense _conv_out;
    };

    // -----------------------------------------------------------------------
    // Qwen3AudioEncoder
    //
    // Conv stem → sinusoidal position encoding
    //           → Windowed Transformer (block-diagonal mask)
    //           → Final LayerNorm
    //           → Projector (proj1 GELU + proj2)
    // -----------------------------------------------------------------------
    class Qwen3AudioEncoder {
    public:
      Qwen3AudioEncoder(const Model& model, const std::string& scope,
                        dim_t n_window_infer, dim_t n_window)
        : _device(model.device())
        , _conv_stem(model, scope)
        , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 16))
        , _n_window_infer(n_window_infer)
        , _n_window(n_window)
        , _layers(layers::build_layers_list<const layers::TransformerEncoderLayer>(
                    model,
                    scope + "/layer",
                    _num_heads,
                    /*pre_norm=*/true,
                    ops::ActivationType::GELU))
        , _layer_norm(model, scope + "/layer_norm")
        , _gelu_act(ops::ActivationType::GELU)
        , _proj1(model, scope + "/proj1", &_gelu_act)
        , _proj2(model, scope + "/proj2")
      {
      }

      DataType output_type() const {
        return _proj2.output_type();
      }

      // features: [B, 128, T]
      // output:   [B, N_audio_tokens, output_dim]
      // window_tokens: audio token window size (-1 = offline/auto)
      void operator()(const StorageView& features, StorageView& output,
                      int window_tokens = -1) {
        PROFILE("Qwen3AudioEncoder");

        const Device dev = _device;
        const DataType dt = _layer_norm.output_type();

        const dim_t B = features.dim(0);
        const dim_t F = features.dim(1);  // 128
        const dim_t T = features.dim(2);  // mel frames

        // 1. Chunk-based Conv2D stem.
        //
        // The reference implementation splits the mel spectrogram into chunks
        // of (n_window * 2) frames and processes each chunk independently.
        // This causes position encoding to reset at every chunk boundary and
        // changes the total token count vs. processing the full sequence at once.
        //
        // For T=1000, chunk_mel=100:  10 chunks × 13 tokens = 130 tokens.
        // Processing all 1000 frames at once gives ceil(1000/8) = 125 tokens.
        const dim_t chunk_mel   = _n_window * 2;          // 100 for n_window=50
        const dim_t num_chunks  = (T + chunk_mel - 1) / chunk_mel;

        // Build zero-filled chunk batch on CPU float32: [num_chunks*B, F, chunk_mel]
        StorageView chunked({num_chunks * B, F, chunk_mel}, DataType::FLOAT32);
        std::fill(chunked.data<float>(),
                  chunked.data<float>() + chunked.size(), 0.0f);

        // Copy mel slices into chunked (move features to CPU for slicing).
        {
          StorageView feats_cpu = features.to(Device::CPU);
          if (feats_cpu.dtype() != DataType::FLOAT32)
            feats_cpu = feats_cpu.to(DataType::FLOAT32);
          const float* src = feats_cpu.data<float>();
          float*       dst = chunked.data<float>();
          for (dim_t b = 0; b < B; ++b) {
            for (dim_t c = 0; c < num_chunks; ++c) {
              const dim_t t_start = c * chunk_mel;
              const dim_t t_len   = std::min(chunk_mel, T - t_start);
              for (dim_t f = 0; f < F; ++f) {
                const float* row_src = src + (b * F + f) * T + t_start;
                float*       row_dst = dst + ((b * num_chunks + c) * F + f) * chunk_mel;
                std::copy(row_src, row_src + t_len, row_dst);
                // Remaining elements are already zero (last-chunk padding).
              }
            }
          }
        }

        // Cast to model dtype and move to compute device.
        StorageView chunked_dev = chunked.to(dt).to(dev);

        // Apply conv stem to the whole chunk batch.
        // Qwen3AudioConvStem::operator() expects [batch, F, W] and internally
        // reshapes to [batch, 1, F, W], so passing [num_chunks*B, F, chunk_mel]
        // processes all chunks in one efficient batched conv call.
        StorageView x(dt, dev);
        _conv_stem(chunked_dev, x);
        // x: [num_chunks*B, chunk_tokens, d_model]  (chunk_tokens≈13)

        const dim_t chunk_tokens = x.dim(1);
        const dim_t d_model      = x.dim(2);

        // 2. Whisper-style sinusoidal PE, reset at each chunk boundary.
        //
        // PE shape is [1, chunk_tokens, d_model]; it broadcasts over the
        // num_chunks*B leading dimension, so every chunk gets positions 0..12.
        _add_chunk_sinusoidal_positions(x, chunk_tokens, d_model);

        // 3. Compute the number of valid tokens in the (possibly short) last chunk
        //    and reshape x to [B, total_tokens, d_model].
        const dim_t last_chunk_mel = T - (num_chunks - 1) * chunk_mel;
        dim_t last_chunk_tokens = last_chunk_mel;
        last_chunk_tokens = (last_chunk_tokens + 1) / 2;
        last_chunk_tokens = (last_chunk_tokens + 1) / 2;
        last_chunk_tokens = (last_chunk_tokens + 1) / 2;
        const dim_t total_tokens = (num_chunks - 1) * chunk_tokens + last_chunk_tokens;

        if (last_chunk_tokens == chunk_tokens) {
          // No padding to remove — simple reshape.
          x.reshape({B, total_tokens, d_model});
        } else {
          // Trim padding tokens from the last chunk.  Work on CPU to avoid
          // device-specific copy complexity.
          StorageView x_f32 = x.to(Device::CPU).to(DataType::FLOAT32);
          StorageView trimmed({B, total_tokens, d_model}, DataType::FLOAT32);
          const float* xp = x_f32.data<float>();
          float*       tp = trimmed.data<float>();
          for (dim_t b = 0; b < B; ++b) {
            dim_t tok_off = 0;
            for (dim_t c = 0; c < num_chunks; ++c) {
              const dim_t valid = (c == num_chunks - 1) ? last_chunk_tokens
                                                        : chunk_tokens;
              const float* s   = xp + (b * num_chunks + c) * chunk_tokens * d_model;
              float*       dst2 = tp + (b * total_tokens + tok_off) * d_model;
              std::copy(s, s + valid * d_model, dst2);
              tok_off += valid;
            }
          }
          x = trimmed.to(dt).to(dev);
        }

        // 4. Windowed attention mask (block-diagonal).
        const dim_t seq_len    = x.dim(1);
        const dim_t eff_window = (window_tokens > 0)
          ? static_cast<dim_t>(window_tokens)
          : _compute_window_tokens();

        std::unique_ptr<StorageView> attn_mask;
        if (eff_window > 0 && eff_window < seq_len) {
          StorageView mask_f32   = _make_windowed_mask(seq_len, eff_window);
          StorageView mask_typed = mask_f32.to(dt).to(dev);
          attn_mask = std::make_unique<StorageView>(std::move(mask_typed));
        }

        // 5. Transformer encoder layers.
        StorageView layer_out(dt, dev);
        for (const auto& layer : _layers) {
          (*layer)(x, /*lengths=*/nullptr, layer_out, /*padder=*/nullptr,
                   attn_mask.get());
          x = std::move(layer_out);
        }

        // 6. Final LayerNorm.
        _layer_norm(x, layer_out);
        x = std::move(layer_out);

        // 7. Projector: proj1 (GELU) → proj2.
        StorageView p1(dt, dev);
        _proj1(x, p1);
        _proj2(p1, output);
      }

    private:
      // Number of audio tokens for n_window_infer mel frames.
      // Each stride-2 conv uses ceil division.
      dim_t _compute_window_tokens() const {
        dim_t f = _n_window_infer;
        f = (f + 1) / 2;
        f = (f + 1) / 2;
        f = (f + 1) / 2;
        return f;
      }

      // Add Whisper-style sinusoidal position encodings to x in-place.
      //
      // x has shape [num_chunks*B, chunk_tokens, d_model].
      // PE shape is [1, chunk_tokens, d_model], which broadcasts over the
      // leading dimension so that every chunk gets the same positions 0..T-1
      // (i.e. positions reset at each chunk boundary — matching the reference).
      //
      // Formula (Whisper / qwen-asr SinusoidsPositionEmbedding):
      //   log_timescale_increment = log(10000) / (d_model/2 - 1)
      //   inv_timescales[i]       = exp(-log_timescale_increment * i)
      //   pe[pos, :d/2]           = sin(pos * inv_timescales)
      //   pe[pos, d/2:]           = cos(pos * inv_timescales)
      //
      // This is a concat (not interleaved) layout.
      void _add_chunk_sinusoidal_positions(StorageView& x,
                                           dim_t chunk_tokens,
                                           dim_t d_model) const {
        // x: [num_chunks*B, chunk_tokens, d_model]
        const dim_t batch_size = x.dim(0);
        const Device  dev = x.device();
        const DataType dt = x.dtype();

        // Compute PE for one chunk: [chunk_tokens, d_model] on CPU float32.
        const dim_t half       = d_model / 2;
        const float log_ts_inc = std::log(10000.0f) / static_cast<float>(half - 1);

        StorageView pe_one({chunk_tokens, d_model}, DataType::FLOAT32);
        float* pe_data = pe_one.data<float>();
        for (dim_t pos = 0; pos < chunk_tokens; ++pos) {
          for (dim_t i = 0; i < half; ++i) {
            const float inv_ts = std::exp(-log_ts_inc * static_cast<float>(i));
            const float angle  = static_cast<float>(pos) * inv_ts;
            pe_data[pos * d_model + i]        = std::sin(angle);
            pe_data[pos * d_model + half + i] = std::cos(angle);
          }
        }

        // Tile PE to [num_chunks*B, chunk_tokens, d_model] so that ops::Add
        // does not need to broadcast (CT2 Add expects identical shapes).
        const dim_t pe_elems = chunk_tokens * d_model;
        StorageView pe_tiled({batch_size, chunk_tokens, d_model}, DataType::FLOAT32);
        float* tiled_ptr = pe_tiled.data<float>();
        for (dim_t i = 0; i < batch_size; ++i)
          std::copy(pe_data, pe_data + pe_elems, tiled_ptr + i * pe_elems);

        StorageView pe_dev = pe_tiled.to(dt).to(dev);
        ops::Add()(x, pe_dev, x);
      }

      // Block-diagonal attention mask: 0 within block, -1e9 across blocks.
      // Returns [1, 1, seq_len, seq_len] float32 on CPU.
      //
      // Complexity: O(seq_len²) time and memory.
      // In practice this is bounded by the model's intended input length:
      //   default n_window_infer=800 mel frames → eff_window=100 audio tokens
      //   default n_window=50 → chunk_mel=100 frames → ~13 tokens/chunk
      //   For 8 seconds of audio (T≈800): seq_len≈104 → mask ≈ 43 KB (negligible).
      //   For very long audio (e.g. T=4000, 40s): seq_len≈520 → mask ≈ 1 MB.
      // Inputs significantly longer than n_window_infer are outside the intended
      // operating range and should be split into multiple API calls.
      static StorageView _make_windowed_mask(dim_t seq_len, dim_t window_tokens) {
        const float NEG_INF = -1e9f;
        StorageView mask({1, 1, seq_len, seq_len}, DataType::FLOAT32);
        float* data = mask.data<float>();
        for (dim_t i = 0; i < seq_len; ++i)
          for (dim_t j = 0; j < seq_len; ++j)
            data[i * seq_len + j] =
              ((i / window_tokens) == (j / window_tokens)) ? 0.0f : NEG_INF;
        return mask;
      }

      const Device _device;
      Qwen3AudioConvStem _conv_stem;
      const dim_t _num_heads;
      const dim_t _n_window_infer;
      const dim_t _n_window;

      const std::vector<std::unique_ptr<const layers::TransformerEncoderLayer>> _layers;
      const layers::LayerNorm _layer_norm;
      const ops::ActivationType _gelu_act;
      const layers::Dense _proj1;
      const layers::Dense _proj2;
    };

    // -----------------------------------------------------------------------
    // Qwen3Decoder — thin public wrapper exposing decode_from_embeds.
    // -----------------------------------------------------------------------
    class Qwen3Decoder : public layers::TransformerDecoder {
    public:
      using layers::TransformerDecoder::TransformerDecoder;

      // KV-cache prefill from pre-computed embeddings (skips embedding lookup).
      void forward_with_embeds(const StorageView& inputs_embeds,
                               const StorageView& lengths,
                               layers::DecoderState& state,
                               StorageView& logits) {
        decode_from_embeds(inputs_embeds, &lengths, /*step=*/0, state, &logits);
      }
    };

    // -----------------------------------------------------------------------
    // Pimpl
    // -----------------------------------------------------------------------
    struct Qwen3ASRReplica::Impl {
      std::unique_ptr<Qwen3AudioEncoder> audio_encoder;
      std::unique_ptr<Qwen3Decoder>      decoder;
    };

    // -----------------------------------------------------------------------
    // Qwen3ASRModel
    // -----------------------------------------------------------------------
    const Vocabulary& Qwen3ASRModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t Qwen3ASRModel::current_spec_revision() const {
      return 1;
    }

    bool Qwen3ASRModel::is_quantizable(const std::string& variable_name) const {
      // Conv2D stem weights are 4D tensors; ops::Conv2D does not support int8,
      // so exclude them from quantization.
      if (variable_name.find("conv_stem") != std::string::npos)
        return false;
      return Model::is_quantizable(variable_name);
    }

    bool Qwen3ASRModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name)
             && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> Qwen3ASRModel::clone() const {
      return std::make_unique<Qwen3ASRModel>(*this);
    }

    void Qwen3ASRModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = config.value("unk_token", "<|endoftext|>");
      vocab_info.bos_token = config.value("bos_token", "<|im_start|>");
      vocab_info.eos_token = config.value("eos_token", "<|im_end|>");

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error(
          "Qwen3ASRModel: Cannot load vocabulary from model directory");
    }

    // -----------------------------------------------------------------------
    // Qwen3ASRReplica
    // -----------------------------------------------------------------------
    std::unique_ptr<Qwen3ASRReplica>
    Qwen3ASRReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const Qwen3ASRModel*>(&model))
        throw std::invalid_argument("The model is not a Qwen3ASR model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr  = model.shared_from_this();
      const auto concrete   = std::static_pointer_cast<const Qwen3ASRModel>(model_ptr);
      return std::make_unique<Qwen3ASRReplica>(concrete);
    }

    Qwen3ASRReplica::Qwen3ASRReplica(
      const std::shared_ptr<const Qwen3ASRModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _impl(std::make_unique<Impl>())
    {
      const auto& cfg = model->config;
      const dim_t n_window_infer =
        static_cast<dim_t>(cfg.value("audio_n_window_infer", int32_t(800)));
      const dim_t n_window =
        static_cast<dim_t>(cfg.value("audio_n_window", int32_t(50)));

      _impl->audio_encoder = std::make_unique<Qwen3AudioEncoder>(
        *model, "audio_encoder", n_window_infer, n_window);

      _impl->decoder = std::make_unique<Qwen3Decoder>(*model, "decoder");
    }

    Qwen3ASRReplica::~Qwen3ASRReplica() = default;

    // -----------------------------------------------------------------------
    // encode
    // -----------------------------------------------------------------------
    StorageView Qwen3ASRReplica::encode(const StorageView& features,
                                         const Qwen3ASROptions& options) {
      PROFILE("Qwen3ASRReplica::encode");
      const auto scoped_device_setter = _model->get_scoped_device_setter();

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      StorageView audio_out(_impl->audio_encoder->output_type(), _model->device());
      (*_impl->audio_encoder)(features, audio_out, options.encoder_window_tokens);
      return audio_out;
    }

    // -----------------------------------------------------------------------
    // _build_inputs_embeds: embed tokens, replace audio_token_id with audio_features
    // -----------------------------------------------------------------------
    void Qwen3ASRReplica::_build_inputs_embeds(
      const StorageView& input_ids,      // (batch, seq_len) int32, CPU
      const StorageView& audio_features, // (batch, audio_seq, lm_hidden)
      const size_t audio_token_id,
      StorageView& inputs_embeds) const
    {
      const dim_t batch   = input_ids.dim(0);
      const dim_t seq_len = input_ids.dim(1);
      const dim_t hidden  = audio_features.dim(2);
      const Device device = _model->device();

      // Token embeddings in float32.
      // The Gather op requires ids on the same device as the embedding table,
      // and the output StorageView must have the embedding layer's dtype
      // (e.g. float16 for float16 models) so that Gather can write into it.
      layers::Embeddings embeddings(*_model, "decoder/embeddings");
      StorageView ids_dev = (input_ids.device() == device) ? input_ids : input_ids.to(device);
      StorageView raw_embeds(embeddings.output_type(), device);
      embeddings(ids_dev, raw_embeds);
      inputs_embeds = raw_embeds.to(DataType::FLOAT32);

      // Audio-token injection is done on CPU because it requires a scatter-style
      // write driven by a variable-length index list (audio_token_id positions),
      // which has no direct CTranslate2 GPU op equivalent.  The round-trip
      // (GPU → CPU for audio_features + embeddings, then CPU → GPU for the
      // fused result) is acceptable here because it occurs once per generate()
      // call before the decode loop, not inside the decode loop itself.
      // Future improvement: implement a custom CUDA scatter kernel.
      StorageView audio_cpu = audio_features.to(DataType::FLOAT32).to(Device::CPU);
      StorageView ids_cpu   = input_ids.to(Device::CPU);
      StorageView emb_cpu   = inputs_embeds.to(Device::CPU);

      const int32_t* ids_data = ids_cpu.data<int32_t>();
      float*         emb_data = emb_cpu.data<float>();
      const float*   aud_data = audio_cpu.data<float>();

      for (dim_t b = 0; b < batch; ++b) {
        // Count audio placeholder slots for this batch item
        dim_t num_slots = 0;
        for (dim_t s = 0; s < seq_len; ++s)
          if (static_cast<size_t>(ids_data[b * seq_len + s]) == audio_token_id)
            ++num_slots;

        const dim_t audio_len = audio_features.dim(1);
        if (num_slots != audio_len)
          throw std::runtime_error(
            "Qwen3ASRReplica::_build_inputs_embeds: batch item " +
            std::to_string(b) + " has " + std::to_string(num_slots) +
            " audio_token_id placeholders but audio_features has " +
            std::to_string(audio_len) + " audio tokens. "
            "Ensure input_ids has exactly one audio_token_id per audio token.");

        dim_t audio_pos = 0;
        for (dim_t s = 0; s < seq_len; ++s) {
          if (static_cast<size_t>(ids_data[b * seq_len + s]) == audio_token_id) {
            const float* src = aud_data + (b * audio_len + audio_pos) * hidden;
            float*       dst = emb_data + (b * seq_len + s) * hidden;
            std::copy(src, src + hidden, dst);
            ++audio_pos;
          }
        }
      }

      // Move result back to model device; decoder will cast to its compute dtype
      inputs_embeds = emb_cpu.to(device);
    }

    // -----------------------------------------------------------------------
    // generate: full ASR pipeline
    //   encode → build_inputs_embeds → prefill → autoregressive decode
    // -----------------------------------------------------------------------
    std::vector<Qwen3ASRResult>
    Qwen3ASRReplica::generate(const StorageView& features,
                               const std::vector<std::vector<size_t>>& prompts,
                               const Qwen3ASROptions& options) {
      PROFILE("Qwen3ASRReplica::generate");
      const auto scoped_device_setter = _model->get_scoped_device_setter();

#ifdef CT2_WITH_CUDA
      const cuda::UseTrueFp16GemmInScope use_true_fp16_gemm(false);
#endif

      const Device device  = _model->device();
      const dim_t  batch   = static_cast<dim_t>(prompts.size());
      const size_t audio_token_id = options.audio_token_id;

      // 1. Encode mel spectrogram
      StorageView audio_features = encode(features, options);

      // 2. Build padded input_ids on CPU (all prompts padded to the same length)
      StorageView input_ids = layers::make_sequence_inputs(prompts, Device::CPU);
      const dim_t prefix_len = input_ids.dim(1);

      // 3. Build fused embeddings: text + audio
      // _build_inputs_embeds always returns float32; cast to the model's compute
      // dtype so the decoder receives the expected type (e.g. float16).
      const DataType compute_dt = audio_features.dtype();
      StorageView inputs_embeds(device);
      _build_inputs_embeds(input_ids, audio_features, audio_token_id, inputs_embeds);
      if (inputs_embeds.dtype() != compute_dt)
        inputs_embeds = inputs_embeds.to(compute_dt);

      // 4. Prefix lengths (per batch item)
      StorageView prefix_lengths({batch}, DataType::INT32, Device::CPU);
      for (dim_t b = 0; b < batch; ++b)
        prefix_lengths.data<int32_t>()[b] =
          static_cast<int32_t>(prompts[b].size());
      prefix_lengths = prefix_lengths.to(device);

      // 5. Prefill: decoder forward on fused embeddings
      layers::DecoderState state = _impl->decoder->initial_state(/*iterative_decoding=*/true);
      StorageView prefix_logits(compute_dt, device);
      _impl->decoder->forward_with_embeds(
        inputs_embeds, prefix_lengths, state, prefix_logits);
      inputs_embeds.release();

      // prefix_logits: (batch, prefix_len, vocab_size)
      const dim_t vocab_size = prefix_logits.dim(2);

      // 6. Slice logits at last prompt position for each batch item
      StorageView last_logits(prefix_logits.dtype(), device);
      {
        StorageView tmp(prefix_logits.dtype(), device);
        ops::Slide(1, prefix_len - 1, 1)(prefix_logits, tmp);
        tmp.reshape({batch, vocab_size});
        last_logits = std::move(tmp);
      }
      prefix_logits.release();

      // ----------------------------------------------------------------
      // Decode loop parameters
      // ----------------------------------------------------------------
      const Vocabulary& vocab = _model->get_vocabulary();
      const size_t eos_id     = vocab.eos_id();
      const dim_t  beam_size  = static_cast<dim_t>(std::max<size_t>(1, options.beam_size));
      const dim_t  num_hyp    = static_cast<dim_t>(std::max<size_t>(1, options.num_hypotheses));
      const size_t max_candidates = std::max(
        static_cast<size_t>(num_hyp),
        static_cast<size_t>(std::round(float(beam_size) * options.patience)));

      // ----------------------------------------------------------------
      // Repetition penalty helper
      // ----------------------------------------------------------------
      auto apply_rep_penalty = [&](StorageView& logits,
                                   const std::vector<std::vector<size_t>>& token_lists) {
        if (options.repetition_penalty == 1.0f || token_lists.empty()) return;
        const Device   dev = logits.device();
        const DataType dt  = logits.dtype();
        StorageView cpu_f32 = logits.to(Device::CPU);
        if (dt != DataType::FLOAT32) cpu_f32 = cpu_f32.to(DataType::FLOAT32);
        float* data = cpu_f32.data<float>();
        const dim_t n = cpu_f32.dim(0), v = cpu_f32.dim(1);
        for (dim_t i = 0; i < n; ++i) {
          const auto& ids = token_lists[
            (static_cast<size_t>(i) < token_lists.size())
            ? static_cast<size_t>(i) : token_lists.size() - 1];
          for (size_t id : ids) {
            if (static_cast<dim_t>(id) < v) {
              float& logit = data[i * v + id];
              if (logit > 0.f) logit /= options.repetition_penalty;
              else              logit *= options.repetition_penalty;
            }
          }
        }
        if (dt != DataType::FLOAT32) cpu_f32 = cpu_f32.to(dt);
        logits = cpu_f32.to(dev);
      };

      // ==============================================================
      // GREEDY / SAMPLING  (beam_size == 1)
      // ==============================================================
      if (beam_size == 1) {
        const bool use_random = (options.sampling_temperature != 1.0f
                                 || options.sampling_topk != 1);
        std::unique_ptr<Sampler> sampler;
        if (use_random)
          sampler = std::make_unique<RandomSampler>(
            static_cast<dim_t>(options.sampling_topk),
            /*topp=*/1.0f,
            options.sampling_temperature);
        else
          sampler = std::make_unique<BestSampler>();

        std::vector<std::vector<size_t>> generated(batch);
        std::vector<float> cum_lp(batch, 0.0f);
        std::vector<bool>  done(batch, false);

        auto do_sample = [&](StorageView logits) {
          apply_rep_penalty(logits, generated);

          StorageView log_probs(device);
          if (options.return_scores)
            ops::LogSoftMax()(logits, log_probs);

          StorageView ids_cpu(DataType::INT32, Device::CPU);
          StorageView sc_cpu(logits.dtype(), Device::CPU);
          (*sampler)(logits, ids_cpu, sc_cpu, /*num_samples=*/1);

          StorageView lp_cpu_f32;
          if (options.return_scores && log_probs) {
            lp_cpu_f32 = log_probs.to(Device::CPU);
            if (lp_cpu_f32.dtype() != DataType::FLOAT32)
              lp_cpu_f32 = lp_cpu_f32.to(DataType::FLOAT32);
          }

          for (dim_t b = 0; b < batch; ++b) {
            if (done[b]) continue;
            const size_t id = static_cast<size_t>(ids_cpu.data<int32_t>()[b]);
            generated[b].push_back(id);
            if (options.return_scores && lp_cpu_f32)
              cum_lp[b] += lp_cpu_f32.data<float>()[b * vocab_size + id];
            if (id == eos_id) done[b] = true;
          }
        };

        // First token from prefill logits
        do_sample(std::move(last_logits));

        StorageView step_logits(compute_dt, device);
        for (size_t t = 1; t < options.max_new_tokens; ++t) {
          if (std::all_of(done.begin(), done.end(), [](bool d) { return d; }))
            break;

          StorageView step_ids({batch}, DataType::INT32, Device::CPU);
          for (dim_t b = 0; b < batch; ++b)
            step_ids.data<int32_t>()[b] = done[b]
              ? static_cast<int32_t>(eos_id)
              : static_cast<int32_t>(generated[b].back());
          step_ids = step_ids.to(device);

          (*_impl->decoder)(
            static_cast<dim_t>(prefix_len + t - 1),
            step_ids, state, &step_logits);

          do_sample(step_logits);
        }

        std::vector<Qwen3ASRResult> results(batch);
        for (dim_t b = 0; b < batch; ++b) {
          auto& r = results[b];
          r.sequences_ids = {generated[b]};
          r.sequences.resize(1);
          for (size_t id : generated[b])
            r.sequences[0].push_back(vocab.to_token(id));
          if (options.return_scores) r.scores = {cum_lp[b]};
        }
        return results;
      }

      // ==============================================================
      // BEAM SEARCH  (beam_size > 1)
      // ==============================================================
      const dim_t num_cand = beam_size * 2;
      const float NEG_INF  = std::numeric_limits<float>::lowest();

      // Initial beam: TopK(beam_size) from prefill logits
      StorageView init_lp(compute_dt, device);
      ops::LogSoftMax()(last_logits, init_lp);
      last_logits.release();

      StorageView init_scores_dev(compute_dt, device), init_ids_dev(DataType::INT32, device);
      ops::TopK topk_init(beam_size);
      topk_init(init_lp, init_scores_dev, init_ids_dev);
      init_lp.release();

      StorageView init_scores = init_scores_dev.to(Device::CPU);
      if (init_scores.dtype() != DataType::FLOAT32)
        init_scores = init_scores.to(DataType::FLOAT32);
      StorageView init_ids = init_ids_dev.to(Device::CPU);
      init_scores_dev.release();
      init_ids_dev.release();

      struct Hypothesis { std::vector<size_t> tokens; float score; };
      std::vector<std::vector<std::vector<size_t>>> beam_tokens(
        batch, std::vector<std::vector<size_t>>(beam_size));
      std::vector<float> beam_scores(batch * beam_size, NEG_INF);
      std::vector<std::vector<Hypothesis>> completed(batch);
      std::vector<bool> batch_done(batch, false);

      for (dim_t b = 0; b < batch; ++b) {
        for (dim_t k = 0; k < beam_size; ++k) {
          const size_t tok = static_cast<size_t>(
            init_ids.data<int32_t>()[b * beam_size + k]);
          const float  sc  = init_scores.data<float>()[b * beam_size + k];
          beam_tokens[b][k] = {tok};
          if (tok == eos_id) {
            const float lp = (options.length_penalty != 0.0f)
              ? std::pow(1.0f, options.length_penalty) : 1.0f;
            completed[b].push_back({{tok}, sc / lp});
          } else {
            beam_scores[b * beam_size + k] = sc;
          }
        }
        if (completed[b].size() >= max_candidates) {
          std::sort(completed[b].begin(), completed[b].end(),
                    [](const auto& a, const auto& b) { return a.score > b.score; });
          float best = NEG_INF;
          for (dim_t k = 0; k < beam_size; ++k)
            best = std::max(best, beam_scores[b * beam_size + k]);
          if (best < completed[b][num_hyp - 1].score) batch_done[b] = true;
        }
      }

      // Expand KV cache for beam_size beams per batch item
      static_cast<layers::Decoder*>(_impl->decoder.get())->replicate_state(state, beam_size);

      StorageView step_logits(compute_dt, device);
      for (size_t t = 1; t < options.max_new_tokens; ++t) {
        if (std::all_of(batch_done.begin(), batch_done.end(),
                        [](bool d) { return d; }))
          break;

        StorageView step_ids({batch * beam_size}, DataType::INT32, Device::CPU);
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t k = 0; k < beam_size; ++k) {
            const size_t last = beam_tokens[b][k].empty()
              ? eos_id : beam_tokens[b][k].back();
            step_ids.data<int32_t>()[b * beam_size + k] =
              static_cast<int32_t>(last);
          }
        step_ids = step_ids.to(device);

        (*_impl->decoder)(
          static_cast<dim_t>(prefix_len + t - 1),
          step_ids, state, &step_logits);

        // Repetition penalty
        if (options.repetition_penalty != 1.0f) {
          std::vector<std::vector<size_t>> flat(batch * beam_size);
          for (dim_t b = 0; b < batch; ++b)
            for (dim_t k = 0; k < beam_size; ++k)
              flat[b * beam_size + k] = beam_tokens[b][k];
          apply_rep_penalty(step_logits, flat);
        }

        ops::LogSoftMax()(step_logits);
        StorageView lp_cpu = step_logits.to(Device::CPU);
        if (lp_cpu.dtype() != DataType::FLOAT32)
          lp_cpu = lp_cpu.to(DataType::FLOAT32);
        float* lp_data = lp_cpu.data<float>();

        // Add beam scores
        for (dim_t i = 0; i < batch * beam_size; ++i) {
          const float bs = beam_scores[i];
          for (dim_t v = 0; v < vocab_size; ++v)
            lp_data[i * vocab_size + v] = (bs <= NEG_INF / 2.0f)
              ? NEG_INF : lp_data[i * vocab_size + v] + bs;
        }

        lp_cpu.reshape({batch, beam_size * vocab_size});
        StorageView topk_vals(DataType::FLOAT32, Device::CPU);
        StorageView topk_ids(DataType::INT32, Device::CPU);
        ops::TopK topk_cand(num_cand);
        topk_cand(lp_cpu, topk_vals, topk_ids);

        const float*   tv  = topk_vals.data<float>();
        const int32_t* tid = topk_ids.data<int32_t>();

        std::vector<std::vector<std::vector<size_t>>> new_beams(
          batch, std::vector<std::vector<size_t>>(beam_size));
        std::vector<float>   new_scores(batch * beam_size, NEG_INF);
        std::vector<int32_t> gather_idx(batch * beam_size, 0);

        for (dim_t b = 0; b < batch; ++b) {
          if (batch_done[b]) {
            for (dim_t k = 0; k < beam_size; ++k) {
              new_beams[b][k] = beam_tokens[b][k];
              gather_idx[b * beam_size + k] = b * beam_size + k;
            }
            continue;
          }

          dim_t next_k = 0;
          for (dim_t ci = 0; ci < num_cand && next_k < beam_size; ++ci) {
            const float   sc  = tv [b * num_cand + ci];
            const int32_t fid = tid[b * num_cand + ci];
            if (sc <= NEG_INF / 2.0f) break;

            const dim_t beam_orig = fid / vocab_size;
            const dim_t token_id  = fid % vocab_size;

            if (token_id == static_cast<dim_t>(eos_id)) {
              const float len =
                float(beam_tokens[b][beam_orig].size() + 1);
              const float pen = (options.length_penalty != 0.0f)
                ? std::pow(len, options.length_penalty) : 1.0f;
              completed[b].push_back(
                {beam_tokens[b][beam_orig], sc / pen});
            } else {
              new_beams[b][next_k] = beam_tokens[b][beam_orig];
              new_beams[b][next_k].push_back(static_cast<size_t>(token_id));
              new_scores[b * beam_size + next_k] = sc;
              gather_idx[b * beam_size + next_k] = b * beam_size + beam_orig;
              ++next_k;
            }
          }

          // Dead-fill unused slots
          for (; next_k < beam_size; ++next_k) {
            new_beams[b][next_k] = {};
            gather_idx[b * beam_size + next_k] = b * beam_size;
          }

          if (completed[b].size() >= max_candidates) {
            std::sort(completed[b].begin(), completed[b].end(),
                      [](const auto& a, const auto& b) { return a.score > b.score; });
            const float best = new_scores[b * beam_size];
            if (best <= NEG_INF / 2.0f ||
                best < completed[b][num_hyp - 1].score)
              batch_done[b] = true;
          }
        }

        // Reorder KV cache
        StorageView gi_view({batch * beam_size}, DataType::INT32, Device::CPU);
        std::copy(gather_idx.begin(), gather_idx.end(), gi_view.data<int32_t>());
        gi_view = gi_view.to(device);
        _impl->decoder->update_state(state, std::move(gi_view), beam_size, nullptr);

        beam_tokens = std::move(new_beams);
        beam_scores = std::move(new_scores);
      }

      // Force-finish active beams
      for (dim_t b = 0; b < batch; ++b) {
        if (batch_done[b]) continue;
        for (dim_t k = 0; k < beam_size; ++k) {
          const float sc = beam_scores[b * beam_size + k];
          if (sc <= NEG_INF / 2.0f || beam_tokens[b][k].empty()) continue;
          const float len = float(beam_tokens[b][k].size());
          const float pen = (options.length_penalty != 0.0f)
            ? std::pow(len, options.length_penalty) : 1.0f;
          completed[b].push_back({beam_tokens[b][k], sc / pen});
        }
      }

      std::vector<Qwen3ASRResult> results(batch);
      for (dim_t b = 0; b < batch; ++b) {
        std::sort(completed[b].begin(), completed[b].end(),
                  [](const auto& a, const auto& b) { return a.score > b.score; });
        auto& r = results[b];
        const size_t n = std::min(completed[b].size(), static_cast<size_t>(num_hyp));
        r.sequences.reserve(n);
        r.sequences_ids.reserve(n);
        if (options.return_scores) r.scores.reserve(n);
        for (size_t h = 0; h < n; ++h) {
          std::vector<std::string> toks;
          toks.reserve(completed[b][h].tokens.size());
          for (size_t id : completed[b][h].tokens)
            toks.push_back(vocab.to_token(id));
          r.sequences.push_back(std::move(toks));
          r.sequences_ids.push_back(completed[b][h].tokens);
          if (options.return_scores) r.scores.push_back(completed[b][h].score);
        }
      }
      return results;
    }

    // -----------------------------------------------------------------------
    // Qwen3ASR pool
    // -----------------------------------------------------------------------
    std::future<StorageView>
    Qwen3ASR::encode(const StorageView& features, Qwen3ASROptions options) {
      return post<StorageView>(
        [features = features.sync_copy(), options = std::move(options)]
        (Qwen3ASRReplica& replica) mutable {
          return replica.encode(features, options);
        });
    }

    std::vector<std::future<Qwen3ASRResult>>
    Qwen3ASR::generate(const StorageView& features,
                        std::vector<std::vector<size_t>> prompts,
                        Qwen3ASROptions options) {
      const size_t batch_size = prompts.size();
      return post_batch<Qwen3ASRResult>(
        [features  = features.sync_copy(),
         prompts   = std::move(prompts),
         options   = std::move(options)]
        (Qwen3ASRReplica& replica) mutable {
          return replica.generate(features, prompts, options);
        },
        batch_size);
    }

  }  // namespace models
}  // namespace ctranslate2
