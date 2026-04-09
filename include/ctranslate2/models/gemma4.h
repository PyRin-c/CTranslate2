#pragma once

#include <vector>
#include <string>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/transformer.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    // -----------------------------------------------------------------------
    // Gemma4GenerationOptions
    // -----------------------------------------------------------------------
    struct Gemma4GenerationOptions {
      // Maximum number of new tokens to generate (excluding the prompt).
      size_t max_new_tokens = 256;

      // Beam size (1 = greedy / sampling, >1 = beam search).
      size_t beam_size = 1;

      // Beam search patience factor (see https://arxiv.org/abs/2204.05424).
      float patience = 1.0f;

      // Length normalisation exponent applied to completed hypotheses.
      float length_penalty = 1.0f;

      // Repetition penalty > 1 penalises previously generated tokens.
      float repetition_penalty = 1.0f;

      // Restrict random sampling to the top-K vocabulary entries (0 = full).
      size_t sampling_topk = 1;

      // Temperature for random sampling.
      float sampling_temperature = 1.0f;

      // Number of hypotheses to include in the result per batch item.
      size_t num_hypotheses = 1;

      // Whether to populate result scores.
      bool return_scores = false;

      // Token ID of the image placeholder in the input sequence.
      // Must match the value used when building input_ids on the Python side.
      // This is stored in config.json as "image_token_id" by the converter.
      size_t image_token_id = 0;
    };

    // -----------------------------------------------------------------------
    // Gemma4GenerationResult  (one per batch item)
    // -----------------------------------------------------------------------
    struct Gemma4GenerationResult {
      // Top-N generated sequences decoded to token strings.
      std::vector<std::vector<std::string>> sequences;
      // Same sequences expressed as vocabulary IDs.
      std::vector<std::vector<size_t>> sequences_ids;
      // Hypothesis scores (empty unless return_scores=true).
      std::vector<float> scores;
    };

    // Forward declaration: Gemma4Decoder is a private subclass of
    // TransformerDecoder defined in gemma4.cc.  Declared here so
    // Gemma4Replica can hold a unique_ptr without including the implementation.
    class Gemma4Decoder;

    // -----------------------------------------------------------------------
    // Gemma4VisionSelfAttention
    //
    // Self-attention for the Gemma 4 SigLIP vision encoder:
    //   - Fused QKV linear (linear[0]) + output linear (linear[1])
    //   - Per-head q_norm and k_norm (learnable RMSNorm)
    //   - v_norm (scaleless RMSNorm, gamma = ones)
    //   - Bidirectional (no causal mask, no KV caching)
    //   - 2D RoPE applied externally before calling operator()
    // -----------------------------------------------------------------------
    class Gemma4VisionSelfAttention : public layers::Layer {
    public:
      Gemma4VisionSelfAttention(const Model& model, const std::string& scope,
                                dim_t num_heads, dim_t head_dim);

      // x: [B, T, D], cos/sin: [B*T, head_dim] (precomputed 2D RoPE)
      void operator()(const StorageView& x,
                      const StorageView& cos,
                      const StorageView& sin,
                      StorageView& output) const;

      DataType output_type() const override {
        return _linear.back().output_type();
      }

      dim_t output_size() const override {
        return _linear.back().output_size();
      }

    private:
      const std::vector<layers::Dense> _linear;  // [0]=QKV fused, [1]=output
      const std::unique_ptr<layers::LayerNorm> _q_norm;
      const std::unique_ptr<layers::LayerNorm> _k_norm;
      const std::unique_ptr<layers::LayerNorm> _v_norm;
      const dim_t _num_heads;
      const dim_t _head_dim;
    };


    // -----------------------------------------------------------------------
    // Gemma4VisionEncoderLayer
    //
    // Full SigLIP transformer layer with pre/post layer norms and GeGLU FFN.
    // -----------------------------------------------------------------------
    class Gemma4VisionEncoderLayer : public layers::Layer {
    public:
      Gemma4VisionEncoderLayer(const Model& model, const std::string& scope,
                               dim_t num_heads, dim_t head_dim);

      // input: [B, T, D]
      // cos/sin: [B*T, head_dim] precomputed 2D RoPE for this batch
      void operator()(const StorageView& input,
                      const StorageView& cos,
                      const StorageView& sin,
                      StorageView& output) const;

      DataType output_type() const override {
        return _ff2.output_type();
      }

      dim_t output_size() const override {
        return _ff2.output_size();
      }

    private:
      const layers::LayerNorm _input_layer_norm;
      const layers::LayerNorm _post_attention_layer_norm;
      const layers::LayerNorm _pre_feedforward_layer_norm;
      const layers::LayerNorm _post_feedforward_layer_norm;
      const Gemma4VisionSelfAttention _self_attention;
      // GeGLU FFN: out = down(act(gate) * up)
      const layers::Dense _ff_gate;    // ffn/linear_0   (with activation)
      const layers::Dense _ff_up;      // ffn/linear_0_noact (no activation)
      const layers::Dense _ff2;        // ffn/linear_1   (down projection)
    };


    // -----------------------------------------------------------------------
    // Gemma4VisionEncoder
    //
    // Processes patches to produce hidden states:
    //   patch_embedder (Linear + 2D position table) → N encoder layers
    //
    // The 2D RoPE cos/sin tables are stored as weights and gathered at runtime
    // using the per-batch pixel_position_ids.
    // -----------------------------------------------------------------------
    class Gemma4VisionEncoder : public layers::Layer {
    public:
      Gemma4VisionEncoder(const Model& model, const std::string& scope);

      // pixel_values: [B, T, 3*patch_size^2]   (flattened patches)
      // pixel_position_ids: [B, T, 2]           (int32 x/y position IDs)
      // output: [B, T, vision_hidden_size]
      void operator()(const StorageView& pixel_values,
                      const StorageView& pixel_position_ids,
                      StorageView& output) const;

      DataType output_type() const override {
        return _layers.back()->output_type();
      }

      dim_t output_size() const override {
        return _layers.back()->output_size();
      }

    private:
      // Build the 2D RoPE cos/sin tensors for the current batch.
      // rope_cos: [max_positions, spatial_dim] (preloaded weight)
      // pixel_position_ids: [B, T, 2] (int32)
      // Returns cos and sin of shape [B*T, head_dim] ready for ops::Rotary.
      void build_2d_rope(const StorageView& pixel_position_ids,
                         StorageView& cos,
                         StorageView& sin) const;

      // Patch embedder components
      const layers::Dense _input_proj;             // Linear(3*p^2 → hidden)
      const StorageView _position_embedding_table; // [2, max_pos, hidden]

      // Precomputed 2D RoPE tables: [max_positions, spatial_dim]
      const StorageView _rope_cos;
      const StorageView _rope_sin;

      const dim_t _num_heads;
      const dim_t _head_dim;

      const std::vector<std::unique_ptr<Gemma4VisionEncoderLayer>> _layers;
    };


    // -----------------------------------------------------------------------
    // Gemma4MultimodalProjector
    //
    // Projects pooled vision tokens into the text model's hidden space.
    //   pre_projection_norm (scaleless RMSNorm) → projection (Linear)
    // -----------------------------------------------------------------------
    class Gemma4MultimodalProjector : public layers::Layer {
    public:
      Gemma4MultimodalProjector(const Model& model, const std::string& scope);

      // input: [num_tokens, vision_hidden_size]
      // output: [num_tokens, text_hidden_size]
      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _projection.output_type();
      }

      dim_t output_size() const override {
        return _projection.output_size();
      }

    private:
      const layers::LayerNorm _pre_projection_norm;
      const layers::Dense _projection;
    };


    // -----------------------------------------------------------------------
    // Gemma4Model  (Model base class)
    // -----------------------------------------------------------------------
    class Gemma4Model : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

      // Token ID of the image placeholder used in the text prompt.
      // Read from config.json key "image_token_id" during initialize().
      size_t get_image_token_id() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
      size_t _image_token_id = 0;
    };


    // -----------------------------------------------------------------------
    // Gemma4Replica  (ModelReplica — one per device thread)
    // -----------------------------------------------------------------------
    class Gemma4Replica : public ModelReplica {
    public:
      static std::unique_ptr<Gemma4Replica> create_from_model(const Model& model);

      explicit Gemma4Replica(const std::shared_ptr<const Gemma4Model>& model);

      // Encode pixel patches and project to text space.
      // pixel_values:       [batch, num_patches, 3*patch_size^2]
      // pixel_position_ids: [batch, num_patches, 2]  (int32 x/y coordinates)
      // Returns: [batch, num_patches, text_hidden_size]
      StorageView encode_vision(const StorageView& pixel_values,
                                const StorageView& pixel_position_ids);

      // End-to-end multimodal generation.
      // input_ids:     [batch, seq_len]  int32 (CPU) — text prompt with image
      //                placeholder tokens at vision positions.
      // vision_embeds: [batch, num_img_tokens, text_hidden]  float — output of
      //                encode_vision(), already projected into text embedding space.
      // Returns one Gemma4GenerationResult per batch item.
      std::vector<Gemma4GenerationResult>
      generate(const StorageView& input_ids,
               const StorageView& vision_embeds,
               const Gemma4GenerationOptions& options);

    private:
      // Build inputs_embeds by embedding input_ids and replacing image
      // placeholder positions with vision_embeds.  Also applies the Gemma 4
      // embedding scale (sqrt(text_hidden_size)).
      void _build_inputs_embeds(const StorageView& input_ids,
                                const StorageView& vision_embeds,
                                size_t image_token_id,
                                StorageView& inputs_embeds) const;

      const std::shared_ptr<const Gemma4Model> _model;
      const Gemma4VisionEncoder _vision_encoder;
      const Gemma4MultimodalProjector _multimodal_projector;
      // Gemma4Decoder is a TransformerDecoder subclass that exposes
      // forward_with_embeds() for the multimodal prefill path.
      const std::unique_ptr<Gemma4Decoder> _decoder;
    };


    // -----------------------------------------------------------------------
    // Gemma4  (public API — ReplicaPool of Gemma4Replica)
    // -----------------------------------------------------------------------
    class Gemma4 : public ReplicaPool<Gemma4Replica> {
    public:
      using ReplicaPool::ReplicaPool;

      std::future<StorageView>
      encode_vision(const StorageView& pixel_values,
                    const StorageView& pixel_position_ids);

      // Asynchronous multimodal generation.
      // Returns one future per batch item (batch_size = input_ids.dim(0)).
      std::vector<std::future<Gemma4GenerationResult>>
      generate(const StorageView& input_ids,
               const StorageView& vision_embeds,
               Gemma4GenerationOptions options = {});
    };

  }  // namespace models
}  // namespace ctranslate2
