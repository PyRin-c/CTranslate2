#pragma once

#include "ctranslate2/layers/attention.h"
#include "ctranslate2/layers/flash_attention.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    class FeedForwardNetwork : public Layer
    {
    public:
      FeedForwardNetwork(const models::Model& model,
                         const std::string& scope,
                         const bool pre_norm = true,
                         const ops::ActivationType activation_type = ops::ActivationType::ReLU);

      void operator()(const StorageView& input, StorageView& output) const;

      DataType output_type() const override {
        return _ff2.output_type();
      }

      dim_t output_size() const override {
        return _ff2.output_size();
      }

    private:
      const std::unique_ptr<const LayerNorm> _layer_norm;
      const bool _pre_norm;
      const ops::ActivationType _activation_type;
      const Dense _ff1;
      const std::unique_ptr<const Dense> _ff1_noact;
      const Dense _ff2;
      const bool _tensor_parallel;
    };

    class TransformerEncoderLayer : public Layer
    {
    public:
      TransformerEncoderLayer(const models::Model& model,
                              const std::string& scope,
                              const dim_t num_heads,
                              const bool pre_norm = true,
                              const ops::ActivationType activation_type = ops::ActivationType::ReLU,
                              bool use_flash_attention = false);

      void operator()(const StorageView& input,
                      const StorageView* lengths,
                      StorageView& output,
                      const Padder* padder = nullptr,
                      StorageView* position_bias = nullptr) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

      const AttentionLayer& get_self_attention() const {
        return *_self_attention;
      }

    private:
      std::unique_ptr<AttentionLayer> _self_attention;
      const std::unique_ptr<const LayerNorm> _input_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_attention_layer_norm;
      const std::unique_ptr<const LayerNorm> _pre_feedforward_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_feedforward_layer_norm;
      const FeedForwardNetwork _ff;
    };

    class TransformerDecoderLayer : public Layer
    {
    public:
      TransformerDecoderLayer(const models::Model& model,
                              const std::string& scope,
                              const dim_t num_heads,
                              const bool pre_norm = true,
                              const ops::ActivationType activation_type = ops::ActivationType::ReLU,
                              const bool use_flash_attention = true,
                              Alibi* alibi = nullptr);

      void operator()(const StorageView& input,
                      const StorageView* input_lengths,
                      const StorageView* memory,
                      const StorageView* memory_lengths,
                      StorageView* cached_self_attn_keys,
                      StorageView* cached_self_attn_values,
                      StorageView* cached_attn_keys,
                      StorageView* cached_attn_values,
                      StorageView& output,
                      StorageView* attention = nullptr,
                      const Padder* input_padder = nullptr,
                      const Padder* memory_padder = nullptr,
                      bool return_normalized_attention = true,
                      StorageView* position_bias = nullptr,
                      dim_t offset = 0,
                      const StorageView* per_layer_input = nullptr,
                      bool kv_read_only = false) const;

      DataType output_type() const override {
        return _ff.output_type();
      }

      dim_t output_size() const override {
        return _ff.output_size();
      }

      bool has_cross_attention() const {
        return bool(_encoder_attention);
      }

      const AttentionLayer& get_self_attention() const {
        return *_self_attention;
      }

      // Returns the source layer index for KV cache sharing, or -1 if this layer
      // has its own KV cache (the common case).
      dim_t kv_shared_layer_index() const {
        return _kv_shared_layer_index;
      }

    private:
      // Index of the source layer whose KV cache this layer borrows (-1 = own cache).
      const dim_t _kv_shared_layer_index;
      const std::unique_ptr<AttentionLayer> _self_attention;
      const std::unique_ptr<const LayerNorm> _shared_layer_norm;
      const std::unique_ptr<const LayerNorm> _input_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_attention_layer_norm;
      const std::unique_ptr<const LayerNorm> _pre_feedforward_layer_norm;
      const std::unique_ptr<const LayerNorm> _post_feedforward_layer_norm;
      const std::unique_ptr<const AttentionLayer> _encoder_attention;
      const FeedForwardNetwork _ff;
      const std::unique_ptr<const LayerNorm> _external_pre_encoder_attention_layer_norm;
      const std::unique_ptr<const LayerNorm> _external_post_encoder_attention_layer_norm;
      // Gemma 4: per-layer scalar multiplied to the output after FFN residual
      const float _layer_scalar;
      // Gemma 4: per-layer embeddings (PLE)
      const std::unique_ptr<const Dense> _per_layer_input_gate;
      const std::unique_ptr<const Dense> _per_layer_projection;
      const std::unique_ptr<const LayerNorm> _post_per_layer_input_norm;
      // Gemma 4 MoE: router components
      const std::unique_ptr<const LayerNorm> _router_norm;
      const std::unique_ptr<const StorageView> _router_scale_prescaled;  // scale * (1/√H)
      const std::unique_ptr<const Dense> _router_proj;                   // H → num_experts
      const std::unique_ptr<const StorageView> _router_per_expert_scale; // [num_experts]
      // Gemma 4 MoE: expert weights (3D tensors)
      const std::unique_ptr<const StorageView> _experts_gate_up_proj;    // [E, 2*I, H]
      const std::unique_ptr<const StorageView> _experts_down_proj;       // [E, H, I]
      const dim_t _moe_top_k;
      // Gemma 4 MoE: extra layer norms
      const std::unique_ptr<const LayerNorm> _post_feedforward_layernorm_1;
      const std::unique_ptr<const LayerNorm> _pre_feedforward_layernorm_2;
      const std::unique_ptr<const LayerNorm> _post_feedforward_layernorm_2;
    };

    class TransformerEncoder : public Encoder
    {
    public:
      TransformerEncoder(const models::Model& model, const std::string& scope);

      void operator()(const std::vector<StorageView>& ids,
                      const StorageView* lengths,
                      StorageView& output) override;

      size_t num_input_features() const override {
        return _embeddings.num_inputs();
      }

      DataType output_type() const override {
        return _layers.back()->output_type();
      }

      dim_t output_size() const override {
        return _layers.back()->output_size();
      }

    private:
      const ParallelEmbeddings _embeddings;
      const std::unique_ptr<const StorageView> _embeddings_scale;
      const dim_t _num_heads;
      const ComputeType _compute_type;
      const std::unique_ptr<const LayerNorm> _layernorm_embedding;
      const std::unique_ptr<const LayerNorm> _output_norm;
      const bool _use_flash_attention;
      const std::vector<std::unique_ptr<const TransformerEncoderLayer>> _layers;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const bool _tensor_parallel;
    };

    class TransformerDecoder : public Decoder
    {
    public:
      TransformerDecoder(const models::Model& model, const std::string& scope);

      DecoderState initial_state(bool iterative_decoding = true) const override;
      bool replicate_state(const std::string& name) const override;

      void operator()(dim_t step,
                      const StorageView& ids,
                      DecoderState& state,
                      StorageView* logits = nullptr,
                      StorageView* attention = nullptr) override;
      void operator()(const StorageView& ids,
                      const StorageView& lengths,
                      DecoderState& state,
                      StorageView& logits,
                      StorageView* attention = nullptr) override;

      void set_alignment_heads(const dim_t layer, const dim_t num_heads_to_average);
      void set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads);

      std::unique_ptr<StorageView>
      get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const;

      virtual bool return_normalized_attention() const {
        return true;
      }

    protected:
      Dense& output_layer() override {
        return _proj;
      }

      void decode(const StorageView& ids,
                  const StorageView* lengths,
                  dim_t step,
                  DecoderState& state,
                  StorageView* outputs = nullptr,
                  StorageView* attention = nullptr,
                  bool return_logits = true);

      // Like decode() but uses pre-computed embeddings instead of token IDs.
      // input_ids (optional): [batch, seq_len] int32 token IDs, used to compute
      // the token-side PLE contribution (Gemma 4 E-series only).
      void decode_from_embeds(const StorageView& inputs_embeds,
                              const StorageView* input_ids,
                              const StorageView* lengths,
                              dim_t step,
                              DecoderState& state,
                              StorageView* outputs = nullptr,
                              StorageView* attention = nullptr,
                              bool return_logits = true);

      // Backward-compatible overload (no input_ids → token PLE skipped).
      void decode_from_embeds(const StorageView& inputs_embeds,
                              const StorageView* lengths,
                              dim_t step,
                              DecoderState& state,
                              StorageView* outputs = nullptr,
                              StorageView* attention = nullptr,
                              bool return_logits = true);

      // Shared output-projection helper: applies output_norm, project_out,
      // outputs_scale, and the final linear projection (or hidden-state passthrough).
      // Called by both decode() and decode_from_embeds() to avoid code duplication.
      void _apply_output_projection(StorageView& layer_in,
                                    StorageView& layer_out,
                                    StorageView* outputs,
                                    bool return_logits,
                                    bool is_sequence,
                                    const Padder* input_padder);

      const dim_t _num_heads;
      const ComputeType _compute_type;
      const Embeddings _embeddings;
      const bool _start_from_zero_embedding;
      const std::unique_ptr<const StorageView> _embeddings_scale;
      std::unique_ptr<const StorageView> _outputs_scale;
      const std::unique_ptr<const LayerNorm> _layernorm_embedding;
      const std::unique_ptr<const LayerNorm> _output_norm;
      const std::unique_ptr<const Dense> _project_in;
      const std::unique_ptr<const Dense> _project_out;
      const std::unique_ptr<Alibi> _alibi;
      const bool _use_flash_attention;
      const std::vector<std::unique_ptr<const TransformerDecoderLayer>> _layers;
      const std::unique_ptr<PositionEncoder> _position_encoder;
      const bool _with_encoder_attention;
      std::vector<std::vector<dim_t>> _alignment_heads;
      bool _average_alignment_heads;
      Dense _proj;
      const dim_t _sliding_window;
      const bool _tensor_parallel;
      // Gemma 4: final logit soft-capping: tanh(x/cap)*cap applied after lm_head
      const float _final_logit_softcap;
      // Gemma 4 E-series: per-layer embedding table [vocab_size, num_layers * per_layer_dim] int8.
      // Looked up by token IDs to produce the token-side PLE contribution.
      const std::unique_ptr<const StorageView> _per_layer_embeddings;
      // Companion scale factors [vocab_size] float32 (one per vocabulary-token row).
      const std::unique_ptr<const StorageView> _per_layer_embedding_scales;
      // Gemma 4 E-series: model-projection for PLE [hidden_size → num_layers * per_layer_dim]
      const std::unique_ptr<const Dense> _per_layer_model_proj;
      // Gemma 4 E-series: per-layer projection norm [per_layer_dim]
      const std::unique_ptr<const LayerNorm> _per_layer_proj_norm;

      // Compute combined per-layer input tensors for all layers.
      // Returns a vector of length num_layers; each element is [batch, seq, per_layer_dim].
      // If PLE weights are not present, returns an empty vector.
      std::vector<StorageView> _compute_per_layer_inputs(
          const StorageView* ids,              // [batch, seq] int32, optional
          const StorageView& inputs_embeds     // [batch, seq, hidden]
      ) const;
    };

  }
}
