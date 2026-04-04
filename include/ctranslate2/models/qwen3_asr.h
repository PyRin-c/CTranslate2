#pragma once

#include <vector>

#include "ctranslate2/generation.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    // -----------------------------------------------------------------------
    // Options for Qwen3-ASR generation
    // -----------------------------------------------------------------------
    struct Qwen3ASROptions {
      // Maximum number of new tokens to generate.
      size_t max_new_tokens = 512;

      // Beam size (1 = greedy).
      size_t beam_size = 1;

      // Beam search patience factor.
      float patience = 1;

      // Length penalty applied during beam search.
      float length_penalty = 1;

      // Repetition penalty (> 1 penalizes repetition).
      float repetition_penalty = 1;

      // Top-K sampling candidates (0 = full distribution).
      size_t sampling_topk = 1;

      // Sampling temperature.
      float sampling_temperature = 1;

      // Number of hypotheses to return.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Audio placeholder token ID (read from model config; usually 151676).
      size_t audio_token_id = 151676;

      // Windowed attention window size in audio tokens for the encoder.
      // -1 → offline (full attention with block-diagonal mask matching n_window_infer).
      //  N → streaming (N-token window).
      int encoder_window_tokens = -1;
    };

    struct Qwen3ASRResult {
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;

      size_t num_sequences() const {
        return sequences.size();
      }
    };

    // -----------------------------------------------------------------------
    // Qwen3ASRModel — manages shared weights across replicas
    // -----------------------------------------------------------------------
    class Qwen3ASRModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

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
    };

    // -----------------------------------------------------------------------
    // Qwen3ASRReplica — per-device inference instance
    // -----------------------------------------------------------------------
    class Qwen3ASRReplica : public ModelReplica {
    public:
      static std::unique_ptr<Qwen3ASRReplica> create_from_model(const Model& model);

      explicit Qwen3ASRReplica(const std::shared_ptr<const Qwen3ASRModel>& model);
      ~Qwen3ASRReplica() override;

      // Encode mel spectrogram → audio features.
      // features: (batch, 128, T) — mel spectrogram, variable T.
      // Returns: (batch, N_audio_tokens, lm_hidden_size).
      StorageView encode(const StorageView& features,
                         const Qwen3ASROptions& options = {});

      // Full pipeline: encode + prefill + autoregressive decode.
      // features: (batch, 128, T)
      // prompts:  token_ids including audio_token_id placeholders
      std::vector<Qwen3ASRResult>
      generate(const StorageView& features,
               const std::vector<std::vector<size_t>>& prompts,
               const Qwen3ASROptions& options = {});

    private:
      const std::shared_ptr<const Qwen3ASRModel> _model;

      struct Impl;
      std::unique_ptr<Impl> _impl;

      // Inject audio_features into token_embeddings at audio_token_id positions.
      void _build_inputs_embeds(const StorageView& input_ids,
                                 const StorageView& audio_features,
                                 size_t audio_token_id,
                                 StorageView& inputs_embeds) const;
    };

    // -----------------------------------------------------------------------
    // Qwen3ASR — thread-safe, multi-replica pool
    // -----------------------------------------------------------------------
    class Qwen3ASR : public ReplicaPool<Qwen3ASRReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      std::future<StorageView>
      encode(const StorageView& features,
             Qwen3ASROptions options = {});

      std::vector<std::future<Qwen3ASRResult>>
      generate(const StorageView& features,
               std::vector<std::vector<size_t>> prompts,
               Qwen3ASROptions options = {});
    };

  }  // namespace models
}  // namespace ctranslate2
