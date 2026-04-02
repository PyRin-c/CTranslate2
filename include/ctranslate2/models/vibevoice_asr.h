#pragma once

#include <vector>

#include "ctranslate2/generation.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    // -----------------------------------------------------------------------
    // Options for VibeVoice-ASR generation
    // -----------------------------------------------------------------------
    struct VibeVoiceAsrOptions {
      // Maximum number of tokens to generate (excluding audio tokens).
      size_t max_new_tokens = 8192;

      // Beam size (set 1 for greedy search).
      size_t beam_size = 1;

      // Beam search patience factor.
      float patience = 1;

      // Length penalty applied during beam search.
      float length_penalty = 1;

      // Repetition penalty (set > 1 to penalize).
      float repetition_penalty = 1;

      // Randomly sample from the top K candidates (set 0 to use full distribution).
      size_t sampling_topk = 1;

      // Sampling temperature.
      float sampling_temperature = 1;

      // Number of hypotheses to return.
      size_t num_hypotheses = 1;

      // Include scores in the result.
      bool return_scores = false;

      // Audio token placeholder ID (default matches VibeVoice-ASR spec).
      // Automatically set from config.json; normally no need to change.
      size_t audio_token_id = 151648;

      // Chunk size for encode_chunked() in samples (default: 60s @ 24kHz).
      size_t chunk_size = 1440000;

      // Inject VAE reparameterisation noise during encode().
      // Set to true only for training-style sampling; leave false (default) for inference.
      // When false the mean of the acoustic latent distribution is used directly,
      // avoiding random output and the precision-loss of a float32 round-trip on
      // float16 models.
      bool add_vae_noise = false;
    };

    struct VibeVoiceAsrResult {
      // Generated token sequences (decoded to strings by the caller if needed).
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;

      size_t num_sequences() const {
        return sequences.size();
      }
    };

    // -----------------------------------------------------------------------
    // VibeVoice-ASR Model (manages weights shared across replicas)
    // -----------------------------------------------------------------------
    class VibeVoiceAsrModel : public Model {
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
    // VibeVoice-ASR Replica (per-device inference instance)
    // -----------------------------------------------------------------------
    class VibeVoiceAsrReplica : public ModelReplica {
    public:
      static std::unique_ptr<VibeVoiceAsrReplica>
      create_from_model(const Model& model);

      VibeVoiceAsrReplica(const std::shared_ptr<const VibeVoiceAsrModel>& model);
      ~VibeVoiceAsrReplica() override;

      // Encodes raw waveform into audio feature embeddings.
      // waveform: (batch, num_samples) @ 24kHz, float32
      // Returns: (batch, audio_seq_len, lm_hidden_size)
      StorageView encode(const StorageView& waveform, const VibeVoiceAsrOptions& options);

      // Generates transcription tokens given pre-computed audio features.
      // input_ids:      (batch, seq_len) — text + audio_token_id placeholders
      // audio_features: output of encode(), (batch, audio_seq_len, lm_hidden_size)
      std::vector<VibeVoiceAsrResult>
      generate(const StorageView& input_ids,
               const StorageView& audio_features,
               const VibeVoiceAsrOptions& options);

    private:
      const std::shared_ptr<const VibeVoiceAsrModel> _model;

      // Pimpl for internal layer objects (avoids exposing impl headers publicly).
      struct Impl;
      std::unique_ptr<Impl> _impl;

      // Injects audio_features into positions where input_ids == audio_token_id.
      void _build_inputs_embeds(const StorageView& input_ids,
                                 const StorageView& audio_features,
                                 size_t audio_token_id,
                                 StorageView& inputs_embeds) const;
    };

    // -----------------------------------------------------------------------
    // VibeVoice-ASR Pool (thread-safe, multi-replica entry point)
    // -----------------------------------------------------------------------
    class VibeVoiceAsr : public ReplicaPool<VibeVoiceAsrReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      // Asynchronously encodes waveform on the pool device.
      std::future<StorageView>
      encode(const StorageView& waveform,
             const VibeVoiceAsrOptions& options = {});

      // Asynchronously generates transcription.
      std::vector<std::future<VibeVoiceAsrResult>>
      generate(const StorageView& input_ids,
               const StorageView& audio_features,
               VibeVoiceAsrOptions options = {});
    };

  }  // namespace models
}  // namespace ctranslate2
