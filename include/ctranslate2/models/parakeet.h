#pragma once

#include <vector>
#include <memory>

#include "ctranslate2/layers/conformer.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    // -----------------------------------------------------------------------
    // ParakeetOptions
    // -----------------------------------------------------------------------
    struct ParakeetOptions {
      // Use TDT decoder (true) or CTC decoder (false).
      bool use_tdt = false;

      // Maximum search steps in TDT decoding (safety guard against infinite loops).
      // 0 → use encoder_length * 2.
      size_t max_tdt_steps = 0;
    };

    // -----------------------------------------------------------------------
    // ParakeetResult: one hypothesis per batch item.
    // -----------------------------------------------------------------------
    struct ParakeetResult {
      std::vector<std::vector<size_t>> ids;   // token ids per batch item
      // TDT only: encoder-frame index at which each token was emitted.
      // Empty when using CTC decoding.
      std::vector<std::vector<dim_t>> token_start_frames;
    };

    // -----------------------------------------------------------------------
    // ParakeetModel
    // -----------------------------------------------------------------------
    class ParakeetModel : public Model {
    public:
      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;
    };

    // -----------------------------------------------------------------------
    // ParakeetReplica
    // -----------------------------------------------------------------------
    class ParakeetReplica : public ModelReplica {
    public:
      static std::unique_ptr<ParakeetReplica> create_from_model(const Model& model);
      explicit ParakeetReplica(const std::shared_ptr<const ParakeetModel>& model);

      // Encode mel spectrogram: [batch, 80, T] → [batch, T', 1024]
      StorageView encode(const StorageView& mel,
                         const StorageView& lengths,
                         StorageView& out_lengths);

      // Full transcription pipeline.
      ParakeetResult
      transcribe(const StorageView& mel,
                 const StorageView& lengths,
                 const ParakeetOptions& options = {});

    private:
      const std::shared_ptr<const ParakeetModel> _model;
      std::unique_ptr<layers::ConformerEncoder>  _encoder;
      // CTC decoder components
      std::unique_ptr<layers::Dense>             _ctc_output_proj;
      // TDT decoder components
      std::unique_ptr<layers::Embeddings>        _tdt_embedding;
      std::unique_ptr<layers::Dense>             _tdt_enc_proj;
      std::unique_ptr<layers::Dense>             _tdt_pred_proj;
      std::unique_ptr<layers::Dense>             _tdt_output_head;
      // LSTM weights held as raw model variables
      const StorageView* _lstm0_wih = nullptr;
      const StorageView* _lstm0_whh = nullptr;
      const StorageView* _lstm0_bias = nullptr;
      const StorageView* _lstm1_wih = nullptr;
      const StorageView* _lstm1_whh = nullptr;
      const StorageView* _lstm1_bias = nullptr;

      // Model config — read dynamically from stored scalars and weight shapes
      // at construction time (supports parakeet-tdt_ctc and parakeet-tdt variants).
      size_t _vocab_size;     // number of tokens including blank
      size_t _blank_id;       // vocab_size - 1
      size_t _num_durations;  // TDT duration bins (typically 5: 0..4 frames)
      size_t _pred_dim;       // prediction network embedding/LSTM input dim
      size_t _lstm_hidden;    // LSTM hidden size
      size_t _lstm_layers;    // number of LSTM layers

      // CTC greedy decode
      std::vector<size_t> ctc_decode_sequence(const float* log_probs,
                                              dim_t T,
                                              dim_t vocab_size,
                                              dim_t row_stride) const;

      // TDT greedy decode; start_frames receives the encoder frame index
      // at which each non-blank token was emitted.
      std::vector<size_t> tdt_decode_sequence(const StorageView& enc_out,
                                              dim_t enc_length,
                                              size_t max_steps,
                                              std::vector<dim_t>& start_frames) const;

      // Single LSTM cell step (one layer)
      void lstm_step(const StorageView& x_t,         // [1, input_size]
                     const StorageView& wih,         // [4H, input_size]
                     const StorageView& whh,         // [4H, H]
                     const StorageView& bias,        // [4H]
                     StorageView& h,                 // [1, H] in/out
                     StorageView& c) const;          // [1, H] in/out
    };

    // -----------------------------------------------------------------------
    // Parakeet (ReplicaPool facade)
    // -----------------------------------------------------------------------
    class Parakeet : public ReplicaPool<ParakeetReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      std::future<StorageView>
      encode(const StorageView& mel, const StorageView& lengths);

      std::future<ParakeetResult>
      transcribe(const StorageView& mel,
                 const StorageView& lengths,
                 ParakeetOptions options = {});
    };

  }
}
