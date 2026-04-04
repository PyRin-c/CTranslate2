#include "module.h"

#include <ctranslate2/models/qwen3_asr.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    // -----------------------------------------------------------------------
    // Qwen3ASR Python wrapper
    // -----------------------------------------------------------------------
    class Qwen3ASRWrapper : public ReplicaPoolHelper<models::Qwen3ASR> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      StorageView encode(const StorageView& features,
                         int encoder_window_tokens) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        models::Qwen3ASROptions opts;
        opts.encoder_window_tokens = encoder_window_tokens;
        return _pool->encode(features, opts).get();
      }

      std::vector<models::Qwen3ASRResult>
      generate(const StorageView& features,
               const std::vector<std::vector<size_t>>& prompts,
               size_t max_new_tokens,
               size_t beam_size,
               float patience,
               float length_penalty,
               float repetition_penalty,
               size_t sampling_topk,
               float sampling_temperature,
               size_t num_hypotheses,
               bool return_scores,
               size_t audio_token_id,
               int encoder_window_tokens) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        models::Qwen3ASROptions opts;
        opts.max_new_tokens = max_new_tokens;
        opts.beam_size = beam_size;
        opts.patience = patience;
        opts.length_penalty = length_penalty;
        opts.repetition_penalty = repetition_penalty;
        opts.sampling_topk = sampling_topk;
        opts.sampling_temperature = sampling_temperature;
        opts.num_hypotheses = num_hypotheses;
        opts.return_scores = return_scores;
        opts.audio_token_id = audio_token_id;
        opts.encoder_window_tokens = encoder_window_tokens;

        auto futures = _pool->generate(features, prompts, opts);
        std::vector<models::Qwen3ASRResult> results;
        results.reserve(futures.size());
        for (auto& f : futures)
          results.emplace_back(f.get());
        return results;
      }
    };


    void register_qwen3_asr(py::module& m) {
      py::class_<models::Qwen3ASRResult>(
        m, "Qwen3ASRResult",
        "A generation result from the Qwen3-ASR model.")

        .def_readonly("sequences", &models::Qwen3ASRResult::sequences,
                      "Generated token sequences (list of list of str).")
        .def_readonly("sequences_ids", &models::Qwen3ASRResult::sequences_ids,
                      "Generated token IDs (list of list of int).")
        .def_readonly("scores", &models::Qwen3ASRResult::scores,
                      "Hypothesis scores (list of float, empty unless return_scores=True).")
        .def("__repr__", [](const models::Qwen3ASRResult& r) {
          return "Qwen3ASRResult(sequences="
                 + std::string(py::repr(py::cast(r.sequences))) + ")";
        })
        ;

      py::class_<Qwen3ASRWrapper>(
        m, "Qwen3ASR",
        R"pbdoc(
          Qwen3-ASR CTranslate2 inference class.

          Loads a converted Qwen3-ASR model directory and provides
          ``encode()`` and ``generate()`` for fast CPU/GPU inference.
        )pbdoc")

        .def(py::init<const std::string&,
                      const std::string&,
                      const std::variant<int, std::vector<int>>&,
                      const StringOrMap&,
                      size_t,
                      size_t,
                      long,
                      bool,
                      bool,
                      py::object>(),
             py::arg("model_path"),
             py::arg("device") = "cpu",
             py::arg("device_index") = 0,
             py::arg("compute_type") = "default",
             py::arg("inter_threads") = 1,
             py::arg("intra_threads") = 0,
             py::arg("max_queued_batches") = 0,
             py::arg("flash_attention") = false,
             py::arg("tensor_parallel") = false,
             py::arg("files") = py::none(),
             R"pbdoc(
               Initializes the Qwen3-ASR model.

               Arguments:
                 model_path: Path to the converted model directory.
                 device: Device to use (``"cpu"`` or ``"cuda"``).
                 device_index: Index of the device to use.
                 compute_type: Model weight type (e.g. ``"float16"``, ``"bfloat16"``).
                 inter_threads: Maximum number of parallel requests.
                 intra_threads: Number of OpenMP threads per request.
                 max_queued_batches: Queue capacity; 0 = unbounded.
                 flash_attention: Use Flash Attention when available.
                 tensor_parallel: Enable tensor parallelism across devices.
                 files: Dictionary of in-memory model files.
             )pbdoc")

        .def("encode", &Qwen3ASRWrapper::encode,
             py::arg("features"),
             py::arg("encoder_window_tokens") = -1,
             R"pbdoc(
               Encodes a mel spectrogram into audio feature embeddings.

               Arguments:
                 features: Float32 StorageView of shape ``(batch, 128, T)``
                           containing mel spectrogram frames.
                           All items in the batch must share the same ``T`` dimension
                           (variable-length audio in a single batch is not supported).
                 encoder_window_tokens: Encoder attention window in audio tokens.
                   ``-1`` (default) uses offline full-attention (block-diagonal mask
                   covering the entire sequence). A positive value enables streaming
                   windowed attention.

               Returns:
                 Float32 StorageView of shape ``(batch, N_audio_tokens, lm_hidden_size)``.
             )pbdoc")

        .def("generate", &Qwen3ASRWrapper::generate,
             py::arg("features"),
             py::arg("prompts"),
             py::arg("max_new_tokens") = 512,
             py::arg("beam_size") = 1,
             py::arg("patience") = 1.0f,
             py::arg("length_penalty") = 1.0f,
             py::arg("repetition_penalty") = 1.0f,
             py::arg("sampling_topk") = 1,
             py::arg("sampling_temperature") = 1.0f,
             py::arg("num_hypotheses") = 1,
             py::arg("return_scores") = false,
             py::arg("audio_token_id") = 151676,
             py::arg("encoder_window_tokens") = -1,
             R"pbdoc(
               Encodes audio and generates transcription tokens.

               Arguments:
                 features: Float32 StorageView of shape ``(batch, 128, T)``
                           containing mel spectrogram frames.
                           All items in the batch must share the same ``T`` dimension
                           (variable-length audio in a single batch is not supported).
                 prompts: List of token-ID sequences (list of list of int)
                          containing text tokens and ``audio_token_id`` placeholders.
                 max_new_tokens: Maximum number of tokens to generate.
                 beam_size: Beam size (1 = greedy decoding).
                 patience: Beam search patience factor.
                 length_penalty: Exponential length penalty for beam search.
                 repetition_penalty: Repetition penalty (> 1 penalizes repetition).
                 sampling_topk: Top-K sampling candidates (0 = full distribution).
                 sampling_temperature: Sampling temperature.
                 num_hypotheses: Number of output hypotheses to return per batch element.
                   When ``num_hypotheses > beam_size``, the actual number of returned
                   hypotheses may be less than ``num_hypotheses`` (no error is raised);
                   to guarantee N hypotheses, set ``beam_size >= num_hypotheses``.
                 return_scores: Include hypothesis log-probability scores.
                 audio_token_id: Token ID used as the audio placeholder (default 151676).
                 encoder_window_tokens: Encoder attention window in audio tokens.
                   ``-1`` (default) uses offline full-attention.

               Returns:
                 List of :class:`Qwen3ASRResult`, one per batch element.
                 Each result contains at most ``num_hypotheses`` sequences.
             )pbdoc")
        ;
    }

  }  // namespace python
}  // namespace ctranslate2
