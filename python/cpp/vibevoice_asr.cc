#include "module.h"

#include <ctranslate2/models/vibevoice_asr.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    // -----------------------------------------------------------------------
    // VibeVoiceAsr Python wrapper
    // -----------------------------------------------------------------------
    class VibeVoiceAsrWrapper : public ReplicaPoolHelper<models::VibeVoiceAsr> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      StorageView encode(const StorageView& waveform,
                         size_t chunk_size,
                         size_t audio_token_id,
                         bool add_vae_noise) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        models::VibeVoiceAsrOptions opts;
        opts.chunk_size = chunk_size;
        opts.audio_token_id = audio_token_id;
        opts.add_vae_noise = add_vae_noise;
        return _pool->encode(waveform, opts).get();
      }

      std::vector<models::VibeVoiceAsrResult>
      generate(const StorageView& input_ids,
               const StorageView& audio_features,
               size_t max_new_tokens,
               size_t beam_size,
               float patience,
               float length_penalty,
               float repetition_penalty,
               size_t sampling_topk,
               float sampling_temperature,
               size_t num_hypotheses,
               bool return_scores,
               size_t audio_token_id) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        models::VibeVoiceAsrOptions opts;
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

        auto futures = _pool->generate(input_ids, audio_features, opts);
        std::vector<models::VibeVoiceAsrResult> results;
        results.reserve(futures.size());
        for (auto& f : futures)
          results.emplace_back(f.get());
        return results;
      }
    };


    void register_vibevoice_asr(py::module& m) {
      py::class_<models::VibeVoiceAsrResult>(
        m, "VibeVoiceAsrResult",
        "A generation result from the VibeVoice-ASR model.")

        .def_readonly("sequences", &models::VibeVoiceAsrResult::sequences,
                      "Generated token sequences (list of list of str).")
        .def_readonly("sequences_ids", &models::VibeVoiceAsrResult::sequences_ids,
                      "Generated token IDs (list of list of int).")
        .def_readonly("scores", &models::VibeVoiceAsrResult::scores,
                      "Hypothesis scores (list of float, empty unless return_scores=True).")
        .def("__repr__", [](const models::VibeVoiceAsrResult& r) {
          return "VibeVoiceAsrResult(sequences="
                 + std::string(py::repr(py::cast(r.sequences))) + ")";
        })
        ;

      py::class_<VibeVoiceAsrWrapper>(
        m, "VibeVoiceAsr",
        R"pbdoc(
          VibeVoice-ASR CTranslate2 inference class.

          Loads a converted VibeVoice-ASR model directory and provides
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
               Initializes the VibeVoice-ASR model.

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

        .def("encode", &VibeVoiceAsrWrapper::encode,
             py::arg("waveform"),
             py::arg("chunk_size") = 1440000,
             py::arg("audio_token_id") = 151648,
             py::arg("add_vae_noise") = false,
             R"pbdoc(
               Encodes a raw waveform into audio feature embeddings.

               Arguments:
                 waveform: Float32 StorageView of shape ``(batch, num_samples)``
                           sampled at 24 kHz.
                 chunk_size: Samples per chunk for long-form audio (default 60s).
                 audio_token_id: Token ID used as the audio placeholder.
                 add_vae_noise: Inject VAE reparameterisation noise into the acoustic
                                latents (default ``False``). Leave ``False`` for
                                deterministic inference; set ``True`` only to reproduce
                                training-time stochastic sampling.

               Returns:
                 Float32 StorageView of shape ``(batch, audio_seq_len, lm_hidden_size)``.
             )pbdoc")

        .def("generate", &VibeVoiceAsrWrapper::generate,
             py::arg("input_ids"),
             py::arg("audio_features"),
             py::arg("max_new_tokens") = 8192,
             py::arg("beam_size") = 1,
             py::arg("patience") = 1.0f,
             py::arg("length_penalty") = 1.0f,
             py::arg("repetition_penalty") = 1.0f,
             py::arg("sampling_topk") = 1,
             py::arg("sampling_temperature") = 1.0f,
             py::arg("num_hypotheses") = 1,
             py::arg("return_scores") = false,
             py::arg("audio_token_id") = 151648,
             R"pbdoc(
               Generates transcription tokens from pre-computed audio features.

               Arguments:
                 input_ids: Int32 StorageView of shape ``(batch, seq_len)``
                            containing text tokens and audio_token_id placeholders.
                 audio_features: Output of :meth:`encode`.
                 max_new_tokens: Maximum tokens to generate.
                 beam_size: Beam size (1 = greedy).
                 patience: Beam search patience factor.
                 length_penalty: Exponential length penalty.
                 repetition_penalty: Repetition penalty.
                 sampling_topk: Top-K sampling (0 = full distribution).
                 sampling_temperature: Sampling temperature.
                 num_hypotheses: Number of output hypotheses.
                 return_scores: Include hypothesis scores in the result.
                 audio_token_id: Token ID used as the audio placeholder.

               Returns:
                 List of :class:`VibeVoiceAsrResult`, one per batch element.
             )pbdoc")
        ;
    }

  }  // namespace python
}  // namespace ctranslate2
