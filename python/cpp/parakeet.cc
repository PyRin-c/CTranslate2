#include "module.h"

#include <ctranslate2/models/parakeet.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class ParakeetWrapper : public ReplicaPoolHelper<models::Parakeet> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      StorageView encode(const StorageView& mel,
                         const StorageView& lengths,
                         const bool to_cpu) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        auto result = _pool->encode(mel, lengths).get();
        if (to_cpu)
          return result.to(Device::CPU);
        return result;
      }

      models::ParakeetResult
      transcribe(const StorageView& mel,
                 const StorageView& lengths,
                 bool use_tdt,
                 size_t max_tdt_steps) {
        models::ParakeetOptions options;
        options.use_tdt       = use_tdt;
        options.max_tdt_steps = max_tdt_steps;
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->transcribe(mel, lengths, options).get();
      }
    };


    void register_parakeet(py::module& m) {
      py::class_<models::ParakeetResult>(m, "ParakeetResult",
                                         "A transcription result from the Parakeet model.")

        .def_readonly("ids", &models::ParakeetResult::ids,
                      "Token IDs for each batch item "
                      "(list of list of int, one entry per batch element).")

        .def_readonly("token_start_frames", &models::ParakeetResult::token_start_frames,
                      "Encoder-frame index at which each token was emitted (TDT only). "
                      "Empty list for each batch item when using CTC decoding. "
                      "Multiply by the subsampling factor (8) × mel frame shift (10 ms) "
                      "to convert to seconds.")

        .def("__repr__", [](const models::ParakeetResult& r) {
          return "ParakeetResult(ids=" + std::string(py::repr(py::cast(r.ids))) + ")";
        })
        ;

      py::class_<ParakeetWrapper>(
        m, "Parakeet",
        R"pbdoc(
            Implements the NVIDIA Parakeet speech recognition model
            (parakeet-tdt_ctc family, CTC and TDT decoding).

            Example::

                import ctranslate2
                import numpy as np

                model = ctranslate2.models.Parakeet("parakeet_ct2/")
                mel   = ctranslate2.StorageView.from_array(np.zeros((1, 80, 800), dtype=np.float32))
                lens  = ctranslate2.StorageView.from_array(np.array([800], dtype=np.int32))
                result = model.transcribe(mel, lens)
                print(result.ids)
        )pbdoc")

        .def(py::init<const std::string&, const std::string&,
                      const std::variant<int, std::vector<int>>&, const StringOrMap&,
                      size_t, size_t, long, bool, bool, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("flash_attention")=false,
             py::arg("tensor_parallel")=false,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes a Parakeet model from a converted model directory.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (``cpu`` or ``cuda``).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values are: default, auto, int8,
                     int8_float32, int8_float16, int8_bfloat16, int16, float16, bfloat16,
                     float32).
                   inter_threads: Number of workers to allow executing multiple batches in
                     parallel.
                   intra_threads: Number of OpenMP threads per worker (0 for default).
                   max_queued_batches: Maximum number of batches in the worker queue
                     (-1 for unlimited, 0 for automatic).
                   flash_attention: Run model with Flash Attention 2.
                   tensor_parallel: Run model in tensor parallel mode.
                   files: Load model files from memory (dict mapping filename → bytes/file-like).
             )pbdoc")

        .def_property_readonly("device", &ParakeetWrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &ParakeetWrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &ParakeetWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &ParakeetWrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &ParakeetWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("tensor_parallel", &ParakeetWrapper::tensor_parallel,
                               "Run model with tensor parallel mode.")
        .def_property_readonly("num_active_batches", &ParakeetWrapper::num_active_batches,
                               "Number of batches waiting to be processed or currently processed.")

        .def("encode", &ParakeetWrapper::encode,
             py::arg("mel"),
             py::arg("lengths"),
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Runs the Conformer encoder on a mel spectrogram.

                 Arguments:
                   mel: Mel spectrogram, float32 array of shape
                     ``[batch, n_mels, time]`` (n_mels = 80 for the 0.6b-ja model).
                   lengths: Frame lengths for each batch item, int32 array of shape
                     ``[batch]``.
                   to_cpu: Copy the encoder output to the CPU before returning.

                 Returns:
                   Encoder output of shape ``[batch, T', 1024]``.
             )pbdoc")

        .def("transcribe", &ParakeetWrapper::transcribe,
             py::arg("mel"),
             py::arg("lengths"),
             py::kw_only(),
             py::arg("use_tdt")=false,
             py::arg("max_tdt_steps")=0,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Transcribes a batch of mel spectrograms.

                 Arguments:
                   mel: Mel spectrogram, float32 array of shape
                     ``[batch, n_mels, time]`` (n_mels = 80).
                   lengths: Frame lengths for each batch item, int32 array of shape
                     ``[batch]``.
                   use_tdt: Use the TDT decoder (default: False → CTC decoder).
                   max_tdt_steps: Maximum number of steps in TDT decoding per sequence.
                     0 means ``encoder_length * 2`` (default).

                 Returns:
                   A :class:`ctranslate2.models.ParakeetResult` containing token IDs
                   for each batch element.
             )pbdoc")

        .def("unload_model", &ParakeetWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model but keeps enough runtime context to quickly resume.

                 Arguments:
                   to_cpu: If ``True``, the model is moved to CPU memory rather than
                     fully unloaded.
             )pbdoc")

        .def("load_model", &ParakeetWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Loads the model back to the initial device.

                 Arguments:
                   keep_cache: If ``True``, the CPU model cache is kept after reloading.
             )pbdoc")

        .def_property_readonly("model_is_loaded", &ParakeetWrapper::model_is_loaded,
                               "Whether the model is loaded on the initial device and ready.")
        ;
    }

  }
}
