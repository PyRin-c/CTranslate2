#include "module.h"

#include <ctranslate2/models/gemma4.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    // -----------------------------------------------------------------------
    // Gemma4Wrapper
    // -----------------------------------------------------------------------
    class Gemma4Wrapper : public ReplicaPoolHelper<models::Gemma4> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      // Encode pixel patches and project to text embedding space.
      // pixel_values:       StorageView [batch, num_patches, 3*patch_size^2]
      // pixel_position_ids: StorageView [batch, num_patches, 2]  (int32 x/y)
      // Returns: StorageView [batch, num_patches, text_hidden_size]
      StorageView encode_vision(const StorageView& pixel_values,
                                const StorageView& pixel_position_ids) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->encode_vision(pixel_values, pixel_position_ids).get();
      }

      // Token ID of the image placeholder as stored in config.json.
      size_t image_token_id() {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        const auto* m = dynamic_cast<const models::Gemma4Model*>(model().get());
        if (!m) throw std::runtime_error("Unexpected model type");
        return m->get_image_token_id();
      }

      // End-to-end multimodal generation.
      // input_ids:     [batch, seq_len] int32 — text tokens with image placeholders.
      // vision_embeds: [batch, num_img_tokens, text_hidden] — output of encode_vision().
      std::vector<models::Gemma4GenerationResult>
      generate(const StorageView& input_ids,
               const StorageView& vision_embeds,
               size_t max_new_tokens,
               size_t beam_size,
               float patience,
               float length_penalty,
               float repetition_penalty,
               size_t sampling_topk,
               float sampling_temperature,
               size_t num_hypotheses,
               bool return_scores,
               size_t image_token_id) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        models::Gemma4GenerationOptions opts;
        opts.max_new_tokens       = max_new_tokens;
        opts.beam_size            = beam_size;
        opts.patience             = patience;
        opts.length_penalty       = length_penalty;
        opts.repetition_penalty   = repetition_penalty;
        opts.sampling_topk        = sampling_topk;
        opts.sampling_temperature = sampling_temperature;
        opts.num_hypotheses       = num_hypotheses;
        opts.return_scores        = return_scores;
        opts.image_token_id       = image_token_id;

        auto futures = _pool->generate(input_ids, vision_embeds, std::move(opts));
        std::vector<models::Gemma4GenerationResult> results;
        results.reserve(futures.size());
        for (auto& f : futures)
          results.emplace_back(f.get());
        return results;
      }
    };


    void register_gemma4(py::module& m) {

      // Gemma4GenerationResult
      py::class_<models::Gemma4GenerationResult>(
        m, "Gemma4GenerationResult",
        "A generation result from the Gemma 4 multimodal model.")

        .def_readonly("sequences", &models::Gemma4GenerationResult::sequences,
                      "Generated token sequences (list of list of str).")
        .def_readonly("sequences_ids", &models::Gemma4GenerationResult::sequences_ids,
                      "Generated token IDs (list of list of int).")
        .def_readonly("scores", &models::Gemma4GenerationResult::scores,
                      "Hypothesis scores (list of float, empty unless return_scores=True).")
        .def("__repr__", [](const models::Gemma4GenerationResult& r) {
          return "Gemma4GenerationResult(sequences="
                 + std::string(py::repr(py::cast(r.sequences))) + ")";
        })
        ;

      py::class_<Gemma4Wrapper>(
        m, "Gemma4",
        R"pbdoc(
            Gemma 4 multimodal model (vision encoder + text decoder).

            This class exposes the vision encoding pipeline:
              pixel patches → SigLIP encoder → multimodal projector → text embeddings.

            And end-to-end multimodal generation:
              encode_vision() → generate() → text tokens.

            Example::

                import ctranslate2
                import numpy as np

                model = ctranslate2.models.Gemma4("gemma4_ct2/")

                # pixel_values: [batch, num_patches, 3 * patch_size^2]
                pixel_values = np.random.randn(1, 256, 588).astype(np.float32)
                pixel_position_ids = np.zeros((1, 256, 2), dtype=np.int32)

                pv = ctranslate2.StorageView.from_array(pixel_values)
                pp = ctranslate2.StorageView.from_array(pixel_position_ids)
                vision_embeds = model.encode_vision(pv, pp)

                # input_ids: [1, seq_len] int32, with image_token_id placeholders
                input_ids = ctranslate2.StorageView.from_array(
                    np.array([[1, 262144, 262144, ...]], dtype=np.int32))
                results = model.generate(input_ids, vision_embeds,
                                         image_token_id=262144)
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
             py::kw_only(),
             py::arg("device_index") = 0,
             py::arg("compute_type") = "default",
             py::arg("inter_threads") = 1,
             py::arg("intra_threads") = 0,
             py::arg("max_queued_batches") = 0,
             py::arg("flash_attention") = false,
             py::arg("tensor_parallel") = false,
             py::arg("files") = py::none(),
             R"pbdoc(
                 Initializes a Gemma 4 multimodal model from a converted model directory.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (``"cpu"``, ``"cuda"``, or ``"auto"``).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type (possible values: ``default``, ``auto``, ``int8``,
                     ``int8_float32``, ``int8_float16``, ``int8_bfloat16``, ``int16``,
                     ``float16``, ``bfloat16``, ``float32``).
                   inter_threads: Number of workers to allow executing multiple batches in parallel.
                   intra_threads: Number of OpenMP threads per worker (0 for default).
                   max_queued_batches: Maximum number of batches in the worker queue
                     (-1 for unlimited, 0 for automatic).
                   flash_attention: Run with Flash Attention 2 for self-attention layers.
                   tensor_parallel: Run with tensor parallel mode.
                   files: Load model files from memory. A dictionary mapping file names to
                     file contents (file-like or bytes objects). When set, ``model_path``
                     acts as an identifier.
             )pbdoc")

        .def_property_readonly("device", &Gemma4Wrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &Gemma4Wrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &Gemma4Wrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &Gemma4Wrapper::num_replicas,
                               "Number of model workers backing this instance.")
        .def_property_readonly("num_queued_batches", &Gemma4Wrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("num_active_batches", &Gemma4Wrapper::num_active_batches,
                               "Number of batches currently processed or waiting.")
        .def_property_readonly("tensor_parallel", &Gemma4Wrapper::tensor_parallel,
                               "Whether tensor parallel mode is enabled.")
        .def_property_readonly("model_is_loaded", &Gemma4Wrapper::model_is_loaded,
                               "Whether the model is loaded and ready to use.")
        .def_property_readonly("image_token_id", &Gemma4Wrapper::image_token_id,
                               "Token ID of the image placeholder as stored in the model's config.json.")

        .def("encode_vision",
             &Gemma4Wrapper::encode_vision,
             py::arg("pixel_values"),
             py::arg("pixel_position_ids"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes pixel patches through the SigLIP vision encoder and
                 projects the result into the text model's hidden space.

                 Arguments:
                   pixel_values: Float array of shape
                     ``[batch_size, num_patches, 3 * patch_size^2]`` containing
                     flattened patch pixels (normalised to the model's input range).
                   pixel_position_ids: Int32 array of shape
                     ``[batch_size, num_patches, 2]`` containing the (x, y) grid
                     position index of each patch.

                 Returns:
                   A :class:`ctranslate2.StorageView` of shape
                   ``[batch_size, num_patches, text_hidden_size]`` on the same
                   device as the model.
             )pbdoc")

        .def("generate",
             &Gemma4Wrapper::generate,
             py::arg("input_ids"),
             py::arg("vision_embeds"),
             py::arg("max_new_tokens") = 256,
             py::arg("beam_size") = 1,
             py::arg("patience") = 1.0f,
             py::arg("length_penalty") = 1.0f,
             py::arg("repetition_penalty") = 1.0f,
             py::arg("sampling_topk") = 1,
             py::arg("sampling_temperature") = 1.0f,
             py::arg("num_hypotheses") = 1,
             py::arg("return_scores") = false,
             py::arg("image_token_id") = 0,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Generates text conditioned on a text prompt and vision embeddings.

                 The :attr:`input_ids` tensor should contain the tokenised prompt
                 with ``image_token_id`` placeholder tokens at every position that
                 should be replaced by a vision embedding.  The number of placeholder
                 tokens must equal ``vision_embeds.shape[1]`` (``num_img_tokens``).

                 Arguments:
                   input_ids: Int32 array of shape ``[batch_size, seq_len]``
                     on CPU.  Contains text token IDs with ``image_token_id``
                     placeholders at image positions.
                   vision_embeds: Float array of shape
                     ``[batch_size, num_img_tokens, text_hidden_size]`` —
                     typically the output of :meth:`encode_vision`.
                   max_new_tokens: Maximum number of tokens to generate beyond
                     the prompt (default 256).
                   beam_size: Beam size for beam search (1 = greedy/sampling).
                   patience: Beam search patience factor.
                   length_penalty: Length normalisation exponent.
                   repetition_penalty: Penalty > 1 discourages token repetition.
                   sampling_topk: Top-K for random sampling (0 = full vocab).
                   sampling_temperature: Temperature for random sampling.
                   num_hypotheses: Number of hypotheses to return per item.
                   return_scores: Populate :attr:`Gemma4GenerationResult.scores`.
                   image_token_id: Token ID used as the image placeholder in
                     ``input_ids``.  Must match the value stored in the model's
                     ``config.json`` (key ``image_token_id``).

                 Returns:
                   List of :class:`Gemma4GenerationResult`, one per batch item.
             )pbdoc")

        .def("unload_model",
             &Gemma4Wrapper::unload_model,
             py::arg("to_cpu") = false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Unloads the model weights, keeping enough context to reload quickly.

                 Arguments:
                   to_cpu: If ``True``, move weights to CPU memory instead of
                     fully releasing them.
             )pbdoc")

        .def("load_model",
             &Gemma4Wrapper::load_model,
             py::arg("keep_cache") = false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Reloads the model back onto the original device.

                 Arguments:
                   keep_cache: If ``True``, retain the CPU-side weight cache
                     after reloading.
             )pbdoc")
        ;
    }

  }  // namespace python
}  // namespace ctranslate2
