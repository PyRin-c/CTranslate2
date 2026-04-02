// VibeVoice-ASR CTranslate2 backend implementation.
//
// Architecture (Transformers v5.4.0):
//   acoustic_tokenizer_encoder / semantic_tokenizer_encoder:
//     stem:       CausalConv1d + ConvNeXt blocks (depths[0])
//     conv_layers[0..5]: strided CausalConv1d + ConvNeXt blocks (depths[i+1])
//     head:       CausalConv1d (channels → hidden_size)
//   multi_modal_projector:
//     acoustic: linear_1 → RMSNorm → linear_2
//     semantic: linear_1 → RMSNorm → linear_2  (outputs added)
//   decoder: 28-layer Qwen2 decoder-only LM
//
// Weight name convention (matches vibevoice_asr_spec.py):
//   acoustic_tokenizer_encoder/stem/conv/weight
//   acoustic_tokenizer_encoder/stem/stage_0/norm/gamma
//   acoustic_tokenizer_encoder/stem/stage_0/mixer/weight
//   acoustic_tokenizer_encoder/stem/stage_0/gamma
//   acoustic_tokenizer_encoder/stem/stage_0/ffn_norm/gamma
//   acoustic_tokenizer_encoder/stem/stage_0/ffn_linear1/weight
//   acoustic_tokenizer_encoder/stem/stage_0/ffn_linear2/weight
//   acoustic_tokenizer_encoder/stem/stage_0/ffn_gamma
//   acoustic_tokenizer_encoder/conv_layers_0/conv/weight
//   acoustic_tokenizer_encoder/conv_layers_0/stage_0/...
//   acoustic_tokenizer_encoder/head/weight
//   semantic_tokenizer_encoder/...
//   multi_modal_projector/acoustic_linear_1/weight
//   multi_modal_projector/acoustic_norm/gamma
//   multi_modal_projector/acoustic_linear_2/weight
//   multi_modal_projector/semantic_linear_1/weight
//   multi_modal_projector/semantic_norm/gamma
//   multi_modal_projector/semantic_linear_2/weight
//   decoder/...  (standard TransformerDecoder layout)

#include "ctranslate2/models/vibevoice_asr.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "ctranslate2/decoding.h"
#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/layers/transformer.h"
#include "ctranslate2/models/model_reader.h"
#include "ctranslate2/ops/ops.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/sampling.h"
#include "../dispatch.h"

namespace ctranslate2 {
  namespace models {

    // Applies LayerScale in-place: x (*, C) *= gamma [C].
    // Uses primitives::mul_batch_broadcast for efficient broadcast multiply.
    static void apply_layer_scale(const StorageView& gamma, StorageView& x) {
      const dim_t C = gamma.dim(0);
      const dim_t total = x.size();
      DEVICE_AND_TYPE_DISPATCH(x.device(), x.dtype(),
        (primitives<D>::mul_batch_broadcast(
          gamma.data<T>(), x.data<T>(), x.data<T>(), C, total)));
    }

    // -----------------------------------------------------------------------
    // PaddingCache: stateful causal padding for chunked AcousticEncoder inference.
    //
    // Each CausalConv1d and ConvNeXtBlock mixer registers a unique cache_key.
    // On the first chunk the cache is zeros (same as non-chunked zero-padding).
    // On subsequent chunks, the real tail from the previous chunk is prepended,
    // giving correct causal context across chunk boundaries.
    //
    // Key assignment:
    //   stem conv              → "stem_conv"
    //   stem stage i mixer     → "stem_s<i>"
    //   layer j conv           → "l<j>_conv"
    //   layer j stage i mixer  → "l<j>_s<i>"
    //   head conv              → "head_conv"
    // -----------------------------------------------------------------------
    struct PaddingCache {
      std::unordered_map<std::string, StorageView> data;

      // Prepend cached tail to x along the time (last) axis, then update cache.
      // x:        (batch, channels, seq_len) — channel-first
      // key:      unique name for this conv layer within this encoder
      // left_pad: samples to prepend (= _causal_padding)
      // out:      (batch, channels, left_pad + seq_len)
      void prepend_and_update(const std::string& key,
                               const StorageView& x,
                               dim_t left_pad,
                               StorageView& out) {
        // Lazy-initialize with zeros on the first chunk.
        if (data.find(key) == data.end()) {
          StorageView zeros({x.dim(0), x.dim(1), left_pad}, x.dtype(), x.device());
          zeros.zero();
          data.emplace(key, std::move(zeros));
        }
        StorageView& cache = data.at(key);

        // out = [cache | x]
        ops::Concat(2)({&cache, &x}, out);

        // cache = last left_pad samples of out
        // (handles shortfall when seq_len < left_pad via out.dim(2) >= left_pad)
        ops::Slide(2, out.dim(2) - left_pad, left_pad)(out, cache);
      }
    };

    // -----------------------------------------------------------------------
    // ConvNeXtBlock: two-path residual block with pre-norm + LayerScale each.
    //
    // Sub-block 1 (mixer path):
    //   h = norm(x)         # RMSNorm
    //   h = mixer(h)        # depth-wise causal conv1d
    //   h = gamma * h       # LayerScale (element-wise, broadcast over batch×seq)
    //   x = x + h
    //
    // Sub-block 2 (FFN path):
    //   h = ffn_norm(x)
    //   h = ffn_linear2(gelu(ffn_linear1(h)))
    //   h = ffn_gamma * h
    //   x = x + h
    //
    // Data layout: (batch, seq_len, channels) — channel-last, consistent with
    // CTranslate2's LayerNorm and Dense conventions.
    // The mixer (depth-wise conv) is transposed to channel-first internally.
    // -----------------------------------------------------------------------
    class ConvNeXtBlock {
    public:
      // cache_key: unique identifier for this block's mixer within its PaddingCache.
      // Pass an empty string to disable cache support (non-chunked path).
      ConvNeXtBlock(const Model& model, const std::string& scope,
                    std::string cache_key = "")
        : _norm(model, scope + "/norm")
        , _ffn_norm(model, scope + "/ffn_norm")
        , _ffn_linear1(model, scope + "/ffn_linear1")
        , _ffn_linear2(model, scope + "/ffn_linear2")
        , _cache_key(std::move(cache_key))
      {
        // mixer: CausalConv1d wrapping a depth-wise nn.Conv1d.
        // Weight shape: (out_channels, 1, kernel_size) for depth-wise (groups=out_channels).
        const StorageView& w = model.get_variable(scope + "/mixer/weight");
        const dim_t out_channels = w.dim(0);
        _kernel_size = w.dim(2);
        _causal_padding = _kernel_size - 1;

        _mixer = std::make_unique<layers::Conv1D>(
          model, scope + "/mixer",
          /*stride=*/1,
          /*padding=*/0,
          /*dilation=*/1,
          /*groups=*/out_channels   // depth-wise
        );

        // LayerScale parameters: direct nn.Parameter tensors of shape [C].
        _gamma     = &model.get_variable(scope + "/gamma");
        _ffn_gamma = &model.get_variable(scope + "/ffn_gamma");
      }

      // x: (batch, seq_len, channels)  [channel-last]
      // cache: optional PaddingCache for chunked inference (null = non-chunked).
      void operator()(StorageView& x, PaddingCache* cache = nullptr) const {
        // --- Sub-block 1: mixer path ---
        {
          // Pre-norm.
          StorageView h(x.device());
          _norm(x, h);

          // Transpose to channel-first for conv: (batch, channels, seq_len).
          StorageView h_cf(x.device());
          ops::Transpose({0, 2, 1})(h, h_cf);

          // Causal padding: use cache tail if available, otherwise zeros.
          StorageView padded(x.device());
          if (cache != nullptr && _causal_padding > 0 && !_cache_key.empty())
            cache->prepend_and_update(_cache_key, h_cf, _causal_padding, padded);
          else
            _apply_causal_padding(h_cf, padded);

          // Depth-wise conv: output shape (batch, channels, seq_len).
          StorageView mixed(x.device());
          (*_mixer)(padded, mixed);

          // Transpose back to channel-last.
          StorageView mixed_cl(x.device());
          ops::Transpose({0, 2, 1})(mixed, mixed_cl);

          // LayerScale: mixed_cl *= gamma  (broadcast over batch × seq_len).
          apply_layer_scale(*_gamma, mixed_cl);

          // Residual add.
          ops::Add()(x, mixed_cl, x);
        }

        // --- Sub-block 2: FFN path (no conv, no cache needed) ---
        {
          StorageView h(x.device());
          _ffn_norm(x, h);

          StorageView after_fc1(x.device()), after_gelu(x.device()), after_fc2(x.device());
          _ffn_linear1(h, after_fc1);
          ops::GELU()(after_fc1, after_gelu);
          _ffn_linear2(after_gelu, after_fc2);

          // FFN LayerScale.
          apply_layer_scale(*_ffn_gamma, after_fc2);

          // Residual add.
          ops::Add()(x, after_fc2, x);
        }
      }

    private:
      std::unique_ptr<layers::Conv1D> _mixer;
      layers::LayerNorm _norm;
      layers::LayerNorm _ffn_norm;
      layers::Dense _ffn_linear1;
      layers::Dense _ffn_linear2;
      const StorageView* _gamma;
      const StorageView* _ffn_gamma;
      dim_t _kernel_size;
      dim_t _causal_padding;
      std::string _cache_key;

      // Prepend _causal_padding zero frames along the time (last) axis.
      // x: (batch, channels, seq_len) — channel-first.
      void _apply_causal_padding(const StorageView& x, StorageView& out) const {
        const dim_t batch    = x.dim(0);
        const dim_t channels = x.dim(1);
        StorageView zeros({batch, channels, _causal_padding}, x.dtype(), x.device());
        zeros.zero();
        ops::Concat(/*axis=*/2)({&zeros, &x}, out);
      }
    };

    // -----------------------------------------------------------------------
    // CausalConv1d: loads a single Conv1D (no padding — caller pads if needed).
    // Corresponds to VibeVoiceAcousticTokenizerCausalConv1d wrapping nn.Conv1d.
    // Weight path: scope/weight, scope/bias (optional).
    // Data layout in: (batch, channels, seq_len)  [channel-first]
    // Data layout out: (batch, channels_out, seq_len_out)
    // -----------------------------------------------------------------------
    class CausalConv1d {
    public:
      // cache_key: unique identifier for this conv within its PaddingCache.
      // Pass an empty string to disable cache support (non-chunked path).
      CausalConv1d(const Model& model, const std::string& scope,
                   dim_t stride = 1, std::string cache_key = "")
        : _stride(stride)
        , _cache_key(std::move(cache_key))
      {
        const StorageView& w = model.get_variable(scope + "/weight");
        _kernel_size = w.dim(2);
        // Correct causal padding formula (matches Python reference):
        //   causal_padding = (kernel_size - 1) * dilation - (stride - 1)
        // dilation is always 1 for all CausalConv1d in VibeVoice-ASR, so:
        //   causal_padding = kernel_size - stride
        _causal_padding = _kernel_size - _stride;

        _conv = std::make_unique<layers::Conv1D>(
          model, scope,
          /*stride=*/stride,
          /*padding=*/0
        );
      }

      // x: (batch, channels, seq_len) → out: (batch, channels_out, seq_len_out)
      // cache: optional PaddingCache for chunked inference (null = non-chunked).
      void operator()(const StorageView& x, StorageView& out,
                      PaddingCache* cache = nullptr) const {
        StorageView padded(x.device());
        if (cache != nullptr && _causal_padding > 0 && !_cache_key.empty())
          cache->prepend_and_update(_cache_key, x, _causal_padding, padded);
        else
          _apply_causal_padding(x, padded);
        (*_conv)(padded, out);
      }

    private:
      std::unique_ptr<layers::Conv1D> _conv;
      dim_t _stride;
      dim_t _kernel_size;
      dim_t _causal_padding;
      std::string _cache_key;

      void _apply_causal_padding(const StorageView& x, StorageView& out) const {
        if (_causal_padding == 0) {
          out = x;
          return;
        }
        const dim_t batch    = x.dim(0);
        const dim_t channels = x.dim(1);
        StorageView zeros({batch, channels, _causal_padding}, x.dtype(), x.device());
        zeros.zero();
        ops::Concat(/*axis=*/2)({&zeros, &x}, out);
      }
    };

    // -----------------------------------------------------------------------
    // AcousticEncoder: stem + conv_layers + head
    //
    // scope/stem/conv          → initial CausalConv1d
    // scope/stem/stage_N/...   → ConvNeXt blocks (depths[0])
    // scope/conv_layers_N/conv → downsampling CausalConv1d
    // scope/conv_layers_N/stage_M/... → ConvNeXt blocks (depths[N+1])
    // scope/head               → final CausalConv1d (channels → hidden_size)
    // -----------------------------------------------------------------------
    class AcousticEncoder {
    public:
      AcousticEncoder(const Model& model, const std::string& scope,
                      const std::vector<int>& depths,
                      int num_downsampling,
                      const std::vector<int>& downsampling_ratios)
      {
        // stem: initial conv
        _stem_conv = std::make_unique<CausalConv1d>(
          model, scope + "/stem/conv", /*stride=*/1, "stem_conv");

        // stem: ConvNeXt blocks (depths[0])
        for (int i = 0; i < depths[0]; ++i) {
          _stem_blocks.emplace_back(std::make_unique<ConvNeXtBlock>(
            model,
            scope + "/stem/stage_" + std::to_string(i),
            "stem_s" + std::to_string(i)));
        }

        // conv_layers: downsampling conv + ConvNeXt blocks
        for (int layer_i = 0; layer_i < num_downsampling; ++layer_i) {
          const std::string layer_scope =
            scope + "/conv_layers_" + std::to_string(layer_i);
          const dim_t stride = (layer_i < static_cast<int>(downsampling_ratios.size()))
                               ? downsampling_ratios[layer_i]
                               : 1;
          _layer_convs.emplace_back(
            std::make_unique<CausalConv1d>(
              model, layer_scope + "/conv", stride,
              "l" + std::to_string(layer_i) + "_conv"));

          std::vector<std::unique_ptr<ConvNeXtBlock>> blocks;
          const int ndepth = (layer_i + 1 < static_cast<int>(depths.size()))
                             ? depths[layer_i + 1]
                             : 0;
          for (int b = 0; b < ndepth; ++b) {
            blocks.emplace_back(std::make_unique<ConvNeXtBlock>(
              model,
              layer_scope + "/stage_" + std::to_string(b),
              "l" + std::to_string(layer_i) + "_s" + std::to_string(b)));
          }
          _layer_blocks.emplace_back(std::move(blocks));
        }

        // head: final projection conv
        _head = std::make_unique<CausalConv1d>(
          model, scope + "/head", /*stride=*/1, "head_conv");
      }

      // waveform: (batch, 1, num_samples) [channel-first]
      // output:   (batch, audio_seq_len, hidden_size) [channel-last]
      // cache:    optional PaddingCache for chunked inference (null = non-chunked).
      void encode(const StorageView& waveform, StorageView& output,
                  PaddingCache* cache = nullptr) const {
        // stem conv: (batch, 1, num_samples) → (batch, num_filters, T)
        StorageView x(waveform.device());
        (*_stem_conv)(waveform, x, cache);

        // stem blocks operate on (batch, seq_len, channels) — transpose first
        {
          StorageView x_cl(waveform.device());
          ops::Transpose({0, 2, 1})(x, x_cl);
          for (const auto& block : _stem_blocks)
            (*block)(x_cl, cache);
          ops::Transpose({0, 2, 1})(x_cl, x);
        }

        // conv_layers
        for (size_t i = 0; i < _layer_convs.size(); ++i) {
          // Downsampling conv (channel-first).
          StorageView ds_out(waveform.device());
          (*_layer_convs[i])(x, ds_out, cache);
          x = std::move(ds_out);

          // ConvNeXt blocks (channel-last).
          if (!_layer_blocks[i].empty()) {
            StorageView x_cl(waveform.device());
            ops::Transpose({0, 2, 1})(x, x_cl);
            for (const auto& block : _layer_blocks[i])
              (*block)(x_cl, cache);
            ops::Transpose({0, 2, 1})(x_cl, x);
          }
        }

        // head conv: (batch, channels, T) → (batch, hidden_size, T)
        StorageView head_out(waveform.device());
        (*_head)(x, head_out, cache);

        // Transpose to channel-last: (batch, T, hidden_size)
        ops::Transpose({0, 2, 1})(head_out, output);
      }

    private:
      std::unique_ptr<CausalConv1d> _stem_conv;
      std::vector<std::unique_ptr<ConvNeXtBlock>> _stem_blocks;
      std::vector<std::unique_ptr<CausalConv1d>> _layer_convs;
      std::vector<std::vector<std::unique_ptr<ConvNeXtBlock>>> _layer_blocks;
      std::unique_ptr<CausalConv1d> _head;
    };

    // -----------------------------------------------------------------------
    // MultiModalProjector: (acoustic + semantic) → lm_hidden_size
    //
    // acoustic: linear_1 → RMSNorm → linear_2
    // semantic: linear_1 → RMSNorm → linear_2
    // output = acoustic_out + semantic_out
    // -----------------------------------------------------------------------
    class MultiModalProjector {
    public:
      MultiModalProjector(const Model& model, const std::string& scope)
        : _acoustic_linear_1(model, scope + "/acoustic_linear_1")
        , _acoustic_norm(model, scope + "/acoustic_norm")
        , _acoustic_linear_2(model, scope + "/acoustic_linear_2")
        , _semantic_linear_1(model, scope + "/semantic_linear_1")
        , _semantic_norm(model, scope + "/semantic_norm")
        , _semantic_linear_2(model, scope + "/semantic_linear_2")
      {}

      // acoustic_feat: (batch, seq, acoustic_hidden)
      // semantic_feat: (batch, seq, semantic_hidden)
      // output:        (batch, seq, lm_hidden_size)
      void operator()(const StorageView& acoustic_feat,
                      const StorageView& semantic_feat,
                      StorageView& output) const {
        // Acoustic branch.
        StorageView a1(acoustic_feat.device()), an(acoustic_feat.device()), a2(acoustic_feat.device());
        _acoustic_linear_1(acoustic_feat, a1);
        _acoustic_norm(a1, an);
        _acoustic_linear_2(an, a2);

        // Semantic branch.
        StorageView s1(semantic_feat.device()), sn(semantic_feat.device()), s2(semantic_feat.device());
        _semantic_linear_1(semantic_feat, s1);
        _semantic_norm(s1, sn);
        _semantic_linear_2(sn, s2);

        // Add the two projected outputs.
        ops::Add()(a2, s2, output);
      }

    private:
      layers::Dense _acoustic_linear_1;
      layers::LayerNorm _acoustic_norm;
      layers::Dense _acoustic_linear_2;
      layers::Dense _semantic_linear_1;
      layers::LayerNorm _semantic_norm;
      layers::Dense _semantic_linear_2;
    };

    // -----------------------------------------------------------------------
    // VibeVoiceDecoder: private subclass of TransformerDecoder.
    //
    // TransformerDecoder::decode_from_embeds() is a protected method and
    // calling it from outside the class hierarchy would require adding a
    // public method to the library's ABI.  Instead we define a thin subclass
    // here (in the .cc file) that wraps the protected helper as a public
    // method.  The public header (transformer.h) is left unchanged.
    // -----------------------------------------------------------------------
    class VibeVoiceDecoder : public layers::TransformerDecoder {
    public:
      using layers::TransformerDecoder::TransformerDecoder;

      // Prefill the KV-cache with pre-computed embeddings (skips embedding
      // lookup).  Equivalent to step=0 autoregressive decode over the full
      // prefix.
      // inputs_embeds : (batch, seq_len, hidden)
      // lengths       : (batch,) int32 — valid length of each sequence
      // logits        : (batch, seq_len, vocab) output
      void forward_with_embeds(const StorageView& inputs_embeds,
                               const StorageView& lengths,
                               layers::DecoderState& state,
                               StorageView& logits) {
        decode_from_embeds(inputs_embeds, &lengths, /*step=*/0, state, &logits);
      }
    };

    // -----------------------------------------------------------------------
    // Pimpl struct
    // -----------------------------------------------------------------------
    struct VibeVoiceAsrReplica::Impl {
      std::unique_ptr<AcousticEncoder> acoustic_encoder;
      std::unique_ptr<AcousticEncoder> semantic_encoder;
      std::unique_ptr<MultiModalProjector> projector;
      std::unique_ptr<VibeVoiceDecoder> decoder;
    };

    // -----------------------------------------------------------------------
    // VibeVoiceAsrModel
    // -----------------------------------------------------------------------
    const Vocabulary& VibeVoiceAsrModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t VibeVoiceAsrModel::current_spec_revision() const {
      return 1;
    }

    bool VibeVoiceAsrModel::is_quantizable(const std::string& variable_name) const {
      return Model::is_quantizable(variable_name);
    }

    bool VibeVoiceAsrModel::is_linear_weight(const std::string& variable_name) const {
      // Non-linear parameters that must not be quantized.
      // LayerScale parameters are nn.Parameter vectors of shape [C] — not matrix weights.
      static const std::string kNonLinearSuffixes[] = {
        "/gamma",      // ConvNeXtBlock mixer LayerScale
        "/ffn_gamma",  // ConvNeXtBlock FFN LayerScale
      };
      auto ends_with = [](const std::string& s, const std::string& suffix) {
        return s.size() >= suffix.size() &&
               s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
      };
      for (const auto& suffix : kNonLinearSuffixes) {
        if (ends_with(variable_name, suffix))
          return false;
      }
      return is_quantizable(variable_name)
             && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> VibeVoiceAsrModel::clone() const {
      return std::make_unique<VibeVoiceAsrModel>(*this);
    }

    void VibeVoiceAsrModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = config.value("unk_token", "<|endoftext|>");
      vocab_info.bos_token = config.value("bos_token", "<|im_start|>");
      vocab_info.eos_token = config.value("eos_token", "<|im_end|>");

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error(
          "VibeVoiceAsrModel: Cannot load vocabulary from model directory");
    }

    // -----------------------------------------------------------------------
    // VibeVoiceAsrReplica
    // -----------------------------------------------------------------------
    std::unique_ptr<VibeVoiceAsrReplica>
    VibeVoiceAsrReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const VibeVoiceAsrModel*>(&model))
        throw std::invalid_argument("The model is not a VibeVoiceAsrModel");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete = std::static_pointer_cast<const VibeVoiceAsrModel>(model_ptr);
      return std::make_unique<VibeVoiceAsrReplica>(concrete);
    }

    VibeVoiceAsrReplica::VibeVoiceAsrReplica(
      const std::shared_ptr<const VibeVoiceAsrModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _impl(std::make_unique<Impl>())
    {
      const auto& cfg = model->config;

      std::vector<int> depths =
        cfg.value("acoustic_depths", std::vector<int>{3, 3, 3, 3, 3, 3, 8});
      std::vector<int> downsampling_ratios =
        cfg.value("acoustic_downsampling_ratios", std::vector<int>{2, 2, 4, 5, 5, 8});
      const int num_downsampling = static_cast<int>(downsampling_ratios.size());

      _impl->acoustic_encoder = std::make_unique<AcousticEncoder>(
        *model, "acoustic_tokenizer_encoder", depths, num_downsampling, downsampling_ratios);
      _impl->semantic_encoder = std::make_unique<AcousticEncoder>(
        *model, "semantic_tokenizer_encoder", depths, num_downsampling, downsampling_ratios);

      _impl->projector = std::make_unique<MultiModalProjector>(
        *model, "multi_modal_projector");

      _impl->decoder = std::make_unique<VibeVoiceDecoder>(*model, "decoder");
    }

    VibeVoiceAsrReplica::~VibeVoiceAsrReplica() = default;

    // -----------------------------------------------------------------------
    // encode: waveform → acoustic_feat + semantic_feat → projector → audio_features
    // -----------------------------------------------------------------------
    StorageView VibeVoiceAsrReplica::encode(const StorageView& waveform,
                                             const VibeVoiceAsrOptions& options) {
      PROFILE("VibeVoiceAsrReplica::encode");
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      const dim_t num_samples = waveform.dim(1);
      const dim_t chunk_size  = static_cast<dim_t>(options.chunk_size);

      // waveform: (batch, num_samples) → (batch, 1, num_samples) for channel-first conv
      StorageView wav3d = waveform;
      wav3d.reshape({wav3d.dim(0), 1, num_samples});

      // Process in chunks to bound peak memory for long-form audio.
      // PaddingCache carries causal context across chunk boundaries so each chunk
      // sees the real tail of the previous chunk instead of zeros.
      PaddingCache acoustic_cache, semantic_cache;
      std::vector<StorageView> acoustic_chunks, semantic_chunks;

      for (dim_t offset = 0; offset < num_samples; offset += chunk_size) {
        const dim_t len = std::min(chunk_size, num_samples - offset);

        // Slice chunk: (batch, 1, len)
        StorageView chunk(_model->device());
        ops::Slide(2, offset, len)(wav3d, chunk);

        StorageView ac(_model->device()), sc(_model->device());
        _impl->acoustic_encoder->encode(chunk, ac, &acoustic_cache);
        _impl->semantic_encoder->encode(chunk, sc, &semantic_cache);

        acoustic_chunks.push_back(std::move(ac));
        semantic_chunks.push_back(std::move(sc));
      }

      // Concatenate chunk outputs along the sequence dimension (dim 1, channel-last).
      StorageView acoustic_feat(_model->device());
      StorageView semantic_feat(_model->device());
      if (acoustic_chunks.size() == 1) {
        acoustic_feat = std::move(acoustic_chunks[0]);
        semantic_feat = std::move(semantic_chunks[0]);
      } else {
        std::vector<const StorageView*> ac_ptrs, sc_ptrs;
        for (const auto& c : acoustic_chunks) ac_ptrs.push_back(&c);
        for (const auto& c : semantic_chunks) sc_ptrs.push_back(&c);
        ops::Concat(1)(ac_ptrs, acoustic_feat);
        ops::Concat(1)(sc_ptrs, semantic_feat);
      }

      // VAE noise: acoustic_feat += vae_std * randn(batch) * randn_like(acoustic_feat)
      // Follows modeling_vibevoice_asr.py L347-350 (get_audio_features).
      // Only acoustic latents receive noise; semantic latents do not.
      // Falls back to config default (0.625) for models converted without this field.
      //
      // NOTE: This noise injection is disabled by default (options.add_vae_noise == false)
      // because:
      //   1. Inference should be deterministic by default.
      //   2. The float32 round-trip introduces precision loss on float16 models.
      // Enable only when reproducing training-time stochastic sampling.
      const float vae_std = _model->config.value("acoustic_vae_std", 0.625f);
      if (options.add_vae_noise && vae_std > 0.0f) {
        const DataType orig_dtype = acoustic_feat.dtype();
        StorageView af_f32 = acoustic_feat.to(DataType::FLOAT32).to(Device::CPU);
        const dim_t af_batch  = af_f32.dim(0);
        const dim_t af_seq    = af_f32.dim(1);
        const dim_t af_hidden = af_f32.dim(2);
        float* af_data = af_f32.data<float>();

        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (dim_t b = 0; b < af_batch; ++b) {
          const float noise_scale = vae_std * dist(rng);
          for (dim_t t = 0; t < af_seq; ++t) {
            for (dim_t c = 0; c < af_hidden; ++c) {
              af_data[(b * af_seq + t) * af_hidden + c] += noise_scale * dist(rng);
            }
          }
        }

        acoustic_feat = af_f32.to(orig_dtype).to(_model->device());
      }

      // Project: acoustic + semantic → lm_hidden_size
      StorageView audio_features(_model->device());
      (*_impl->projector)(acoustic_feat, semantic_feat, audio_features);
      return audio_features;
    }

    // -----------------------------------------------------------------------
    // _build_inputs_embeds: replace audio_token_id positions with audio_features
    // -----------------------------------------------------------------------
    void VibeVoiceAsrReplica::_build_inputs_embeds(
      const StorageView& input_ids,      // (batch, seq_len) int32
      const StorageView& audio_features, // (batch, audio_seq_len, lm_hidden)
      const size_t audio_token_id,
      StorageView& inputs_embeds) const  // (batch, seq_len, lm_hidden) float32
    {
      const dim_t batch   = input_ids.dim(0);
      const dim_t seq_len = input_ids.dim(1);
      const dim_t hidden  = audio_features.dim(2);

      // 1. Get text embeddings in float32 for safe element-wise injection.
      layers::Embeddings embeddings(*_model, "decoder/embeddings");
      StorageView raw_embeds(_model->device());
      embeddings(input_ids, raw_embeds);
      inputs_embeds = raw_embeds.to(DataType::FLOAT32);

      // Convert audio_features to float32 for uniform copy.
      StorageView audio_f32 = audio_features.to(DataType::FLOAT32);

      // 2. Overwrite audio_token_id positions with audio_features.
      // Both tensors are now float32 on CPU (input_ids is always on CPU).
      StorageView input_ids_cpu = input_ids.to(Device::CPU);
      StorageView inputs_embeds_cpu = inputs_embeds.to(Device::CPU);
      StorageView audio_f32_cpu = audio_f32.to(Device::CPU);

      const int32_t* ids_data = input_ids_cpu.data<int32_t>();
      float* emb_data = inputs_embeds_cpu.data<float>();
      const float* aud_data = audio_f32_cpu.data<float>();

      for (dim_t b = 0; b < batch; ++b) {
        // Count audio placeholder positions in input_ids for this batch item.
        dim_t num_audio_slots = 0;
        for (dim_t s = 0; s < seq_len; ++s) {
          if (static_cast<size_t>(ids_data[b * seq_len + s]) == audio_token_id)
            ++num_audio_slots;
        }
        const dim_t audio_len = audio_f32.dim(1);
        if (num_audio_slots != audio_len)
          throw std::runtime_error(
            "VibeVoiceAsrReplica::_build_inputs_embeds: batch item " +
            std::to_string(b) + " has " + std::to_string(num_audio_slots) +
            " audio_token_id placeholder(s) but audio_features has " +
            std::to_string(audio_len) + " frame(s). "
            "Ensure input_ids contains exactly one audio_token_id per audio frame.");

        dim_t audio_pos = 0;
        for (dim_t s = 0; s < seq_len; ++s) {
          if (static_cast<size_t>(ids_data[b * seq_len + s]) == audio_token_id) {
            const float* src = aud_data + (b * audio_len + audio_pos) * hidden;
            float* dst = emb_data + (b * seq_len + s) * hidden;
            std::copy(src, src + hidden, dst);
            ++audio_pos;
          }
        }
      }

      // Move result back to model device (decode_from_embeds will cast to model dtype).
      inputs_embeds = inputs_embeds_cpu.to(_model->device());
    }

    // -----------------------------------------------------------------------
    // generate: audio embeddings → KV-cache prefill → greedy/sampling/beam decode
    //
    // Design note — why a custom decode loop instead of src/decoding/beam_search.cc:
    // -------------------------------------------------------------------------------
    // The standard CTranslate2 decoding infrastructure (beam_search.cc / greedy_search.cc)
    // drives the decoder via Decoder::operator()(token_ids, lengths, state, logits).
    // That interface starts from token IDs and performs the embedding lookup internally.
    //
    // VibeVoice-ASR requires a two-phase approach:
    //   Phase 1 (prefill) – The prefix sequence interleaves text token embeddings with
    //     audio feature vectors at audio_token_id placeholder positions.  This CANNOT be
    //     expressed as a plain token-ID sequence; it needs pre-computed inputs_embeds.
    //     We therefore call VibeVoiceDecoder::forward_with_embeds() (which wraps the
    //     protected decode_from_embeds()), bypassing the embedding table entirely.
    //
    //   Phase 2 (autoregressive) – After the KV-cache is filled, new tokens are pure
    //     text tokens and could in principle use Decoder::operator()().  However, the
    //     standard beam_search.cc does not support resuming from a pre-populated KV-cache
    //     state produced by a different prefill path.
    //
    // TODO: Once CTranslate2 exposes a common API for embedding-prefilled generation
    //   (e.g., a GenerationCallbacks hook or an explicit "prefill then decode" split),
    //   this custom loop should be replaced to benefit from future infra improvements
    //   (speculative decoding, async batching, etc.).
    //
    // Options honoured:
    //   max_new_tokens         – upper bound on generated tokens
    //   beam_size              – 1 = greedy/sampling, >1 = beam search
    //   patience               – (beam search) extra candidate budget factor
    //   length_penalty         – score /= len^length_penalty when completing a beam
    //   repetition_penalty     – logit penalty for previously generated tokens
    //   sampling_topk          – restrict sampling to top-k (0 = full vocab)
    //   sampling_temperature   – temperature for random sampling
    //   num_hypotheses         – how many hypotheses to return per batch item
    //   return_scores          – populate VibeVoiceAsrResult::scores
    // -----------------------------------------------------------------------
    std::vector<VibeVoiceAsrResult>
    VibeVoiceAsrReplica::generate(const StorageView& input_ids,
                                   const StorageView& audio_features,
                                   const VibeVoiceAsrOptions& options) {
      PROFILE("VibeVoiceAsrReplica::generate");
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      const dim_t batch      = input_ids.dim(0);
      const dim_t prefix_len = input_ids.dim(1);
      const Vocabulary& vocab = _model->get_vocabulary();
      const size_t eos_id    = vocab.eos_id();
      const dim_t beam_size  = static_cast<dim_t>(std::max<size_t>(1, options.beam_size));
      const dim_t num_hyp    = static_cast<dim_t>(std::max<size_t>(1, options.num_hypotheses));
      // max_candidates: completed hypotheses to collect before stopping.
      // Follows CTranslate2 convention: round(beam_size * patience).
      // Clamped to at least num_hyp so completed[num_hyp-1] access is always safe.
      const size_t max_candidates = std::max(
          static_cast<size_t>(num_hyp),
          static_cast<size_t>(std::round(static_cast<float>(beam_size) * options.patience)));

      // -----------------------------------------------------------------
      // Phase 1: Build inputs_embeds (text + audio positions interleaved)
      // -----------------------------------------------------------------
      StorageView inputs_embeds(_model->device());
      _build_inputs_embeds(input_ids, audio_features,
                            options.audio_token_id, inputs_embeds);

      // -----------------------------------------------------------------
      // Phase 2: KV-cache prefill
      // -----------------------------------------------------------------
      layers::DecoderState state = _impl->decoder->initial_state();

      StorageView prefix_lengths({batch}, DataType::INT32, Device::CPU);
      for (dim_t b = 0; b < batch; ++b)
        prefix_lengths.data<int32_t>()[b] = static_cast<int32_t>(prefix_len);

      StorageView prefix_logits(_model->device());
      _impl->decoder->forward_with_embeds(
        inputs_embeds, prefix_lengths, state, prefix_logits);
      inputs_embeds.release();

      const dim_t vocab_size = prefix_logits.dim(2);

      // Slice logit at the last prefix position → (batch, vocab_size)
      StorageView last_logits(_model->device());
      {
        StorageView tmp(_model->device());
        ops::Slide(1, prefix_len - 1, 1)(prefix_logits, tmp);
        tmp.reshape({batch, vocab_size});
        last_logits = std::move(tmp);
      }
      prefix_logits.release();

      // -----------------------------------------------------------------
      // Helper: apply repetition penalty in-place.
      //   logits    : (N, vocab_size) — any device, any float dtype
      //   token_lists[i] : token IDs seen so far for row i
      // Works by moving to CPU float32, modifying, then moving back.
      // -----------------------------------------------------------------
      auto apply_rep_penalty = [&](StorageView& logits,
                                   const std::vector<std::vector<size_t>>& token_lists) {
        if (options.repetition_penalty == 1.0f || token_lists.empty())
          return;
        const Device    dev = logits.device();
        const DataType  dt  = logits.dtype();
        StorageView cpu_f32 = logits.to(Device::CPU);
        if (dt != DataType::FLOAT32)
          cpu_f32 = cpu_f32.to(DataType::FLOAT32);
        float*     data = cpu_f32.data<float>();
        const dim_t n   = cpu_f32.dim(0);
        const dim_t v   = cpu_f32.dim(1);
        for (dim_t i = 0; i < n; ++i) {
          const auto& ids = token_lists[
            static_cast<size_t>(i) < token_lists.size()
              ? static_cast<size_t>(i)
              : token_lists.size() - 1];
          for (size_t id : ids) {
            if (static_cast<dim_t>(id) < v) {
              float& logit = data[i * v + id];
              if (logit > 0.0f) logit /= options.repetition_penalty;
              else               logit *= options.repetition_penalty;
            }
          }
        }
        if (dt != DataType::FLOAT32)
          cpu_f32 = cpu_f32.to(dt);
        logits = cpu_f32.to(dev);
      };

      // =================================================================
      // GREEDY / SAMPLING  (beam_size == 1)
      // =================================================================
      if (beam_size == 1) {
        const bool use_random = (options.sampling_temperature != 1.0f
                                 || options.sampling_topk != 1);
        std::unique_ptr<Sampler> sampler_ptr;
        if (use_random)
          sampler_ptr = std::make_unique<RandomSampler>(
            static_cast<dim_t>(options.sampling_topk),
            /*topp=*/1.0f,
            options.sampling_temperature);
        else
          sampler_ptr = std::make_unique<BestSampler>();

        std::vector<std::vector<size_t>> generated(batch);
        std::vector<float> cum_log_probs(batch, 0.0f);
        std::vector<bool>  done(batch, false);

        // Process one logit frame (batch, vocab_size) → sample one token per item.
        // Takes logits by value so it can apply penalty in-place.
        auto do_sample = [&](StorageView logits) {
          apply_rep_penalty(logits, generated);

          // Pre-compute log-probs only when scores are requested.
          StorageView log_probs(_model->device());
          if (options.return_scores)
            ops::LogSoftMax()(logits, log_probs);

          // Sample — outputs must live on CPU per the Sampler contract.
          StorageView sampled_ids(DataType::INT32, Device::CPU);
          StorageView sampled_scores(logits.dtype(), Device::CPU);
          (*sampler_ptr)(logits, sampled_ids, sampled_scores, /*num_samples=*/1);
          // sampled_ids: (batch, 1), row-major → index b accesses sampled_ids.data()[b]

          // Optionally read log-prob of the chosen token.
          StorageView lp_cpu_f32;
          if (options.return_scores && log_probs) {
            lp_cpu_f32 = log_probs.to(Device::CPU);
            if (lp_cpu_f32.dtype() != DataType::FLOAT32)
              lp_cpu_f32 = lp_cpu_f32.to(DataType::FLOAT32);
          }

          for (dim_t b = 0; b < batch; ++b) {
            if (done[b]) continue;
            const size_t id =
              static_cast<size_t>(sampled_ids.data<int32_t>()[b]);
            generated[b].push_back(id);
            if (options.return_scores && lp_cpu_f32)
              cum_log_probs[b] +=
                lp_cpu_f32.data<float>()[b * vocab_size + id];
            if (id == eos_id)
              done[b] = true;
          }
        };

        // First token comes from the prefill logits.
        do_sample(std::move(last_logits));

        StorageView step_logits(_model->device());
        for (size_t t = 1; t < options.max_new_tokens; ++t) {
          if (std::all_of(done.begin(), done.end(), [](bool d) { return d; }))
            break;

          StorageView step_ids({batch}, DataType::INT32, Device::CPU);
          for (dim_t b = 0; b < batch; ++b)
            step_ids.data<int32_t>()[b] = done[b]
              ? static_cast<int32_t>(eos_id)
              : static_cast<int32_t>(generated[b].back());
          step_ids = step_ids.to(_model->device());

          (*_impl->decoder)(
            static_cast<dim_t>(prefix_len + t - 1),
            step_ids, state, &step_logits);

          do_sample(step_logits);
        }

        std::vector<VibeVoiceAsrResult> results(batch);
        for (dim_t b = 0; b < batch; ++b) {
          VibeVoiceAsrResult res;
          std::vector<std::string> tokens;
          tokens.reserve(generated[b].size());
          for (size_t id : generated[b])
            tokens.push_back(vocab.to_token(id));
          res.sequences     = {std::move(tokens)};
          res.sequences_ids = {generated[b]};
          if (options.return_scores)
            res.scores = {cum_log_probs[b]};
          results[b] = std::move(res);
        }
        return results;
      }

      // =================================================================
      // BEAM SEARCH  (beam_size > 1)
      // =================================================================
      // We expand from beam_size * 2 candidates at every step so that even
      // if half are EOS we can still keep beam_size active hypotheses.
      const dim_t num_candidates = beam_size * 2;

      // ---------- initial beam selection from last_logits ----------
      StorageView init_log_probs(_model->device());
      ops::LogSoftMax()(last_logits, init_log_probs);
      last_logits.release();

      // TopK(beam_size) per batch → (batch, beam_size) scores / ids on device
      StorageView init_scores_dev(_model->device()), init_ids_dev(_model->device());
      ops::TopK topk_init(beam_size);
      topk_init(init_log_probs, init_scores_dev, init_ids_dev);
      init_log_probs.release();

      // Move to CPU float32 for bookkeeping
      StorageView init_scores = init_scores_dev.to(Device::CPU);
      if (init_scores.dtype() != DataType::FLOAT32)
        init_scores = init_scores.to(DataType::FLOAT32);
      StorageView init_ids = init_ids_dev.to(Device::CPU);
      init_scores_dev.release();
      init_ids_dev.release();

      // ---------- beam state ----------
      // beam_tokens[b][k] : generated token sequence for batch b, beam k
      // beam_scores[b*beam_size+k] : cumulative log-prob
      struct Hypothesis { std::vector<size_t> tokens; float score; };

      std::vector<std::vector<std::vector<size_t>>> beam_tokens(
        batch, std::vector<std::vector<size_t>>(beam_size));
      std::vector<float> beam_scores(
        batch * beam_size, std::numeric_limits<float>::lowest());
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
            // Keep beam score at -inf so this slot is dead.
          } else {
            beam_scores[b * beam_size + k] = sc;
          }
        }
        // Check immediate early-exit (all top beams were EOS).
        if (completed[b].size() >= max_candidates) {
          std::sort(completed[b].begin(), completed[b].end(),
                    [](const auto& a, const auto& b) {
                      return a.score > b.score; });
          float best_active = std::numeric_limits<float>::lowest();
          for (dim_t k = 0; k < beam_size; ++k)
            best_active = std::max(best_active,
                                   beam_scores[b * beam_size + k]);
          if (best_active < completed[b][num_hyp - 1].score)
            batch_done[b] = true;
        }
      }

      // Expand KV cache: (batch,) → (batch * beam_size,)
      // Cast to base Decoder* to access replicate_state(state, beam_size)
      // which is hidden by TransformerDecoder::replicate_state(string) override.
      static_cast<layers::Decoder*>(_impl->decoder.get())->replicate_state(state, beam_size);

      // ---------- beam search loop ----------
      // At loop iteration t (1-indexed), we feed token at index (t-1) from
      // each beam at decoder position (prefix_len + t - 1).
      StorageView step_logits(_model->device());

      for (size_t t = 1; t < options.max_new_tokens; ++t) {
        if (std::all_of(batch_done.begin(), batch_done.end(),
                        [](bool d) { return d; }))
          break;

        // Build step_ids: (batch * beam_size,) — last token of each beam.
        StorageView step_ids({batch * beam_size}, DataType::INT32, Device::CPU);
        for (dim_t b = 0; b < batch; ++b)
          for (dim_t k = 0; k < beam_size; ++k) {
            const size_t last_tok = beam_tokens[b][k].empty()
              ? eos_id : beam_tokens[b][k].back();
            step_ids.data<int32_t>()[b * beam_size + k] =
              static_cast<int32_t>(last_tok);
          }
        step_ids = step_ids.to(_model->device());

        // Decoder forward: → step_logits (batch * beam_size, vocab_size)
        (*_impl->decoder)(
          static_cast<dim_t>(prefix_len + t - 1),
          step_ids, state, &step_logits);

        // Repetition penalty (flat beam → token_list mapping).
        if (options.repetition_penalty != 1.0f) {
          std::vector<std::vector<size_t>> flat_tokens(batch * beam_size);
          for (dim_t b = 0; b < batch; ++b)
            for (dim_t k = 0; k < beam_size; ++k)
              flat_tokens[b * beam_size + k] = beam_tokens[b][k];
          apply_rep_penalty(step_logits, flat_tokens);
        }

        // Log-softmax in-place: (batch * beam_size, vocab_size)
        ops::LogSoftMax()(step_logits);

        // Move to CPU float32 for beam scoring.
        StorageView lp_cpu = step_logits.to(Device::CPU);
        if (lp_cpu.dtype() != DataType::FLOAT32)
          lp_cpu = lp_cpu.to(DataType::FLOAT32);
        float* lp_data = lp_cpu.data<float>();

        // Add cumulative beam scores to each row (broadcast over vocab).
        // Dead beams (score == -inf) are filled with -inf to prevent selection.
        const float neg_inf = std::numeric_limits<float>::lowest();
        for (dim_t i = 0; i < batch * beam_size; ++i) {
          const float bs = beam_scores[i];
          if (bs <= neg_inf / 2.0f) {
            for (dim_t v = 0; v < vocab_size; ++v)
              lp_data[i * vocab_size + v] = neg_inf;
          } else {
            for (dim_t v = 0; v < vocab_size; ++v)
              lp_data[i * vocab_size + v] += bs;
          }
        }

        // Reshape to (batch, beam_size * vocab_size) and TopK(num_candidates).
        // Each flat index fi in [0, beam_size * vocab_size):
        //   beam_origin = fi / vocab_size,  token_id = fi % vocab_size
        lp_cpu.reshape({batch, beam_size * vocab_size});
        StorageView topk_vals(DataType::FLOAT32, Device::CPU);
        StorageView topk_flat_ids(DataType::INT32,    Device::CPU);
        ops::TopK topk_cand(num_candidates);
        topk_cand(lp_cpu, topk_vals, topk_flat_ids);
        // topk_vals:     (batch, num_candidates) float32 CPU
        // topk_flat_ids: (batch, num_candidates) int32  CPU

        const float* tv_data  = topk_vals.data<float>();
        const int32_t* ti_data = topk_flat_ids.data<int32_t>();

        // New beam state for this step.
        std::vector<std::vector<std::vector<size_t>>> new_beam_tokens(
          batch, std::vector<std::vector<size_t>>(beam_size));
        std::vector<float>   new_beam_scores(batch * beam_size, neg_inf);
        std::vector<int32_t> gather_idx(batch * beam_size, 0);

        for (dim_t b = 0; b < batch; ++b) {
          if (batch_done[b]) {
            // Keep existing slots as dead dummies (gather from self).
            for (dim_t k = 0; k < beam_size; ++k) {
              new_beam_tokens[b][k] = beam_tokens[b][k];
              gather_idx[b * beam_size + k] = b * beam_size + k;
            }
            continue;
          }

          dim_t next_k = 0;
          for (dim_t ci = 0; ci < num_candidates && next_k < beam_size; ++ci) {
            const float   cand_score   = tv_data [b * num_candidates + ci];
            const int32_t flat_id      = ti_data [b * num_candidates + ci];
            const dim_t   beam_origin  = flat_id / vocab_size;
            const dim_t   token_id     = flat_id % vocab_size;

            if (cand_score <= neg_inf / 2.0f)
              break;  // All remaining candidates are dead.

            if (token_id == static_cast<dim_t>(eos_id)) {
              const float seq_len =
                static_cast<float>(beam_tokens[b][beam_origin].size() + 1);
              const float lp = (options.length_penalty != 0.0f)
                ? std::pow(seq_len, options.length_penalty) : 1.0f;
              completed[b].push_back(
                {beam_tokens[b][beam_origin], cand_score / lp});
            } else {
              new_beam_tokens[b][next_k] = beam_tokens[b][beam_origin];
              new_beam_tokens[b][next_k].push_back(
                static_cast<size_t>(token_id));
              new_beam_scores[b * beam_size + next_k] = cand_score;
              gather_idx    [b * beam_size + next_k]  =
                b * beam_size + beam_origin;
              ++next_k;
            }
          }

          // Fill any unfilled slots with dead dummies.
          for (; next_k < beam_size; ++next_k) {
            new_beam_tokens[b][next_k] = {};
            gather_idx[b * beam_size + next_k] = b * beam_size + 0;
          }

          // Early-exit: if we have enough completed hypotheses and the best
          // remaining active beam cannot beat the worst completed hypothesis.
          if (completed[b].size() >= max_candidates) {
            std::sort(completed[b].begin(), completed[b].end(),
                      [](const auto& a, const auto& b) {
                        return a.score > b.score; });
            const float best_active = new_beam_scores[b * beam_size];
            if (best_active <= neg_inf / 2.0f ||
                best_active < completed[b][num_hyp - 1].score)
              batch_done[b] = true;
          }
        }

        // Reorder KV cache according to beam origins.
        StorageView gather_indices({batch * beam_size}, DataType::INT32, Device::CPU);
        std::copy(gather_idx.begin(), gather_idx.end(),
                  gather_indices.data<int32_t>());
        gather_indices = gather_indices.to(_model->device());
        _impl->decoder->update_state(state, std::move(gather_indices), beam_size,
                                      /*alive_batches=*/nullptr);

        beam_tokens  = std::move(new_beam_tokens);
        beam_scores  = std::move(new_beam_scores);
      }

      // ---------- collect active beams as completed (force-finish) ----------
      for (dim_t b = 0; b < batch; ++b) {
        if (batch_done[b]) continue;
        for (dim_t k = 0; k < beam_size; ++k) {
          const float sc = beam_scores[b * beam_size + k];
          if (sc <= std::numeric_limits<float>::lowest() / 2.0f ||
              beam_tokens[b][k].empty())
            continue;
          const float seq_len = static_cast<float>(beam_tokens[b][k].size());
          const float lp = (options.length_penalty != 0.0f)
            ? std::pow(seq_len, options.length_penalty) : 1.0f;
          completed[b].push_back({beam_tokens[b][k], sc / lp});
        }
      }

      // ---------- build results ----------
      std::vector<VibeVoiceAsrResult> results(batch);
      for (dim_t b = 0; b < batch; ++b) {
        std::sort(completed[b].begin(), completed[b].end(),
                  [](const auto& a, const auto& b) {
                    return a.score > b.score; });
        VibeVoiceAsrResult res;
        const size_t n =
          std::min(completed[b].size(), static_cast<size_t>(num_hyp));
        res.sequences.reserve(n);
        res.sequences_ids.reserve(n);
        if (options.return_scores) res.scores.reserve(n);
        for (size_t h = 0; h < n; ++h) {
          std::vector<std::string> tokens;
          tokens.reserve(completed[b][h].tokens.size());
          for (size_t id : completed[b][h].tokens)
            tokens.push_back(vocab.to_token(id));
          res.sequences.push_back(std::move(tokens));
          res.sequences_ids.push_back(completed[b][h].tokens);
          if (options.return_scores)
            res.scores.push_back(completed[b][h].score);
        }
        results[b] = std::move(res);
      }
      return results;
    }

    // -----------------------------------------------------------------------
    // VibeVoiceAsr pool
    // -----------------------------------------------------------------------
    std::future<StorageView>
    VibeVoiceAsr::encode(const StorageView& waveform,
                          const VibeVoiceAsrOptions& options) {
      return post<StorageView>(
        [waveform = waveform.sync_copy(), options]
        (VibeVoiceAsrReplica& replica) mutable {
          return replica.encode(waveform, options);
        });
    }

    std::vector<std::future<VibeVoiceAsrResult>>
    VibeVoiceAsr::generate(const StorageView& input_ids,
                            const StorageView& audio_features,
                            VibeVoiceAsrOptions options) {
      const size_t batch_size = input_ids.dim(0);
      return post_batch<VibeVoiceAsrResult>(
        [input_ids = input_ids.sync_copy(),
         audio_features = audio_features.sync_copy(),
         options = std::move(options)]
        (VibeVoiceAsrReplica& replica) mutable {
          return replica.generate(input_ids, audio_features, options);
        },
        batch_size);
    }

  }  // namespace models
}  // namespace ctranslate2
