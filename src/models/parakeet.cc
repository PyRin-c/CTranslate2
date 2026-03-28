#include "ctranslate2/models/parakeet.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {
  namespace models {

    // =========================================================================
    // ParakeetModel
    // =========================================================================

    size_t ParakeetModel::current_spec_revision() const {
      return 1;
    }

    bool ParakeetModel::is_quantizable(const std::string& variable_name) const {
      return Model::is_quantizable(variable_name);
    }

    bool ParakeetModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name)
          && variable_name.find("embedding") == std::string::npos
          && variable_name.find("/norm")      == std::string::npos
          && variable_name.find("conv")       == std::string::npos
          && variable_name.find("pos_bias")   == std::string::npos;
    }

    std::unique_ptr<Model> ParakeetModel::clone() const {
      return std::make_unique<ParakeetModel>(*this);
    }

    void ParakeetModel::initialize(ModelReader& /*model_reader*/) {
      // No vocabulary file required: token IDs are integer indices.
    }


    // =========================================================================
    // ParakeetReplica
    // =========================================================================

    std::unique_ptr<ParakeetReplica>
    ParakeetReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const ParakeetModel*>(&model))
        throw std::invalid_argument("The model is not a ParakeetModel");
      const auto scoped = model.get_scoped_device_setter();
      const auto ptr    = model.shared_from_this();
      return std::make_unique<ParakeetReplica>(
        std::static_pointer_cast<const ParakeetModel>(ptr));
    }

    ParakeetReplica::ParakeetReplica(
        const std::shared_ptr<const ParakeetModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _encoder(std::make_unique<layers::ConformerEncoder>(*model, "encoder"))
      , _tdt_embedding(std::make_unique<layers::Embeddings>(*model, "tdt_decoder/embedding"))
      , _tdt_enc_proj(std::make_unique<layers::Dense>(*model, "tdt_decoder/joint/enc_proj"))
      , _tdt_pred_proj(std::make_unique<layers::Dense>(*model, "tdt_decoder/joint/pred_proj"))
      , _tdt_output_head(std::make_unique<layers::Dense>(*model, "tdt_decoder/joint/output_head"))
      , _lstm0_wih(model->get_variable_if_exists("tdt_decoder/lstm_layer_0/weight_ih"))
      , _lstm0_whh(model->get_variable_if_exists("tdt_decoder/lstm_layer_0/weight_hh"))
      , _lstm0_bias(model->get_variable_if_exists("tdt_decoder/lstm_layer_0/bias"))
      , _lstm1_wih(model->get_variable_if_exists("tdt_decoder/lstm_layer_1/weight_ih"))
      , _lstm1_whh(model->get_variable_if_exists("tdt_decoder/lstm_layer_1/weight_hh"))
      , _lstm1_bias(model->get_variable_if_exists("tdt_decoder/lstm_layer_1/bias"))
    {
      // CTC output projection is optional (absent in TDT-only models).
      if (model->get_variable_if_exists("ctc_decoder/output_proj/weight"))
        _ctc_output_proj = std::make_unique<layers::Dense>(*model, "ctc_decoder/output_proj");

      // Read model config from stored scalars (written by the converter).
      // Fall back to inference from weight shapes for forward compatibility.
      const StorageView* vs = model->get_variable_if_exists("vocab_size");
      const StorageView* nd = model->get_variable_if_exists("num_durations");
      // Fallback: vocab_size = number of rows in the embedding weight matrix.
      // (Embeddings::output_size() returns the embedding *dim*, not the vocab size.)
      const StorageView* embed_w = model->get_variable_if_exists("tdt_decoder/embedding/weight");
      _vocab_size    = vs ? static_cast<size_t>(vs->as_scalar<int32_t>())
                          : (embed_w ? static_cast<size_t>(embed_w->dim(0))
                                     : static_cast<size_t>(_tdt_embedding->output_size()));
      _blank_id      = _vocab_size - 1;
      _num_durations = nd ? static_cast<size_t>(nd->as_scalar<int32_t>())
                          : static_cast<size_t>(_tdt_output_head->output_size())
                            - _vocab_size;
      _pred_dim      = _lstm0_wih
                         ? static_cast<size_t>(_lstm0_wih->dim(1))
                         : static_cast<size_t>(_tdt_embedding->output_size());
      _lstm_hidden   = _lstm0_wih
                         ? static_cast<size_t>(_lstm0_wih->dim(0)) / 4
                         : _pred_dim;
      _lstm_layers   = (_lstm1_wih != nullptr) ? 2 : 1;

      // Pad output layer dimensions to preferred_size_multiple for INT8 alignment.
      const dim_t sm = model->preferred_size_multiple();
      if (sm > 1) {
        const Device dev = model->device();
        auto pad_dense = [&](layers::Dense& dense, dim_t orig_size) {
          const dim_t padded = ((orig_size + sm - 1) / sm) * sm;
          if (padded == orig_size) return;
          StorageView index({padded}, DataType::INT32, Device::CPU);
          auto* idx = index.data<int32_t>();
          for (dim_t i = 0; i < orig_size; ++i) idx[i] = static_cast<int32_t>(i);
          for (dim_t i = orig_size; i < padded; ++i) idx[i] = 0;
          if (index.device() != dev)
            index = index.to(dev);
          dense.select_weights(&index);
        };
        if (_ctc_output_proj)
          pad_dense(*_ctc_output_proj, static_cast<dim_t>(_vocab_size));
        pad_dense(*_tdt_output_head, static_cast<dim_t>(_vocab_size + _num_durations));
      }
    }

    StorageView ParakeetReplica::encode(const StorageView& mel,
                                        const StorageView& lengths,
                                        StorageView& out_lengths) {
      const Device model_dev = _model->device();
      StorageView mel_dev     = (mel.device()     == model_dev) ? mel     : mel.to(model_dev);
      StorageView lengths_dev = (lengths.device() == model_dev) ? lengths : lengths.to(model_dev);
      const DataType dtype = mel_dev.dtype();
      StorageView enc_out(dtype, model_dev);
      _encoder->operator()(mel_dev, lengths_dev, enc_out, out_lengths);
      return enc_out;
    }


    // -------------------------------------------------------------------------
    // LSTM single step (CPU float32)
    //
    //   gates = x_t @ W_ih^T + h @ W_hh^T + bias
    //   i, f, g, o = split into 4 equal slices along last dim
    //   c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
    //   h_new = sigmoid(o) * tanh(c_new)
    // -------------------------------------------------------------------------
    void ParakeetReplica::lstm_step(const StorageView& x_t,
                                    const StorageView& wih,
                                    const StorageView& whh,
                                    const StorageView& bias,
                                    StorageView& h,
                                    StorageView& c) const {
      const Device dev = x_t.device();
      const DataType dt = x_t.dtype();
      const dim_t H = static_cast<dim_t>(_lstm_hidden);

      // gates = x_t @ W_ih^T + h @ W_hh^T + bias
      const ops::Gemm gemm(/*alpha=*/1.f, /*beta=*/0.f,
                           /*trans_a=*/false, /*trans_b=*/true);
      StorageView ih(dt, dev), hh(dt, dev), gates(dt, dev);
      gemm(x_t, wih, ih);   // [1, 4H]
      gemm(h,   whh, hh);   // [1, 4H]
      ops::Add()(ih, hh, gates);

      // Add bias broadcast: bias is [4H], gates is [1, 4H]
      // Flatten gates to [4H], add, reshape back
      gates.reshape({4 * H});
      ops::Add()(gates, bias, gates);
      gates.reshape({1, 4 * H});

      // Split into 4 gate tensors each [1, H]
      StorageView gi(dt, dev), gf(dt, dev), gg(dt, dev), go_(dt, dev);
      std::vector<StorageView*> gate_parts = {&gi, &gf, &gg, &go_};
      ops::Split(-1)(gates, gate_parts);

      // Activations
      ops::Sigmoid()(gi, gi);
      ops::Sigmoid()(gf, gf);
      ops::Tanh()(gg, gg);
      ops::Sigmoid()(go_, go_);

      // c_new = f * c + i * g
      StorageView fc(dt, dev), ig(dt, dev);
      ops::Mul()(gf, c, fc);
      ops::Mul()(gi, gg, ig);
      ops::Add()(fc, ig, c);   // update c in-place

      // h_new = o * tanh(c_new)
      StorageView tanh_c(dt, dev);
      ops::Tanh()(c, tanh_c);
      ops::Mul()(go_, tanh_c, h);
    }


    // -------------------------------------------------------------------------
    // CTC greedy decode
    // -------------------------------------------------------------------------
    std::vector<size_t>
    ParakeetReplica::ctc_decode_sequence(const float* log_probs,
                                          dim_t T,
                                          dim_t vocab_size,
                                          dim_t row_stride) const {
      std::vector<size_t> ids;
      ids.reserve(static_cast<size_t>(T));

      size_t prev_id = _blank_id;
      for (dim_t t = 0; t < T; ++t) {
        const float* row = log_probs + t * row_stride;
        size_t best = 0;
        float  best_score = row[0];
        for (dim_t v = 1; v < vocab_size; ++v) {
          if (row[v] > best_score) {
            best_score = row[v];
            best = static_cast<size_t>(v);
          }
        }
        if (best != _blank_id && best != prev_id)
          ids.push_back(best);
        prev_id = best;
      }
      return ids;
    }


    // -------------------------------------------------------------------------
    // TDT greedy decode
    // blank_id = 3072 (last token, vocab_size - 1)
    // blank  → advance max(duration, 1) frames
    // token  → emit, advance duration frames (0 is valid)
    // -------------------------------------------------------------------------
    std::vector<size_t>
    ParakeetReplica::tdt_decode_sequence(const StorageView& enc_out,
                                          dim_t enc_length,
                                          size_t max_steps,
                                          std::vector<dim_t>& start_frames) const {
      const Device dev = enc_out.device();
      const DataType dt = enc_out.dtype();
      const size_t safe_max = max_steps > 0 ? max_steps
                                             : static_cast<size_t>(enc_length) * 2;

      std::vector<size_t> result;
      result.reserve(static_cast<size_t>(enc_length));
      start_frames.clear();
      start_frames.reserve(static_cast<size_t>(enc_length));

      // Zero-initialise LSTM hidden/cell states on the same device as enc_out
      auto make_zero = [&](dim_t sz) -> StorageView {
        StorageView s({1, sz}, 0.f, dev);
        return (dt != DataType::FLOAT32) ? s.to(dt) : s;
      };
      StorageView h0 = make_zero(static_cast<dim_t>(_lstm_hidden));
      StorageView c0 = make_zero(static_cast<dim_t>(_lstm_hidden));
      StorageView h1 = make_zero(static_cast<dim_t>(_lstm_hidden));
      StorageView c1 = make_zero(static_cast<dim_t>(_lstm_hidden));

      size_t prev_token = _blank_id;
      dim_t  t    = 0;
      size_t step = 0;

      // enc_out: [1, T', d_model] — batch=1 assumed for per-sequence decode
      const dim_t d_model = enc_out.dim(2);

      while (t < enc_length && step < safe_max) {
        // Slice encoder frame: [1, d_model]
        // enc_out shape: [1, T', d_model] — compute pointer offset directly.
        // enc_frame uses the same device as enc_out so that Dense projections
        // (whose weights live on dev) dispatch correctly.
        StorageView enc_frame(dt, dev);
        {
          void* ptr = static_cast<uint8_t*>(const_cast<void*>(enc_out.buffer()))
                      + static_cast<size_t>(t * d_model) * enc_out.item_size();
          enc_frame.view(ptr, {1, d_model});
        }

        // Prediction network
        // Save current hidden/cell state before LSTM — will be restored on blank
        // (NeMo semantics: LSTM state is only committed when a non-blank token is emitted)
        StorageView h0_saved(h0), c0_saved(c0);
        StorageView h1_saved(h1), c1_saved(c1);

        StorageView prev_id_sv({1}, static_cast<int32_t>(prev_token), dev);
        StorageView emb(dt, dev);
        (*_tdt_embedding)(prev_id_sv, emb);    // [1, PRED_DIM]

        if (_lstm0_wih && _lstm0_whh && _lstm0_bias)
          lstm_step(emb, *_lstm0_wih, *_lstm0_whh, *_lstm0_bias, h0, c0);
        if (_lstm1_wih && _lstm1_whh && _lstm1_bias)
          lstm_step(h0,  *_lstm1_wih, *_lstm1_whh, *_lstm1_bias, h1, c1);

        // Joint: enc_proj + pred_proj → relu → output_head
        StorageView enc_proj(dt, dev), pred_proj(dt, dev);
        (*_tdt_enc_proj)(enc_frame, enc_proj);
        (*_tdt_pred_proj)(h1, pred_proj);

        StorageView joint_in(dt, dev);
        ops::Add()(enc_proj, pred_proj, joint_in);
        ops::ReLU()(joint_in, joint_in);

        StorageView logits(dt, dev);
        (*_tdt_output_head)(joint_in, logits);  // [1, 3078]

        // Argmax on CPU (logits are small: 3078 elements)
        const StorageView logits_cpu = (dev == Device::CPU)
            ? logits
            : logits.to(Device::CPU).to_float32();
        const float* lp = logits_cpu.data<float>();
        size_t token_id = 0;
        float  token_best = lp[0];
        for (size_t v = 1; v < _vocab_size; ++v) {
          if (lp[v] > token_best) { token_best = lp[v]; token_id = v; }
        }
        size_t dur_id = 0;
        float  dur_best = lp[_vocab_size];
        for (size_t d = 1; d < _num_durations; ++d) {
          if (lp[_vocab_size + d] > dur_best) { dur_best = lp[_vocab_size + d]; dur_id = d; }
        }

        static const bool tdt_debug = (std::getenv("TDT_DEBUG") != nullptr);
        if (token_id == _blank_id) {
          // Blank: advance at least 1 frame.
          // Restore LSTM state — blank does not advance the prediction network.
          h0 = std::move(h0_saved); c0 = std::move(c0_saved);
          h1 = std::move(h1_saved); c1 = std::move(c1_saved);
          const dim_t advance = std::max(static_cast<dim_t>(dur_id), dim_t(1));
          if (tdt_debug)
            std::fprintf(stderr, "[TDT] step=%4zu t=%4d BLANK  dur=%zu advance=%d\n",
                         step, (int)t, dur_id, (int)advance);
          t += advance;
        } else {
          // Token: emit and advance by duration (0 is valid).
          // Commit the new LSTM state (saved copies are discarded automatically).
          if (tdt_debug)
            std::fprintf(stderr, "[TDT] step=%4zu t=%4d TOKEN=%5zu dur=%zu advance=%zu\n",
                         step, (int)t, token_id, dur_id, dur_id);
          start_frames.push_back(t);
          result.push_back(token_id);
          prev_token = token_id;
          t += static_cast<dim_t>(dur_id);
        }
        ++step;
      }
      return result;
    }


    // -------------------------------------------------------------------------
    // transcribe
    // -------------------------------------------------------------------------
    ParakeetResult
    ParakeetReplica::transcribe(const StorageView& mel,
                                const StorageView& lengths,
                                const ParakeetOptions& options) {
      // Move inputs to the model's device and cast mel to the model's compute dtype
      // (same pattern as Whisper: move_to(device, dtype)).
      const Device model_dev = _model->device();
      const DataType model_dt = get_default_float_type(_model->effective_compute_type());

      StorageView mel_dev = mel;
      mel_dev.move_to(model_dev, model_dt);
      const StorageView lengths_dev = (lengths.device() == model_dev) ? lengths : lengths.to(model_dev);

      const Device dev = model_dev;
      const DataType dt = mel_dev.dtype();
      const dim_t batch = mel_dev.dim(0);

      StorageView enc_lengths(DataType::INT32, Device::CPU);
      StorageView enc_out = encode(mel_dev, lengths_dev, enc_lengths);
      // enc_out: [batch, T', 1024]

      ParakeetResult result;
      result.ids.resize(static_cast<size_t>(batch));
      result.token_start_frames.resize(static_cast<size_t>(batch));

      if (!options.use_tdt) {
        // ---- CTC path ----
        StorageView logits(dt, dev);
        (*_ctc_output_proj)(enc_out, logits);   // [batch, T', 3073]

        ops::LogSoftMax()(logits, logits);

        // Greedy decode on CPU (copy once — much smaller than enc_out)
        const StorageView logits_cpu = (dev == Device::CPU)
            ? logits
            : logits.to(Device::CPU).to_float32();

        const dim_t T_max  = logits_cpu.dim(1);
        const dim_t V_stride = logits_cpu.dim(2); // may be padded (e.g. 3088)
        const dim_t V_vocab  = static_cast<dim_t>(_vocab_size);
        const float* lp   = logits_cpu.data<float>();
        const auto*  elen = enc_lengths.data<int32_t>();

        for (dim_t b = 0; b < batch; ++b) {
          result.ids[static_cast<size_t>(b)] =
              ctc_decode_sequence(lp + b * T_max * V_stride,
                                  static_cast<dim_t>(elen[b]),
                                  V_vocab,
                                  V_stride);
        }
      } else {
        // ---- TDT path ----
        const auto* elen = enc_lengths.data<int32_t>();
        const dim_t T_max  = enc_out.dim(1);

        for (dim_t b = 0; b < batch; ++b) {
          // Slice enc_out for this batch item: [1, T', 1024]
          // enc_out shape: [batch, T', d_model] — compute pointer offset directly
          const dim_t d_model = enc_out.dim(2);
          StorageView enc_b(dt, dev);
          {
            void* ptr = static_cast<uint8_t*>(const_cast<void*>(enc_out.buffer()))
                        + static_cast<size_t>(b * T_max * d_model) * enc_out.item_size();
            enc_b.view(ptr, {1, T_max, d_model});
          }
          result.ids[static_cast<size_t>(b)] =
              tdt_decode_sequence(enc_b,
                                  static_cast<dim_t>(elen[b]),
                                  options.max_tdt_steps,
                                  result.token_start_frames[static_cast<size_t>(b)]);
        }
      }
      return result;
    }


    // =========================================================================
    // Parakeet (ReplicaPool)
    // =========================================================================

    std::future<StorageView>
    Parakeet::encode(const StorageView& mel, const StorageView& lengths) {
      return post<StorageView>(
        [mel, lengths](ParakeetReplica& replica) {
          StorageView out_lengths;
          return replica.encode(mel, lengths, out_lengths);
        });
    }

    std::future<ParakeetResult>
    Parakeet::transcribe(const StorageView& mel,
                         const StorageView& lengths,
                         ParakeetOptions options) {
      return post<ParakeetResult>(
        [mel, lengths, options](ParakeetReplica& replica) {
          return replica.transcribe(mel, lengths, options);
        });
    }

  }
}
