#include "ctranslate2/layers/conformer.h"

#include <cmath>
#include <stdexcept>

#include "ctranslate2/ops/ops.h"

namespace ctranslate2 {
  namespace layers {

    // Activation type singletons used as pointer parameters in Dense().
    static const ops::ActivationType kSwish   = ops::ActivationType::Swish;

    // =========================================================================
    // ConvModule
    // =========================================================================

    ConvModule::ConvModule(const models::Model& model, const std::string& scope)
      : _norm(model, scope + "/norm")
      , _pointwise_conv1(model, scope + "/pointwise_conv1")
      , _depthwise_conv(model, scope + "/depthwise_conv",
                        /*stride=*/1,
                        /*padding=*/4,    // kernel=9, same-length: pad = (9-1)/2 = 4
                        /*dilation=*/1,
                        /*groups=*/1024)  // depthwise: groups = d_model
      , _pointwise_conv2(model, scope + "/pointwise_conv2")
    {
    }

    void ConvModule::operator()(const StorageView& input, StorageView& output) const {
      const Device device = input.device();
      const DataType dtype = input.dtype();

      // Step 1: LayerNorm
      StorageView x_norm(dtype, device);
      _norm(input, x_norm);

      // Step 2: pointwise_conv1 → [batch, T, 2*d_model]
      StorageView z(dtype, device);
      _pointwise_conv1(x_norm, z);

      // Step 3: GLU  A * sigmoid(B)  (gate is sigmoid, NOT SiLU)
      StorageView A(dtype, device), B(dtype, device);
      ops::Split(-1)(z, A, B);       // equal split along last dim
      ops::Sigmoid()(B, B);
      ops::Mul()(A, B, z);           // reuse z for gate output [batch, T, d_model]

      // Step 4: transpose → [batch, d_model, T]
      StorageView g_t(dtype, device);
      ops::Transpose({0, 2, 1})(z, g_t);

      // Step 5: depthwise_conv (BN already fused into weights)
      StorageView c(dtype, device);
      _depthwise_conv(g_t, c);           // [batch, d_model, T]

      // Step 6: SiLU (Swish) — BN is after depthwise_conv, Swish is after BN
      ops::Swish()(c, c);

      // Step 7: transpose back → [batch, T, d_model]
      StorageView s_t(dtype, device);
      ops::Transpose({0, 2, 1})(c, s_t);

      // Step 8: pointwise_conv2
      StorageView y(dtype, device);
      _pointwise_conv2(s_t, y);

      // Step 9: residual add
      ops::Add()(input, y, output);
    }


    // =========================================================================
    // RelativePositionMultiHeadAttention
    // =========================================================================

    RelativePositionMultiHeadAttention::RelativePositionMultiHeadAttention(
        const models::Model& model,
        const std::string& scope,
        dim_t num_heads)
      : _num_heads(num_heads)
      , _head_dim(1024 / num_heads)
      , _linear_q(model, scope + "/linear_q")
      , _linear_k(model, scope + "/linear_k")
      , _linear_v(model, scope + "/linear_v")
      , _linear_out(model, scope + "/linear_out")
      , _linear_pos(model, scope + "/linear_pos")
      , _pos_bias_u(model.get_variable(scope + "/pos_bias_u"))
      , _pos_bias_v(model.get_variable(scope + "/pos_bias_v"))
    {
    }

    void RelativePositionMultiHeadAttention::compute_pos_encoding(
        dim_t seq_len,
        DataType /*dtype*/,
        Device /*device*/,
        StorageView& pos_enc) const {
      // [2T-1, d_model] sinusoidal encoding, built in float32 on CPU.
      const dim_t d_model = _num_heads * _head_dim;
      const dim_t length  = 2 * seq_len - 1;

      pos_enc.resize({length, d_model});
      auto* data = pos_enc.data<float>();

      for (dim_t idx = 0; idx < length; ++idx) {
        const float pos = static_cast<float>((seq_len - 1) - idx);
        for (dim_t i = 0; i < d_model; i += 2) {
          const float denom = std::pow(10000.f, static_cast<float>(i) / d_model);
          data[idx * d_model + i] = std::sin(pos / denom);
          if (i + 1 < d_model)
            data[idx * d_model + i + 1] = std::cos(pos / denom);
        }
      }
    }

    void RelativePositionMultiHeadAttention::operator()(
        const StorageView& input,
        StorageView& output,
        const StorageView* lengths) const {
      const Device   device  = input.device();
      const DataType dtype   = input.dtype();
      const dim_t    batch   = input.dim(0);
      const dim_t    T       = input.dim(1);
      const dim_t    d_model = static_cast<dim_t>(_num_heads * _head_dim);

      // ── Q, K, V projections [B, T, d_model] ──────────────────────────
      StorageView Q(dtype, device), K(dtype, device), V(dtype, device);
      _linear_q(input, Q);
      _linear_k(input, K);
      _linear_v(input, V);

      // ── Add pos_bias [H, head_dim] to Q [B, T, d_model] ─────────────
      // Reshape bias → [1, 1, d_model], tile → [B, T, d_model], add.
      // ops::Tile and ops::Add are device/dtype agnostic.
      auto apply_bias = [&](const StorageView& bias) -> StorageView {
        StorageView b(bias);
        b.move_to(device, dtype);
        b.reshape({1, 1, d_model});
        ops::Tile(0, batch)(b);   // → [B, 1, d_model]
        ops::Tile(1, T)(b);       // → [B, T, d_model]
        StorageView out(dtype, device);
        ops::Add()(Q, b, out);
        return out;
      };
      StorageView Q_u = apply_bias(_pos_bias_u);   // [B, T, d_model]
      StorageView Q_v = apply_bias(_pos_bias_v);   // [B, T, d_model]

      // ── Split heads: [B, T, d_model] → [B, H, T, head_dim] ──────────
      auto to_heads = [&](StorageView x) -> StorageView {
        x.reshape({batch, T, _num_heads, _head_dim});
        StorageView xt(dtype, device);
        ops::Transpose({0, 2, 1, 3})(x, xt);
        return xt;
      };
      StorageView Quh = to_heads(std::move(Q_u));
      StorageView Qvh = to_heads(std::move(Q_v));
      StorageView Kh  = to_heads(std::move(K));
      StorageView Vh  = to_heads(std::move(V));

      // ── Content score: [B*H, T, D] @ [B*H, D, T] → [B*H, T, T] ─────
      Quh.reshape({batch * _num_heads, T, _head_dim});
      Kh.reshape( {batch * _num_heads, T, _head_dim});
      StorageView score_content(dtype, device);
      ops::MatMul(false, true)(Quh, Kh, score_content);
      score_content.reshape({batch, _num_heads, T, T});

      // ── Sinusoidal PE → linear_pos → position encoding matrix P ──────
      // PE is always built in float32 on CPU, then cast to model dtype/device.
      StorageView pos_enc_f32(DataType::FLOAT32, Device::CPU);
      compute_pos_encoding(T, DataType::FLOAT32, Device::CPU, pos_enc_f32);
      StorageView pos_enc_dev(pos_enc_f32);
      pos_enc_dev.move_to(device, dtype);         // [2T-1, d_model]
      StorageView P(dtype, device);
      _linear_pos(pos_enc_dev, P);                // [2T-1, d_model]

      // P → [H, 2T-1, head_dim] → [1, H, 2T-1, head_dim]
      // Tile batch → [B, H, 2T-1, head_dim] → reshape [B*H, 2T-1, head_dim]
      P.reshape({2 * T - 1, _num_heads, _head_dim});
      StorageView Ph(dtype, device);
      ops::Transpose({1, 0, 2})(P, Ph);           // [H, 2T-1, head_dim]
      Ph.reshape({1, _num_heads, 2 * T - 1, _head_dim});
      ops::Tile(0, batch)(Ph);                    // [B, H, 2T-1, head_dim]
      Ph.reshape({batch * _num_heads, 2 * T - 1, _head_dim});

      // ── Position score: [B*H, T, D] @ [B*H, D, 2T-1] → [B*H, T, 2T-1]
      Qvh.reshape({batch * _num_heads, T, _head_dim});
      StorageView score_pos_raw(dtype, device);
      ops::MatMul(false, true)(Qvh, Ph, score_pos_raw);
      score_pos_raw.reshape({batch, _num_heads, T, 2 * T - 1});

      // ── rel_shift: [B, H, T, 2T-1] → [B, H, T, T] ───────────────────
      StorageView score_pos(dtype, device);
      _rel_shift(score_pos_raw, score_pos);

      // ── Combine scores and scale ──────────────────────────────────────
      ops::Add()(score_content, score_pos, score_content);
      // Scale scalar must match tensor dtype (ops dispatch is type-sensitive).
      StorageView scale_sv(1.f / std::sqrt(static_cast<float>(_head_dim)));
      scale_sv.move_to(Device::CPU, dtype);
      ops::Mul()(score_content, scale_sv, score_content);

      // ── Attention mask (CPU float32, convert score if needed) ─────────
      // ops::SoftMax with lengths does not handle [B,H,T,T]+[B] shapes,
      // so we apply the mask manually in float32 and call SoftMax without lengths.
      if (lengths && lengths->dtype() == DataType::INT32) {
        StorageView sc_f32(score_content);
        sc_f32.move_to(Device::CPU, DataType::FLOAT32);
        const StorageView len_cpu = (lengths->device() == Device::CPU)
            ? *lengths : lengths->to(Device::CPU);
        const auto* len_ptr = len_cpu.data<int32_t>();
        float* sc_ptr = sc_f32.data<float>();
        for (dim_t b = 0; b < batch; ++b) {
          const dim_t valid = static_cast<dim_t>(len_ptr[b]);
          for (dim_t h = 0; h < _num_heads; ++h)
            for (dim_t i = 0; i < T; ++i) {
              float* row = sc_ptr + ((b * _num_heads + h) * T + i) * T;
              for (dim_t j = valid; j < T; ++j)
                row[j] = -1e9f;
            }
        }
        score_content = std::move(sc_f32);
        score_content.move_to(device, dtype);
      }

      // ── Softmax (device/dtype agnostic, no lengths — already masked) ──
      StorageView attn_weights(dtype, device);
      ops::SoftMax()(score_content, attn_weights);

      // ── Weighted sum: attn @ V → [B*H, T, head_dim] ──────────────────
      attn_weights.reshape({batch * _num_heads, T, T});
      Vh.reshape({batch * _num_heads, T, _head_dim});
      StorageView attn_out(dtype, device);
      ops::MatMul()(attn_weights, Vh, attn_out);

      // ── Merge heads: [B*H, T, head_dim] → [B, T, d_model] ───────────
      attn_out.reshape({batch, _num_heads, T, _head_dim});
      StorageView attn_t(dtype, device);
      ops::Transpose({0, 2, 1, 3})(attn_out, attn_t);  // [B, T, H, head_dim]
      attn_t.reshape({batch, T, d_model});

      // ── Output projection ─────────────────────────────────────────────
      _linear_out(attn_t, output);
    }


    // =========================================================================
    // ConformerBlock
    // =========================================================================

    ConformerBlock::ConformerBlock(const models::Model& model,
                                   const std::string& scope,
                                   dim_t num_heads)
      : _norm_ff1(model, scope + "/norm_ff1")
      , _ff1_linear1(model, scope + "/ff1_linear1", &kSwish)
      , _ff1_linear2(model, scope + "/ff1_linear2")
      , _norm_mha(model, scope + "/norm_mha")
      , _mha(model, scope + "/mha", num_heads)
      , _conv(model, scope + "/conv")
      , _norm_ff2(model, scope + "/norm_ff2")
      , _ff2_linear1(model, scope + "/ff2_linear1", &kSwish)
      , _ff2_linear2(model, scope + "/ff2_linear2")
      , _norm_out(model, scope + "/norm_out")
    {
    }

    void ConformerBlock::operator()(const StorageView& input,
                                    StorageView& output,
                                    const StorageView* lengths) const {
      const Device device = input.device();
      const DataType dtype = input.dtype();

      // Scalar 0.5 for Macaron half-scale residual — must match tensor dtype.
      StorageView half_val(0.5f);
      half_val.move_to(Device::CPU, dtype);

      // --- FFN1 (Macaron half-scale) ---
      StorageView x(dtype, device);
      {
        StorageView norm_out(dtype, device);
        _norm_ff1(input, norm_out);
        StorageView ff_inner(dtype, device), ff_out(dtype, device);
        _ff1_linear1(norm_out, ff_inner);    // SiLU is fused into Dense
        _ff1_linear2(ff_inner, ff_out);
        ops::Mul()(ff_out, half_val, ff_out);
        ops::Add()(input, ff_out, x);
      }

      // --- RelPos MHA ---
      {
        StorageView norm_out(dtype, device), mha_out(dtype, device);
        _norm_mha(x, norm_out);
        _mha(norm_out, mha_out, lengths);
        StorageView tmp(dtype, device);
        ops::Add()(x, mha_out, tmp);
        x = std::move(tmp);
      }

      // --- ConvModule ---
      {
        StorageView conv_out(dtype, device);
        _conv(x, conv_out);
        x = std::move(conv_out);
      }

      // --- FFN2 (Macaron half-scale) ---
      {
        StorageView norm_out(dtype, device), ff_inner(dtype, device), ff_out(dtype, device);
        _norm_ff2(x, norm_out);
        _ff2_linear1(norm_out, ff_inner);
        _ff2_linear2(ff_inner, ff_out);
        ops::Mul()(ff_out, half_val, ff_out);
        ops::Add()(x, ff_out, x);
      }

      // --- Output LayerNorm ---
      _norm_out(x, output);
    }


    // =========================================================================
    // ConformerPreEncoder
    // =========================================================================

    ConformerPreEncoder::ConformerPreEncoder(const models::Model& model,
                                             const std::string& scope)
      : _conv0_w(model.get_variable(scope + "/conv0/weight"))
      , _conv0_b(model.get_variable(scope + "/conv0/bias"))
      , _conv2_w(model.get_variable(scope + "/conv2/weight"))
      , _conv2_b(model.get_variable(scope + "/conv2/bias"))
      , _conv3_w(model.get_variable(scope + "/conv3/weight"))
      , _conv3_b(model.get_variable(scope + "/conv3/bias"))
      , _conv5_w(model.get_variable(scope + "/conv5/weight"))
      , _conv5_b(model.get_variable(scope + "/conv5/bias"))
      , _conv6_w(model.get_variable(scope + "/conv6/weight"))
      , _conv6_b(model.get_variable(scope + "/conv6/bias"))
      , _out(model, scope + "/out")
    {
    }

    void ConformerPreEncoder::operator()(const StorageView& input,
                                         const StorageView& lengths,
                                         StorageView& output,
                                         StorageView& out_lengths) const {
      const Device device = input.device();
      const DataType dtype = input.dtype();

      // NeMo transposes [batch, 80, T] → [batch, T, 80] before subsampling, then unsqueezes
      // to [batch, 1, T, 80]. We must match that layout so conv weights are applied correctly.
      StorageView x_tp(dtype, device);
      ops::Transpose({0, 2, 1})(input, x_tp);   // [batch, T, 80]
      StorageView x(dtype, device);
      ops::Unsqueeze({size_t(1)})(x_tp, x);      // [batch, 1, T, 80]

      const ops::ReLU relu_op;

      // conv0: Conv2d(1→256, 3×3, stride=2, pad=1)
      {
        ops::Conv2D conv(STRIDE, STRIDE, PADDING, PADDING, 1);
        StorageView y(dtype, device);
        conv(x, _conv0_w, _conv0_b, y);
        relu_op(y, x);
      }

      // conv2: DepthwiseConv2d  +  conv3: PointwiseConv2d
      {
        ops::Conv2D dw(STRIDE, STRIDE, PADDING, PADDING, GROUPS_DW);
        StorageView y(dtype, device);
        dw(x, _conv2_w, _conv2_b, y);
        ops::Conv2D pw(1, 1, 0, 0, 1);
        pw(y, _conv3_w, _conv3_b, x);
        relu_op(x, x);
      }

      // conv5: DepthwiseConv2d  +  conv6: PointwiseConv2d
      {
        ops::Conv2D dw(STRIDE, STRIDE, PADDING, PADDING, GROUPS_DW);
        StorageView y(dtype, device);
        dw(x, _conv5_w, _conv5_b, y);
        ops::Conv2D pw(1, 1, 0, 0, 1);
        pw(y, _conv6_w, _conv6_b, x);
        relu_op(x, x);
      }

      // x: [batch, 256, T', 10]  (time is dim2, freq is dim3 — matching NeMo layout)
      // Reshape: [batch, ch, frames, freq] → [batch, frames, ch*freq]
      const dim_t batch  = x.dim(0);
      const dim_t ch     = x.dim(1);  // 256
      const dim_t frames = x.dim(2);  // T'
      const dim_t freq   = x.dim(3);  // 10

      // Transpose to [batch, frames, ch, freq] then flatten last 2 dims
      StorageView x_t(dtype, device);
      ops::Transpose({0, 2, 1, 3})(x, x_t);        // [batch, frames, 256, 10]
      x_t.reshape({batch, frames, ch * freq});       // [batch, frames, 2560]

      _out(x_t, output);   // [batch, frames, 1024]

      // Compute output lengths: three subsampling stages (always on CPU)
      out_lengths = StorageView({input.dim(0)}, int32_t(0), Device::CPU);
      const StorageView lengths_cpu = (lengths.device() == Device::CPU)
          ? lengths : lengths.to(Device::CPU);
      const auto* in_len_ptr  = lengths_cpu.data<int32_t>();
      auto*       out_len_ptr = out_lengths.data<int32_t>();
      for (dim_t b = 0; b < input.dim(0); ++b) {
        dim_t l = in_len_ptr[b];
        for (int stage = 0; stage < 3; ++stage)
          l = subsample_length(l, 3, 2, 1);
        out_len_ptr[b] = static_cast<int32_t>(l);
      }
    }


    // =========================================================================
    // ConformerEncoder
    // =========================================================================

    ConformerEncoder::ConformerEncoder(const models::Model& model,
                                       const std::string& scope,
                                       dim_t num_heads)
      : _pre_encode(model, scope + "/pre_encode")
      , _d_model(1024)  // Parakeet encoder hidden size (enc_hidden in NeMo model config)
    {
      for (size_t i = 0;; ++i) {
        const std::string layer_scope = scope + "/layer_" + std::to_string(i);
        if (i == 0) {
          _layers.emplace_back(
            std::make_unique<ConformerBlock>(model, layer_scope, num_heads));
        } else {
          if (!model.layer_exists(layer_scope))
            break;
          _layers.emplace_back(
            std::make_unique<ConformerBlock>(model, layer_scope, num_heads));
        }
      }
    }

    void ConformerEncoder::operator()(const StorageView& input,
                                      const StorageView& lengths,
                                      StorageView& output,
                                      StorageView& out_lengths) const {
      const Device device = input.device();
      const DataType dtype = input.dtype();

      // pre_encode: [batch, 80, T] → [batch, T', d_model]
      StorageView x(dtype, device);
      _pre_encode(input, lengths, x, out_lengths);

      // xscaling: multiply encoder output by sqrt(d_model) before adding positional encoding.
      // This is NeMo's ConformerEncoder xscaling (xscaling=True by default), implemented in
      // PositionalEncoding.forward() as `x = x * self.xscale` where xscale = sqrt(d_model).
      // Purpose: to keep the embedding magnitudes comparable to the sinusoidal positional
      // encoding values, which have unit amplitude, after the pre_encode linear projection.
      // Reference: nemo/collections/asr/parts/submodules/multi_head_attention.py
      StorageView scale_val(std::sqrt(static_cast<float>(_d_model)));
      scale_val.move_to(Device::CPU, dtype);
      ops::Mul()(x, scale_val, x);

      // 24 ConformerBlocks
      for (const auto& layer : _layers) {
        StorageView layer_out(dtype, device);
        (*layer)(x, layer_out, &out_lengths);
        x = std::move(layer_out);
      }

      output = std::move(x);
    }

  }
}
