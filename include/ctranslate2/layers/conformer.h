#pragma once

#include <memory>
#include <vector>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/ops/conv2d.h"
#include "ctranslate2/ops/rel_shift.h"

namespace ctranslate2 {
  namespace layers {

    // -----------------------------------------------------------------------
    // ConvModule
    //
    // Execution order (v3, based on NeMo source):
    //   1. LayerNorm (norm)
    //   2. pointwise_conv1 → [batch, T, 2*d_model]
    //   3. GLU: split → A, B; gate = A * sigmoid(B)  → [batch, T, d_model]
    //      (gate uses sigmoid per NeMo nn.functional.glu, NOT SiLU/Swish)
    //   4. transpose → [batch, d_model, T]
    //   5. depthwise_conv (BN already fused into weights)  → [batch, d_model, T]
    //   6. Swish/SiLU  (post-depthwise activation, distinct from the GLU gate above)
    //   7. transpose back → [batch, T, d_model]
    //   8. pointwise_conv2 → [batch, T, d_model]
    //   9. residual add (output = input + result)
    // -----------------------------------------------------------------------
    class ConvModule {
    public:
      ConvModule(const models::Model& model, const std::string& scope);

      // input/output: [batch, T, d_model]
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      LayerNorm _norm;
      Dense     _pointwise_conv1;  // [2*d_model, d_model]
      Conv1D    _depthwise_conv;   // [d_model, 1, kernel]  groups=d_model, BN-fused
      Dense     _pointwise_conv2;  // [d_model, d_model]
    };


    // -----------------------------------------------------------------------
    // RelativePositionMultiHeadAttention
    //
    // NeMo-style relative-position MHA (Transformer-XL variant).
    //   - separate linear_pos projection (no bias)
    //   - pos_bias_u / pos_bias_v learnable biases
    //   - rel_shift applied to position scores
    // -----------------------------------------------------------------------
    class RelativePositionMultiHeadAttention {
    public:
      RelativePositionMultiHeadAttention(const models::Model& model,
                                         const std::string& scope,
                                         dim_t num_heads);

      // input:   [batch, T, d_model]
      // lengths: optional [batch] sequence lengths for masking
      // output:  [batch, T, d_model]
      void operator()(const StorageView& input,
                      StorageView& output,
                      const StorageView* lengths = nullptr) const;

    private:
      const dim_t _num_heads;
      const dim_t _head_dim;

      Dense _linear_q;
      Dense _linear_k;
      Dense _linear_v;
      Dense _linear_out;
      Dense _linear_pos;       // no bias

      const StorageView& _pos_bias_u;   // [num_heads, head_dim]
      const StorageView& _pos_bias_v;   // [num_heads, head_dim]

      ops::RelShift _rel_shift;

      // Compute sinusoidal position encoding for length seq_len.
      // Returns [2T-1, d_model] on the same device/dtype as the model.
      void compute_pos_encoding(dim_t seq_len,
                                DataType dtype,
                                Device device,
                                StorageView& pos_enc) const;
    };


    // -----------------------------------------------------------------------
    // ConformerBlock
    //
    // Macaron-style block:
    //   FFN1 (×0.5) → RelPos-MHA → ConvModule → FFN2 (×0.5) → norm_out
    // -----------------------------------------------------------------------
    class ConformerBlock {
    public:
      ConformerBlock(const models::Model& model,
                     const std::string& scope,
                     dim_t num_heads = 8);

      // input/output: [batch, T, d_model]
      void operator()(const StorageView& input,
                      StorageView& output,
                      const StorageView* lengths = nullptr) const;

    private:
      // FFN1
      LayerNorm _norm_ff1;
      Dense     _ff1_linear1;   // [4*d_model, d_model]  + SiLU
      Dense     _ff1_linear2;   // [d_model, 4*d_model]

      // MHA
      LayerNorm _norm_mha;
      RelativePositionMultiHeadAttention _mha;

      // ConvModule (norm is inside ConvModule itself)
      ConvModule _conv;

      // FFN2
      LayerNorm _norm_ff2;
      Dense     _ff2_linear1;
      Dense     _ff2_linear2;

      // Output norm
      LayerNorm _norm_out;
    };


    // -----------------------------------------------------------------------
    // ConformerPreEncoder
    //
    // 2-D conv subsampling (stacked DepthwiseSeparable Conv2d blocks).
    // Input : [batch, 80, T]  (mel spectrogram, no channel dim)
    // Output: [batch, T/8, d_model]
    // -----------------------------------------------------------------------
    class ConformerPreEncoder {
    public:
      ConformerPreEncoder(const models::Model& model, const std::string& scope);

      // input:   [batch, 80, T]
      // lengths: [batch] input frame lengths
      // output:  [batch, T', d_model]   T' ≈ T/8
      // out_lengths: [batch] output frame lengths
      void operator()(const StorageView& input,
                      const StorageView& lengths,
                      StorageView& output,
                      StorageView& out_lengths) const;

    private:
      // Convolution weights/biases loaded from model variables
      const StorageView& _conv0_w;   // [256, 1,   3, 3]
      const StorageView& _conv0_b;   // [256]
      const StorageView& _conv2_w;   // [256, 1,   3, 3]  depthwise
      const StorageView& _conv2_b;
      const StorageView& _conv3_w;   // [256, 256, 1, 1]  pointwise
      const StorageView& _conv3_b;
      const StorageView& _conv5_w;   // [256, 1,   3, 3]  depthwise
      const StorageView& _conv5_b;
      const StorageView& _conv6_w;   // [256, 256, 1, 1]  pointwise
      const StorageView& _conv6_b;
      Dense _out;                     // Linear(2560 → 1024)

      // Op instances (configured per conv layer)
      static constexpr dim_t STRIDE   = 2;
      static constexpr dim_t PADDING  = 1;
      static constexpr dim_t GROUPS_DW= 256;

      // Compute output length for one subsampling stage
      static dim_t subsample_length(dim_t in_len, dim_t kernel, dim_t stride, dim_t pad) {
        return (in_len + 2 * pad - kernel) / stride + 1;
      }
    };


    // -----------------------------------------------------------------------
    // ConformerEncoder
    //
    // Full encoder: ConformerPreEncoder + N ConformerBlocks.
    // -----------------------------------------------------------------------
    class ConformerEncoder {
    public:
      ConformerEncoder(const models::Model& model,
                       const std::string& scope,
                       dim_t num_heads = 8);

      // input:      [batch, 80, T]
      // lengths:    [batch]
      // output:     [batch, T', d_model]
      // out_lengths:[batch]
      void operator()(const StorageView& input,
                      const StorageView& lengths,
                      StorageView& output,
                      StorageView& out_lengths) const;

    private:
      ConformerPreEncoder _pre_encode;
      std::vector<std::unique_ptr<ConformerBlock>> _layers;
      const dim_t _d_model;
    };

  }
}
