#include "ctranslate2/ops/conv2d.h"

#include <cstring>

#include "ctranslate2/ops/gemm.h"
#include "cpu/parallel.h"

namespace ctranslate2 {
  namespace ops {

    // ------------------------------------------------------------------
    // im2col
    //
    // Extracts image patches into a column matrix (transposed layout for
    // compatibility with the GEMM-based convolution below).
    //
    // col shape: [batch, groups, out_h*out_w, (in_ch_per_group)*kH*kW]
    //            — each row corresponds to one output position.
    // ------------------------------------------------------------------
    void Conv2D::im2col(const StorageView& input,
                        StorageView& col,
                        dim_t kH, dim_t kW,
                        dim_t out_h, dim_t out_w) const {
      const dim_t batch           = input.dim(0);
      const dim_t in_ch           = input.dim(1);
      const dim_t in_h_dim        = input.dim(2);
      const dim_t in_w_dim        = input.dim(3);
      const dim_t in_ch_per_group = in_ch / _groups;
      const dim_t patch_size      = in_ch_per_group * kH * kW;
      const dim_t n_patches       = out_h * out_w;

      const auto* src = input.data<float>();
      auto*       dst = col.data<float>();

      // Zero-fill for zero-padding
      std::memset(dst, 0, col.size() * sizeof(float));

      // Strides in the input tensor (NCHW layout)
      const dim_t in_w_stride  = 1;
      const dim_t in_h_stride  = in_w_dim;
      const dim_t in_ch_stride = in_h_dim * in_w_dim;
      const dim_t in_n_stride  = in_ch * in_h_dim * in_w_dim;

      for (dim_t n = 0; n < batch; ++n) {
        for (dim_t g = 0; g < _groups; ++g) {
          // col row index = oh * out_w + ow
          // col col index = (c * kH + kh) * kW + kw
          for (dim_t oh = 0; oh < out_h; ++oh) {
            for (dim_t ow = 0; ow < out_w; ++ow) {
              const dim_t col_row = oh * out_w + ow;
              dim_t col_col = 0;
              for (dim_t c = 0; c < in_ch_per_group; ++c) {
                const dim_t in_c = g * in_ch_per_group + c;
                for (dim_t kh = 0; kh < kH; ++kh) {
                  for (dim_t kw = 0; kw < kW; ++kw) {
                    const dim_t ih = oh * _stride_h - _padding_h + kh;
                    const dim_t iw = ow * _stride_w - _padding_w + kw;

                    dim_t col_idx = ((n * _groups + g) * n_patches + col_row) * patch_size + col_col;

                    if (ih >= 0 && ih < in_h_dim && iw >= 0 && iw < in_w_dim) {
                      dst[col_idx] = src[n * in_n_stride
                                        + in_c * in_ch_stride
                                        + ih * in_h_stride
                                        + iw * in_w_stride];
                    }
                    ++col_col;
                  }
                }
              }
            }
          }
        }
      }
    }

    // ------------------------------------------------------------------
    // CPU compute specialisation
    //
    // GEMM layout (for each batch × group slice):
    //   weight_slice : [out_ch_per_group, patch_size]  (M × K)
    //   col_slice    : [n_patches, patch_size]ᵀ         (K × N)  → transpose_b = true
    //   output_slice : [out_ch_per_group, n_patches]    (M × N)
    // ------------------------------------------------------------------
    template <>
    void Conv2D::compute<Device::CPU, float>(const StorageView& input,
                                             const StorageView& weight,
                                             const StorageView* bias,
                                             StorageView& output) const {
      const dim_t batch           = input.dim(0);
      const dim_t in_ch           = input.dim(1);
      const dim_t out_ch          = weight.dim(0);
      const dim_t kH              = weight.dim(2);
      const dim_t kW              = weight.dim(3);
      const dim_t out_h           = output.dim(2);
      const dim_t out_w           = output.dim(3);
      const dim_t in_ch_per_group = in_ch / _groups;
      const dim_t out_ch_per_group= out_ch / _groups;
      const dim_t patch_size      = in_ch_per_group * kH * kW;
      const dim_t n_patches       = out_h * out_w;

      // col: [batch, groups, n_patches, patch_size]
      StorageView col({batch, _groups, n_patches, patch_size}, 0.f, Device::CPU);
      im2col(input, col, kH, kW, out_h, out_w);

      const Gemm gemm(/*alpha=*/1.f, /*beta=*/0.f,
                      /*trans_a=*/false, /*trans_b=*/true);

      const dim_t weight_group_stride = out_ch_per_group * in_ch_per_group * kH * kW;
      const dim_t col_group_stride    = n_patches * patch_size;
      const dim_t out_group_stride    = out_ch_per_group * n_patches;

      auto* w_ptr   = static_cast<const float*>(weight.buffer());
      auto* col_ptr = col.data<float>();
      auto* out_ptr = output.data<float>();

      cpu::parallel_for(0, batch * _groups, 1, [&](dim_t begin, dim_t end) {
        for (dim_t bg = begin; bg < end; ++bg) {
          const dim_t b = bg / _groups;
          const dim_t g = bg % _groups;

          StorageView w_view(DataType::FLOAT32, Device::CPU);
          w_view.view(
            const_cast<float*>(w_ptr + g * weight_group_stride),
            {out_ch_per_group, patch_size});

          StorageView col_view({n_patches, patch_size},
                               col_ptr + (b * _groups + g) * col_group_stride);

          StorageView out_view({out_ch_per_group, n_patches},
                               out_ptr + (b * out_ch + g * out_ch_per_group) * n_patches);

          // GEMM: weight (M×K) × col^T (K×N) → output (M×N)
          gemm(w_view, col_view, out_view);
        }
      });

      // Add bias  [out_ch] broadcasted over [batch, out_ch, out_h*out_w]
      if (bias) {
        const float* b_ptr  = bias->data<float>();
        const dim_t spatial = out_h * out_w;
        cpu::parallel_for(0, batch * out_ch, 1, [&](dim_t begin, dim_t end) {
          for (dim_t bc = begin; bc < end; ++bc) {
            const dim_t c = bc % out_ch;
            float* ptr = out_ptr + bc * spatial;
            const float bval = b_ptr[c];
            for (dim_t s = 0; s < spatial; ++s)
              ptr[s] += bval;
          }
        });
      }
    }

  }
}
