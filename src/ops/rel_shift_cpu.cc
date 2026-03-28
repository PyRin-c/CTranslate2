#include "ctranslate2/ops/rel_shift.h"

#include <cstring>

namespace ctranslate2 {
  namespace ops {

    // ------------------------------------------------------------------
    // rel_shift — CPU float32 implementation
    //
    // Input : x  [B, H, T, 2T-1]
    // Output: y  [B, H, T, T]
    //
    // Algorithm (NeMo multi_head_attention.py, pad direction = LEFT):
    //
    //   Step 1: pad x left by 1 column of zeros → [B, H, T, 2T]
    //   Step 2: reshape (copy) → [B, H, 2T, T]
    //   Step 3: drop first row  → view [B, H, 2T-1, T]
    //   Step 4: take first T rows → [B, H, T, T]  (output)
    // ------------------------------------------------------------------
    template <>
    void RelShift::compute<Device::CPU, float>(
        const StorageView& input, StorageView& output) const {
      const dim_t B             = input.dim(0);
      const dim_t H             = input.dim(1);
      const dim_t T             = input.dim(2);
      const dim_t two_T_minus_1 = input.dim(3);
      const dim_t two_T         = two_T_minus_1 + 1;

      const float* src = input.data<float>();
      float*       dst = output.data<float>();

      // Allocate a padded-then-reshaped intermediate buffer: [B, H, 2T, T]
      std::vector<float> pad_buf(B * H * two_T * T, 0.f);

      for (dim_t b = 0; b < B; ++b) {
        for (dim_t h = 0; h < H; ++h) {
          const float* src_bh = src + (b * H + h) * T * two_T_minus_1;
          float*       dst_bh = pad_buf.data() + (b * H + h) * two_T * T;

          // Step 1+2: left-pad → reshape into [2T, T] buffer
          for (dim_t t = 0; t < T; ++t) {
            const float* src_row = src_bh + t * two_T_minus_1;
            for (dim_t r = 0; r < two_T_minus_1; ++r) {
              dst_bh[t * two_T + (r + 1)] = src_row[r];
            }
          }

          // Step 3+4: skip row 0, copy first T rows of the [2T-1, T] view
          float*       out_bh    = dst + (b * H + h) * T * T;
          const float* remaining = dst_bh + T;       // skip row 0 (T elements)
          const dim_t  stride    = two_T_minus_1;    // row stride in [T, 2T-1] view
          for (dim_t row = 0; row < T; ++row) {
            std::memcpy(out_bh + row * T,
                        remaining + row * stride,
                        T * sizeof(float));
          }
        }
      }
    }

  }
}
