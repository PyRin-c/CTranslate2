#include "ctranslate2/ops/rel_shift.h"

#include "cuda/helpers.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // ------------------------------------------------------------------
    // rel_shift CUDA kernel
    //
    // Each thread computes one output element y[b, h, i, j].
    //
    // Index mapping (derived from the NeMo left-pad + reshape algorithm):
    //
    //   Given output y[i, j] (ignoring b, h), the corresponding source
    //   element in the padded [T, 2T] → reshaped [2T, T] buffer is:
    //
    //     flat     = (i + 1) * T + j        (position after skipping row 0)
    //     row_pad  = flat / (2*T)
    //     col_pad  = flat % (2*T)
    //
    //   If col_pad == 0  → padding zero (no source element)
    //   Otherwise        → input[b, h, row_pad, col_pad - 1]
    //
    // Note: template parameter is named DataT (not T) to avoid conflict
    // with the integer parameter T (sequence length).
    // ------------------------------------------------------------------
    template <typename DataT>
    __global__ void rel_shift_kernel(
        const DataT* __restrict__ input,   // [B, H, T, 2T-1]
        DataT*       __restrict__ output,  // [B, H, T, T]
        int B, int H, int seq_len)
    {
      const int idx   = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
      const int total = B * H * seq_len * seq_len;
      if (idx >= total)
        return;

      const int j      = idx % seq_len;
      const int i      = (idx / seq_len) % seq_len;
      const int h      = (idx / (seq_len * seq_len)) % H;
      const int b      = idx / (H * seq_len * seq_len);
      const int two_T  = 2 * seq_len;

      const int flat    = (i + 1) * seq_len + j;
      const int row_pad = flat / two_T;
      const int col_pad = flat % two_T;

      DataT val = static_cast<DataT>(0.f);
      if (col_pad != 0)
        val = input[((b * H + h) * seq_len + row_pad) * (two_T - 1) + (col_pad - 1)];

      output[idx] = val;
    }


    template <Device D, typename T>
    void RelShift::compute(const StorageView& input, StorageView& output) const {
      using DevT = cuda::device_type<T>;

      const int B        = static_cast<int>(input.dim(0));
      const int H        = static_cast<int>(input.dim(1));
      const int seq_len  = static_cast<int>(input.dim(2));

      const int total   = B * H * seq_len * seq_len;
      const int threads = 256;
      const int blocks  = (total + threads - 1) / threads;

      rel_shift_kernel<<<blocks, threads, 0, cuda::get_cuda_stream()>>>(
          cuda::device_cast(input.data<T>()),
          cuda::device_cast(output.data<T>()),
          B, H, seq_len);
    }


#define DECLARE_IMPL(T)                                                 \
    template void RelShift::compute<Device::CUDA, T>(                   \
        const StorageView&, StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }
}
