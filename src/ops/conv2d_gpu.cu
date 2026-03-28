#include "ctranslate2/ops/conv2d.h"
#include "ctranslate2/ops/gemm.h"

#include "cuda/helpers.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // im2col CUDA kernel for 2D convolution.
    //
    // Fills a column buffer of shape [batch*groups, out_h*out_w, patch_size]
    // where patch_size = in_ch_per_group * kH * kW.
    //
    // Grid:  (ceil(patch_size/threads), out_h*out_w, batch*groups)
    // Block: (threads,)
    template<typename T>
    __global__ void im2col_2d_kernel(
        const T* __restrict__ input,    // [batch, in_ch, in_h, in_w]
        T* __restrict__ col,            // [batch*groups, out_h*out_w, patch_size]
        const int in_channels,
        const int in_h,
        const int in_w,
        const int groups,
        const int kH,
        const int kW,
        const int out_h,
        const int out_w,
        const int stride_h,
        const int stride_w,
        const int pad_h,
        const int pad_w,
        const int patch_size) {         // = in_ch_per_group * kH * kW

      const int ki      = threadIdx.x + blockIdx.x * blockDim.x;  // [0, patch_size)
      const int out_pos = blockIdx.y;                               // [0, out_h*out_w)
      const int bg      = blockIdx.z;                               // [0, batch*groups)

      if (ki >= patch_size) return;

      const int in_ch_per_group = in_channels / groups;
      const int batch_idx = bg / groups;
      const int group_idx = bg % groups;

      // Decompose ki -> (c_in_group, kh_idx, kw_idx)
      const int kw_idx     = ki % kW;
      const int kh_idx     = (ki / kW) % kH;
      const int c_in_group = ki / (kW * kH);

      // Decompose out_pos -> (oh, ow)
      const int oh = out_pos / out_w;
      const int ow = out_pos % out_w;

      // Corresponding input coordinates
      const int ih = oh * stride_h - pad_h + kh_idx;
      const int iw = ow * stride_w - pad_w + kw_idx;
      const int c_in = group_idx * in_ch_per_group + c_in_group;

      T val = T(0);
      if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
        val = input[(batch_idx * in_channels + c_in) * in_h * in_w + ih * in_w + iw];

      col[(bg * out_h * out_w + out_pos) * patch_size + ki] = val;
    }


    // CUDA implementation: im2col + cuBLAS GEMM (no cuDNN required).
    // Follows the same pattern as conv1d_gpu.cu.
    template <Device D, typename T>
    void Conv2D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output) const {

      using DevT = cuda::device_type<T>;

      const dim_t batch           = input.dim(0);
      const dim_t in_channels     = input.dim(1);
      const dim_t in_h            = input.dim(2);
      const dim_t in_w            = input.dim(3);
      const dim_t out_channels    = weight.dim(0);
      const dim_t kH              = weight.dim(2);
      const dim_t kW              = weight.dim(3);
      const dim_t out_h           = output.dim(2);
      const dim_t out_w           = output.dim(3);
      const dim_t in_ch_per_group = in_channels / _groups;
      const dim_t out_ch_per_group= out_channels / _groups;
      const dim_t patch_size      = in_ch_per_group * kH * kW;
      const dim_t n_patches       = out_h * out_w;

      // im2col buffer: [batch*groups, n_patches, patch_size]
      StorageView col({batch * _groups, n_patches, patch_size},
                      DataTypeToEnum<T>::value, D);

      {
        const int threads = 256;
        const dim3 grid(static_cast<unsigned>((patch_size + threads - 1) / threads),
                        static_cast<unsigned>(n_patches),
                        static_cast<unsigned>(batch * _groups));
        im2col_2d_kernel<<<grid, threads, 0, cuda::get_cuda_stream()>>>(
            cuda::device_cast(input.data<T>()),
            cuda::device_cast(col.data<T>()),
            static_cast<int>(in_channels),
            static_cast<int>(in_h),
            static_cast<int>(in_w),
            static_cast<int>(_groups),
            static_cast<int>(kH),
            static_cast<int>(kW),
            static_cast<int>(out_h),
            static_cast<int>(out_w),
            static_cast<int>(_stride_h),
            static_cast<int>(_stride_w),
            static_cast<int>(_padding_h),
            static_cast<int>(_padding_w),
            static_cast<int>(patch_size));
      }

      // GEMM per group (batch-strided), matching conv1d_gpu.cu pattern:
      //   A = weight_g  [out_ch_per_group, patch_size]  stride=0 (shared across batches)
      //   B = col_g     [n_patches, patch_size]          stride=groups*stridep  (trans_b=true)
      //   C = output_g  [out_ch_per_group, n_patches]   stride=groups*strideo
      //
      // Output layout: [batch, out_channels, out_h, out_w]
      //   batch b, group g starts at: b*out_channels*n_patches + g*out_ch_per_group*n_patches
      const dim_t stridew = out_ch_per_group * patch_size;
      const dim_t stridep = n_patches * patch_size;
      const dim_t strideo = out_ch_per_group * n_patches;

      const T* w_ptr = weight.data<T>();
      const T* p_ptr = col.data<T>();
      T*       o_ptr = output.data<T>();

      for (dim_t g = 0; g < _groups; ++g) {
        const T* w_g = w_ptr + g * stridew;
        const T* p_g = p_ptr + g * stridep;
        T*       o_g = o_ptr + g * strideo;

        primitives<Device::CUDA>::gemm_batch_strided(
            /*trans_a=*/false, /*trans_b=*/true,
            out_ch_per_group, n_patches, patch_size,
            /*alpha=*/1.0f,
            w_g, patch_size, /*stride_a=*/0,
            p_g, patch_size, /*stride_b=*/_groups * stridep,
            /*beta=*/0.0f,
            o_g, n_patches, /*stride_c=*/_groups * strideo,
            batch);
      }

      // Add bias [out_channels] broadcast over spatial dims.
      // Temporarily reshape to 3D [batch, out_channels, n_patches] so that
      // apply_bias_and_activation with axis=-2 broadcasts correctly (same as conv1d_gpu.cu).
      if (bias) {
        output.reshape({batch, out_channels, n_patches});
        apply_bias_and_activation(output, bias,
                                  /*activation=*/nullptr,
                                  /*residual=*/nullptr,
                                  /*axis=*/-2);
        output.reshape({batch, out_channels, out_h, out_w});
      }
    }


#define DECLARE_IMPL(T)                                                   \
    template void Conv2D::compute<Device::CUDA, T>(                       \
        const StorageView&, const StorageView&,                           \
        const StorageView*, StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
