#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // 2-D convolution forward pass.
    //
    // Supports:
    //   - regular Conv2d  (groups = 1)
    //   - depthwise Conv2d (groups = in_channels, in_channels_per_group = 1)
    //   - pointwise Conv2d (kernel = 1x1)
    //
    // Input  layout : [batch, in_channels,  height,  width]  (NCHW)
    // Weight layout : [out_channels, in_channels/groups, kH, kW]
    // Output layout : [batch, out_channels, out_height, out_width]
    //
    // CPU implementation: im2col + GEMM (no DNNL dependency for the 2-D case).
    // GPU implementation: cuDNN cudnnConvolutionForward.
    class Conv2D : public Op {
    public:
      Conv2D(dim_t stride_h = 1,
             dim_t stride_w = 1,
             dim_t padding_h = 0,
             dim_t padding_w = 0,
             dim_t groups = 1);

      // With bias
      void operator()(const StorageView& input,
                      const StorageView& weight,
                      const StorageView& bias,
                      StorageView& output) const;

      // Without bias
      void operator()(const StorageView& input,
                      const StorageView& weight,
                      StorageView& output) const;

    private:
      dim_t _stride_h;
      dim_t _stride_w;
      dim_t _padding_h;
      dim_t _padding_w;
      dim_t _groups;

      void operator()(const StorageView& input,
                      const StorageView& weight,
                      const StorageView* bias,
                      StorageView& output) const;

      template <Device D, typename T>
      void compute(const StorageView& input,
                   const StorageView& weight,
                   const StorageView* bias,
                   StorageView& output) const;

      // CPU helper: im2col (NCHW input → column matrix)
      // output_col shape: [batch, groups, out_h*out_w, (in_ch/groups)*kH*kW]
      void im2col(const StorageView& input,
                  StorageView& col,
                  dim_t kH, dim_t kW,
                  dim_t out_h, dim_t out_w) const;
    };

  }
}
