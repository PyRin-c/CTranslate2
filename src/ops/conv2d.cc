#include "ctranslate2/ops/conv2d.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    Conv2D::Conv2D(dim_t stride_h,
                   dim_t stride_w,
                   dim_t padding_h,
                   dim_t padding_w,
                   dim_t groups)
      : _stride_h(stride_h)
      , _stride_w(stride_w)
      , _padding_h(padding_h)
      , _padding_w(padding_w)
      , _groups(groups)
    {
    }

    void Conv2D::operator()(const StorageView& input,
                            const StorageView& weight,
                            const StorageView& bias,
                            StorageView& output) const {
      operator()(input, weight, &bias, output);
    }

    void Conv2D::operator()(const StorageView& input,
                            const StorageView& weight,
                            StorageView& output) const {
      operator()(input, weight, nullptr, output);
    }

    void Conv2D::operator()(const StorageView& input,
                            const StorageView& weight,
                            const StorageView* bias,
                            StorageView& output) const {
      PROFILE("Conv2D");

      // Input:  [batch, in_ch, H, W]
      // Weight: [out_ch, in_ch/groups, kH, kW]
      const dim_t batch    = input.dim(0);
      const dim_t in_h     = input.dim(2);
      const dim_t in_w     = input.dim(3);
      const dim_t out_ch   = weight.dim(0);
      const dim_t kH       = weight.dim(2);
      const dim_t kW       = weight.dim(3);

      // Output spatial dimensions
      // out = floor((in + 2*pad - kernel) / stride) + 1
      const dim_t out_h = (in_h + 2 * _padding_h - kH) / _stride_h + 1;
      const dim_t out_w = (in_w + 2 * _padding_w - kW) / _stride_w + 1;

      output.resize({batch, out_ch, out_h, out_w});

      DEVICE_AND_FLOAT_DISPATCH("Conv2D", input.device(), input.dtype(),
                                (compute<D, T>(input, weight, bias, output)));
    }

  }
}
