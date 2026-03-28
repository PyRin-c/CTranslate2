#include "ctranslate2/ops/conv2d.h"

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Conv2D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output) const {
      const int batch_size = static_cast<int>(input.dim(0));
      const int in_channels = static_cast<int>(input.dim(1));
      const int in_h = static_cast<int>(input.dim(2));
      const int in_w = static_cast<int>(input.dim(3));
      const int out_channels = static_cast<int>(weight.dim(0));
      const int in_channels_per_group = static_cast<int>(weight.dim(1));
      const int kH = static_cast<int>(weight.dim(2));
      const int kW = static_cast<int>(weight.dim(3));
      const int out_h = static_cast<int>(output.dim(2));
      const int out_w = static_cast<int>(output.dim(3));
      const int pad_h = static_cast<int>(_padding_h);
      const int pad_w = static_cast<int>(_padding_w);
      const int stride_h = static_cast<int>(_stride_h);
      const int stride_w = static_cast<int>(_stride_w);
      const int groups = static_cast<int>(_groups);

      cudnnDataType_t data_type = cuda::get_cudnn_data_type(input.dtype());
      cudnnHandle_t handle = cuda::get_cudnn_handle();

      cudnnTensorDescriptor_t input_desc;
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, data_type,
                                             batch_size, in_channels, in_h, in_w));

      cudnnTensorDescriptor_t output_desc;
      CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, data_type,
                                             batch_size, out_channels, out_h, out_w));

      cudnnFilterDescriptor_t weight_desc;
      CUDNN_CHECK(cudnnCreateFilterDescriptor(&weight_desc));
      CUDNN_CHECK(cudnnSetFilter4dDescriptor(weight_desc, data_type, CUDNN_TENSOR_NCHW,
                                             out_channels, in_channels_per_group, kH, kW));

      cudnnConvolutionDescriptor_t conv_desc;
      CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
      CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                  pad_h, pad_w,
                                                  stride_h, stride_w,
                                                  /*dilation_h=*/1, /*dilation_w=*/1,
                                                  CUDNN_CROSS_CORRELATION,
                                                  CUDNN_DATA_FLOAT));

      if (data_type == CUDNN_DATA_HALF)
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
      if (groups > 1)
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, groups));

      // Use IMPLICIT_GEMM — works for all configurations without a workspace.
      cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

      size_t workspace_size = 0;
      void* workspace = nullptr;
      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                          input_desc,
                                                          weight_desc,
                                                          conv_desc,
                                                          output_desc,
                                                          algo,
                                                          &workspace_size));
      if (workspace_size > 0)
        workspace = get_allocator<Device::CUDA>().allocate(workspace_size);

      const float alpha = 1.f;
      const float beta  = 0.f;

      CUDNN_CHECK(cudnnConvolutionForward(handle,
                                          &alpha,
                                          input_desc, input.buffer(),
                                          weight_desc, weight.buffer(),
                                          conv_desc,
                                          algo,
                                          workspace, workspace_size,
                                          &beta,
                                          output_desc, output.buffer()));

      if (workspace)
        get_allocator<Device::CUDA>().free(workspace);

      // Add bias separately using cudnnAddTensor if bias is provided.
      if (bias) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, data_type,
                                               1, out_channels, 1, 1));
        CUDNN_CHECK(cudnnAddTensor(handle,
                                   &alpha,
                                   bias_desc, bias->buffer(),
                                   &alpha,
                                   output_desc, output.buffer()));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
      }

      CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CHECK(cudnnDestroyFilterDescriptor(weight_desc));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
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
