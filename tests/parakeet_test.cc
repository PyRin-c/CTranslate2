// NOTE: This file is specific to PyRin-c/CTranslate2 (feature/add_support_parakeet-ctc).
// Not present in upstream OpenNMT/CTranslate2.
//
// Unit tests for Parakeet-related Ops:
//   - Conv2D  (ops/conv2d.cc)
//   - RelShift (ops/rel_shift.cc)

#include "test_utils.h"

#include "ctranslate2/ops/conv2d.h"
#include "ctranslate2/ops/rel_shift.h"

using namespace ctranslate2;

// =============================================================================
// Conv2D Op Tests
// =============================================================================

// --- Output shape ---

TEST(Conv2DTest, OutputShapeValidPadding) {
  // Input [1, 1, 5, 5], weight [1, 1, 3, 3], stride=1, no padding
  // out = (5 - 3) / 1 + 1 = 3  →  [1, 1, 3, 3]
  StorageView input({1, 1, 5, 5}, std::vector<float>(25, 1.f));
  StorageView weight({1, 1, 3, 3}, std::vector<float>(9, 1.f));
  StorageView output;
  ops::Conv2D()(input, weight, output);
  assert_vector_eq(output.shape(), {1, 1, 3, 3});
}

TEST(Conv2DTest, OutputShapeWithPadding) {
  // Input [1, 1, 5, 5], weight [1, 1, 3, 3], stride=1, padding=1
  // out = (5 + 2*1 - 3) / 1 + 1 = 5  →  [1, 1, 5, 5]
  StorageView input({1, 1, 5, 5}, std::vector<float>(25, 1.f));
  StorageView weight({1, 1, 3, 3}, std::vector<float>(9, 1.f));
  StorageView output;
  ops::Conv2D(/*stride_h=*/1, /*stride_w=*/1, /*padding_h=*/1, /*padding_w=*/1)(
      input, weight, output);
  assert_vector_eq(output.shape(), {1, 1, 5, 5});
}

TEST(Conv2DTest, OutputShapeWithStride2) {
  // Input [1, 1, 8, 8], weight [1, 1, 3, 3], stride=2, no padding
  // out = (8 - 3) / 2 + 1 = 3  →  [1, 1, 3, 3]
  StorageView input({1, 1, 8, 8}, std::vector<float>(64, 1.f));
  StorageView weight({1, 1, 3, 3}, std::vector<float>(9, 1.f));
  StorageView output;
  ops::Conv2D(/*stride_h=*/2, /*stride_w=*/2)(input, weight, output);
  assert_vector_eq(output.shape(), {1, 1, 3, 3});
}

TEST(Conv2DTest, OutputShapeMultipleChannels) {
  // Input [2, 3, 4, 4], weight [8, 3, 2, 2], stride=1, no padding
  // out = (4 - 2) / 1 + 1 = 3  →  [2, 8, 3, 3]
  StorageView input({2, 3, 4, 4}, std::vector<float>(2*3*4*4, 1.f));
  StorageView weight({8, 3, 2, 2}, std::vector<float>(8*3*2*2, 1.f));
  StorageView output;
  ops::Conv2D()(input, weight, output);
  assert_vector_eq(output.shape(), {2, 8, 3, 3});
}

// --- Correctness ---

TEST(Conv2DTest, SumFilter3x3) {
  // 1-channel 3x3 all-ones kernel with no padding on a 3x3 input.
  // Output is a single value equal to the sum of all 9 input elements.
  StorageView input({1, 1, 3, 3},
      std::vector<float>{1, 2, 3,
                         4, 5, 6,
                         7, 8, 9});
  StorageView weight({1, 1, 3, 3}, std::vector<float>(9, 1.f));
  StorageView output;
  ops::Conv2D()(input, weight, output);
  StorageView expected({1, 1, 1, 1}, std::vector<float>{45.f});
  expect_storage_eq(output, expected);
}

TEST(Conv2DTest, Identity1x1Kernel) {
  // 1x1 kernel with weight=1 and no bias: output must equal input.
  StorageView input({1, 1, 3, 3},
      std::vector<float>{1, 2, 3,
                         4, 5, 6,
                         7, 8, 9});
  StorageView weight({1, 1, 1, 1}, std::vector<float>{1.f});
  StorageView output;
  ops::Conv2D()(input, weight, output);
  expect_storage_eq(output, input);
}

TEST(Conv2DTest, BiasAdded) {
  // 1x1 kernel, weight=1, bias=5: output = input + 5.
  StorageView input({1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4});
  StorageView weight({1, 1, 1, 1}, std::vector<float>{1.f});
  StorageView bias({1}, std::vector<float>{5.f});
  StorageView output;
  ops::Conv2D()(input, weight, bias, output);
  StorageView expected({1, 1, 2, 2}, std::vector<float>{6, 7, 8, 9});
  expect_storage_eq(output, expected);
}

TEST(Conv2DTest, MultiChannelBias) {
  // 2 output channels, each with its own bias value.
  // Input [1, 1, 2, 2] = all zeros; weight [2, 1, 1, 1] = all ones.
  // bias = [3, 7], so output channel 0 = 3, channel 1 = 7.
  StorageView input({1, 1, 2, 2}, std::vector<float>(4, 0.f));
  StorageView weight({2, 1, 1, 1}, std::vector<float>{1.f, 1.f});
  StorageView bias({2}, std::vector<float>{3.f, 7.f});
  StorageView output;
  ops::Conv2D()(input, weight, bias, output);
  StorageView expected({1, 2, 2, 2},
      std::vector<float>{3, 3, 3, 3,   // channel 0
                         7, 7, 7, 7}); // channel 1
  expect_storage_eq(output, expected);
}

TEST(Conv2DTest, DepthwiseConv) {
  // Depthwise convolution: groups = in_channels = 2.
  // Each channel has its own 1x1 kernel.
  // Channel 0: scale by 2; Channel 1: scale by 3.
  StorageView input({1, 2, 2, 2},
      std::vector<float>{1, 1, 1, 1,   // channel 0
                         2, 2, 2, 2}); // channel 1
  StorageView weight({2, 1, 1, 1}, std::vector<float>{2.f, 3.f});
  StorageView output;
  ops::Conv2D(/*stride_h=*/1, /*stride_w=*/1,
              /*padding_h=*/0, /*padding_w=*/0,
              /*groups=*/2)(input, weight, output);
  StorageView expected({1, 2, 2, 2},
      std::vector<float>{2, 2, 2, 2,   // ch0 * 2
                         6, 6, 6, 6}); // ch1 * 3
  expect_storage_eq(output, expected);
}

TEST(Conv2DTest, BatchDimension) {
  // Two identical batch items must produce identical outputs.
  const std::vector<float> vals(1*3*3, 1.f);
  StorageView input({2, 1, 3, 3}, std::vector<float>(2*1*3*3, 1.f));
  StorageView weight({1, 1, 3, 3}, std::vector<float>(9, 1.f));
  StorageView output;
  ops::Conv2D()(input, weight, output);
  ASSERT_EQ(output.dim(0), 2);
  const float* out = output.data<float>();
  EXPECT_FLOAT_EQ(out[0], out[1]); // both batch items yield the same scalar
}

// --- Invalid dtype ---

TEST(Conv2DTest, InvalidDtypeInt8) {
  StorageView input({1, 1, 2, 2}, DataType::INT8);
  StorageView weight({1, 1, 1, 1}, DataType::INT8);
  StorageView output;
  EXPECT_THROW(ops::Conv2D()(input, weight, output), std::invalid_argument);
}

TEST(Conv2DTest, InvalidDtypeInt16) {
  StorageView input({1, 1, 2, 2}, DataType::INT16);
  StorageView weight({1, 1, 1, 1}, DataType::INT16);
  StorageView output;
  EXPECT_THROW(ops::Conv2D()(input, weight, output), std::invalid_argument);
}

// =============================================================================
// RelShift Op Tests
// =============================================================================

// --- Output shape ---

TEST(RelShiftTest, OutputShapeT3) {
  // T=3  →  input dim(3) = 2*3-1 = 5  →  output [1, 1, 3, 3]
  StorageView input({1, 1, 3, 5}, std::vector<float>(15, 0.f));
  StorageView output;
  ops::RelShift()(input, output);
  assert_vector_eq(output.shape(), {1, 1, 3, 3});
}

TEST(RelShiftTest, OutputShapeMultiHeadBatch) {
  // B=2, H=4, T=5  →  input [2, 4, 5, 9]  →  output [2, 4, 5, 5]
  StorageView input({2, 4, 5, 9}, std::vector<float>(2*4*5*9, 0.f));
  StorageView output;
  ops::RelShift()(input, output);
  assert_vector_eq(output.shape(), {2, 4, 5, 5});
}

// --- Correctness ---

TEST(RelShiftTest, KnownValuesT2) {
  // B=1, H=1, T=2 — hand-traced through the CPU algorithm:
  //
  // Input rows (T=2 queries, 2T-1=3 relative keys):
  //   row 0 : [1, 2, 3]
  //   row 1 : [4, 5, 6]
  //
  // Step 1+2: left-pad, build pad_buf [2T=4, T=2]:
  //   [0, 1, 2, 3, 0, 4, 5, 6]
  //
  // remaining = pad_buf + T = &pad_buf[2] = [2, 3, 0, 4, 5, 6]
  // stride = 2T-1 = 3
  //   out row 0: remaining[0..1]          = [2, 3]
  //   out row 1: remaining[stride..stride+1] = [4, 5]  (remaining[3..4])
  //
  // Expected output [1, 1, 2, 2]: [[2, 3], [4, 5]]
  StorageView input({1, 1, 2, 3},
      std::vector<float>{1, 2, 3,
                         4, 5, 6});
  StorageView output;
  ops::RelShift()(input, output);
  StorageView expected({1, 1, 2, 2}, std::vector<float>{2, 3, 4, 5});
  expect_storage_eq(output, expected);
}

TEST(RelShiftTest, KnownValuesT1) {
  // T=1  →  input [1, 1, 1, 1] (2*1-1=1), output [1, 1, 1, 1]
  // The single element should pass through unchanged.
  StorageView input({1, 1, 1, 1}, std::vector<float>{7.f});
  StorageView output;
  ops::RelShift()(input, output);
  StorageView expected({1, 1, 1, 1}, std::vector<float>{7.f});
  expect_storage_eq(output, expected);
}

TEST(RelShiftTest, ZeroInputYieldsZeroOutput) {
  StorageView input({1, 2, 4, 7}, std::vector<float>(1*2*4*7, 0.f));
  StorageView output;
  ops::RelShift()(input, output);
  assert_vector_eq(output.shape(), {1, 2, 4, 4});
  const float* out = output.data<float>();
  for (size_t i = 0; i < output.size(); ++i)
    EXPECT_FLOAT_EQ(out[i], 0.f) << "Non-zero at index " << i;
}

// --- Input validation ---

TEST(RelShiftTest, InvalidRank3D) {
  StorageView input({1, 3, 5}, std::vector<float>(15, 0.f));
  StorageView output;
  EXPECT_THROW(ops::RelShift()(input, output), std::invalid_argument);
}

TEST(RelShiftTest, InvalidLastDimNotTwoTMinus1) {
  // T=3 but last dim=4 (should be 5=2*3-1)
  StorageView input({1, 1, 3, 4}, std::vector<float>(12, 0.f));
  StorageView output;
  EXPECT_THROW(ops::RelShift()(input, output), std::invalid_argument);
}

TEST(RelShiftTest, InvalidLastDimT1) {
  // T=1 but last dim=2 (should be 1=2*1-1)
  StorageView input({1, 1, 1, 2}, std::vector<float>(2, 0.f));
  StorageView output;
  EXPECT_THROW(ops::RelShift()(input, output), std::invalid_argument);
}
