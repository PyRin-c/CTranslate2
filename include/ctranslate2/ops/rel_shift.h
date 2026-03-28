#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    // Relative-position shift operation used in NeMo Conformer attention.
    //
    // Converts the position score matrix from shape [B, H, T, 2T-1] to [B, H, T, T]
    // following NeMo multi_head_attention.py:
    //
    //   1. Pad left by 1 zero column:  [B, H, T, 2T-1] → [B, H, T, 2T]
    //   2. Reshape:                     [B, H, T, 2T]   → [B, H, 2T, T]
    //   3. Slice rows [1:]:             [B, H, 2T, T]   → [B, H, 2T-1, T]
    //   4. Slice rows [:T]:             [B, H, 2T-1, T] → [B, H, T, T]
    //
    // Note: step 2 (reshape) requires a contiguous memory copy.
    class RelShift : public Op {
    public:
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& input, StorageView& output) const;
    };

  }
}
