#include "ctranslate2/ops/rel_shift.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    void RelShift::operator()(const StorageView& input, StorageView& output) const {
      PROFILE("RelShift");

      if (input.rank() != 4)
        throw std::invalid_argument("RelShift: input must be 4-D [B, H, T, 2T-1]");

      const dim_t T = input.dim(2);
      if (input.dim(3) != 2 * T - 1)
        throw std::invalid_argument(
          "RelShift: last dim must equal 2*T-1 where T is dim(2)");

      output.resize({input.dim(0), input.dim(1), T, T});

      DEVICE_AND_FLOAT_DISPATCH("RelShift", input.device(), input.dtype(),
                                (compute<D, T>(input, output)));
    }

  }
}
