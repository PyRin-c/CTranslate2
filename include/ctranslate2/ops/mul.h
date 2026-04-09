#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Mul : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const {
        c.resize_as(a);
        if (b.is_scalar()) {
          // When the scalar is stored as float32 (e.g. StorageView(1.5f)) but the
          // compute type is float16, avoid the ASSERT_DTYPE failure by reading the
          // scalar as float and casting to T.
          T scalar_val;
          if (b.dtype() == DataType::FLOAT32)
            scalar_val = static_cast<T>(*b.data<float>());
          else
            scalar_val = b.data<T>()[0];
          primitives<D>::mul(scalar_val, a.data<T>(), c.data<T>(), c.size());
        } else {
          primitives<D>::mul(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
        }
      }
    };

  }
}
