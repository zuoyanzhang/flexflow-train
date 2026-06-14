#include "substitutions/tensor_pattern/eval_list_access.h"
#include "substitutions/tensor_pattern/get_attribute.h"
#include "utils/containers/at_idx.h"
#include "utils/exception.h"
#include "utils/overload.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

TensorAttributeValue
eval_list_access(ParallelTensorAttrs const &attrs,
                 TensorAttributeListIndexAccess const &acc) {
  TensorAttributeValue from_attr = get_attribute(attrs, acc.attribute_key);

  return from_attr.visit<TensorAttributeValue>(overload{
      [&](std::vector<nonnegative_int> const &v) -> TensorAttributeValue {
        return TensorAttributeValue{at_idx(v, acc.index)};
      },
      [&](std::vector<positive_int> const &v) -> TensorAttributeValue {
        return TensorAttributeValue{at_idx(v, acc.index)};
      },
      [](auto &&) -> TensorAttributeValue {
        throw mk_runtime_error("Invalid operand");
      },
  });
}

} // namespace FlexFlow
