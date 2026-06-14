#include "substitutions/tensor_pattern/satisfies_constraint.h"
#include "substitutions/tensor_pattern/tensor_attribute_expr.h"
#include <libassert/assert.hpp>
#include <optional>
#include <stdexcept>

namespace FlexFlow {

bool parallel_tensor_satisfies_constraint(
    ParallelTensorAttrs const &attrs,
    TensorAttributeConstraint const &constraint) {
  std::optional<TensorAttributeValue> expr_val;
  try {
    expr_val = evaluate_attribute_expr(attrs, constraint.attribute_expr);
  } catch (std::runtime_error const &) {
    return false;
  }

  switch (constraint.constraint_type) {
  case ConstraintType::EQUAL:
    return expr_val.value() == constraint.attribute_value;
  case ConstraintType::DIVISIBLE_BY: {
    ASSERT(expr_val.value().has<nonnegative_int>() &&
               constraint.attribute_value.has<nonnegative_int>(),
           "DIVISIBLE_BY constraint requires nonnegative_int values");

    return expr_val.value().get<nonnegative_int>() %
               constraint.attribute_value.get<nonnegative_int>() ==
           0;
  }
  default:
    PANIC("Unknown constraint type", constraint.constraint_type);
  }
}

} // namespace FlexFlow
