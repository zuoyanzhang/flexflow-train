#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "op-attrs/ff_ordered/get_idxs.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/operator_task_space.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/all_of.h"
#include "utils/containers/any_of.h"
#include "utils/containers/contains.h"
#include "utils/containers/extend.h"
#include "utils/containers/filter.h"
#include "utils/containers/product.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/vector_of.h"
#include "utils/exception.h"
#include "utils/expected.h"
#include "utils/fmt/set.h"
#include "utils/orthotope/dim_projection.h"
#include "utils/orthotope/down_projection.h"

namespace FlexFlow {

std::unordered_map<TensorSlotName, IncomingTensorRole>
    get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &attrs) {
  std::unordered_map<TensorSlotName, IncomingTensorRole> result = {
      {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
  };

  if (attrs.elementwise_affine) {
    result[TensorSlotName::GAMMA] = IncomingTensorRole::WEIGHT;
    result[TensorSlotName::BETA] = IncomingTensorRole::WEIGHT;
  }

  return result;
}

static std::optional<std::string>
    check_input_shape(LayerNormAttrs const &attrs,
                      TensorShape const &input_shape) {
  if (any_of(attrs.axes, [&](ff_dim_t axis) {
        return axis.value >= get_num_dims(input_shape.dims);
      })) {
    return fmt::format(
        "LayerNorm axes {} out-of-bounds for input tensor shape {}",
        attrs.axes,
        input_shape);
  }

  return std::nullopt;
}

tl::expected<TensorShape, std::string>
    get_output_shape(LayerNormAttrs const &attrs,
                     TensorShape const &input_shape) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, input_shape);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  return input_shape;
}

tl::expected<TensorShape, std::string>
    get_gamma_weights_shape(LayerNormAttrs const &attrs,
                            TensorShape const &input_shape) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, input_shape);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No gamma weights exist for attrs.elementwise_affine = false");
  }

  std::vector<ff_dim_t> layer_norm_dim_idxs = filter(
      vector_of(get_idxs(input_shape.dims.ff_ordered)),
      [&](ff_dim_t const &dim_idx) { return contains(attrs.axes, dim_idx); });
  std::vector<positive_int> raw_weight_dims =
      transform(layer_norm_dim_idxs, [&](ff_dim_t const &dim_idx) {
        return dim_at_idx(input_shape.dims,
                          relative_ff_dim_t_from_ff_dim_t(dim_idx));
      });

  return TensorShape{
      TensorDims{ff_ordered_of(raw_weight_dims)},
      DataType::FLOAT,
  };
}

tl::expected<TensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &attrs,
                           TensorShape const &input_shape) {
  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No beta weights exist for attrs.elementwise_affine = false");
  }

  return get_gamma_weights_shape(attrs, input_shape);
}

tl::expected<std::unordered_map<TensorSlotName, TensorShape>, std::string>
    get_weight_shapes(LayerNormAttrs const &attrs,
                      TensorShape const &input_shape) {

  TensorShape gamma_shape =
      PROPAGATE_ERR(get_gamma_weights_shape(attrs, input_shape));
  TensorShape beta_shape =
      PROPAGATE_ERR(get_beta_weights_shape(attrs, input_shape));

  return std::unordered_map<TensorSlotName, TensorShape>{
      {
          TensorSlotName::GAMMA,
          gamma_shape,
      },
      {
          TensorSlotName::BETA,
          beta_shape,
      },
  };
}

static std::optional<std::string>
    check_input_degrees(LayerNormAttrs const &attrs,
                        ParallelTensorDimDegrees const &input_degrees) {
  if (any_of(attrs.axes, [&](ff_dim_t axis) {
        return axis.value.unwrap_nonnegative() >=
               input_degrees.shard_degrees.size();
      })) {
    return fmt::format(
        "LayerNorm axes {} out-of-bounds for input tensor degrees {}",
        attrs.axes,
        input_degrees);
  }

  if (input_degrees.sum_degree != SumDegree{1_p}) {
    return fmt::format("Expected sum degree 1, but receieved sum degree {}",
                       input_degrees.sum_degree);
  }

  if (input_degrees.discard_copy_degree != DiscardCopyDegree{1_p}) {
    return fmt::format(
        "Expected discard copy degree 1, but receieved discard copy degree {}",
        input_degrees.discard_copy_degree);
  }

  if (!all_of(attrs.axes, [&](ff_dim_t axis) {
        return input_degrees.shard_degrees.at(axis) == 1_p;
      })) {
    return fmt::format("Expected parallel degree of all dimensions in "
                       "LayerNorm axes {} to be 1, but received input degrees "
                       "{}",
                       attrs.axes,
                       input_degrees);
  }

  return std::nullopt;
}

tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_degrees(attrs, input_degrees);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  return input_degrees;
}

tl::expected<ParallelTensorDimDegrees, std::string>
    get_gamma_weights_parallel_dim_degrees(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  {
    std::optional<std::string> maybe_err_msg =
        check_input_degrees(attrs, input_degrees);
    if (maybe_err_msg.has_value()) {
      return tl::unexpected(maybe_err_msg.value());
    }
  }

  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No gamma weights exist for attrs.elementwise_affine = false");
  }

  std::vector<ff_dim_t> layer_norm_dim_idxs =
      filter(vector_of(get_idxs(input_degrees.shard_degrees)),
             [&](ff_dim_t const &dim_idx) {
               return contains(attrs.axes, dim_idx);
             });
  std::vector<positive_int> raw_weight_degrees =
      transform(layer_norm_dim_idxs, [&](ff_dim_t const &dim_idx) {
        return input_degrees.shard_degrees.at(dim_idx);
      });

  std::vector<positive_int> non_layer_norm_degrees =
      transform(filter(vector_of(get_idxs(input_degrees.shard_degrees)),
                       [&](ff_dim_t const &dim_idx) {
                         return !contains(attrs.axes, dim_idx);
                       }),
                [&](ff_dim_t const &dim_idx) {
                  return input_degrees.shard_degrees.at(dim_idx);
                });

  return ParallelTensorDimDegrees{
      SumDegree{1_p},
      DiscardCopyDegree{product(non_layer_norm_degrees)},
      ff_ordered_of(raw_weight_degrees),
  };
}

tl::expected<ParallelTensorDimDegrees, std::string>
    get_beta_weights_parallel_dim_degrees(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {
  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No beta weights exist for attrs.elementwise_affine = false");
  }

  return get_gamma_weights_parallel_dim_degrees(attrs, input_degrees);
}

tl::expected<std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>,
             std::string>
    get_weight_parallel_dim_degrees(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees) {

  ParallelTensorDimDegrees gamma_degrees = PROPAGATE_ERR(
      get_gamma_weights_parallel_dim_degrees(attrs, input_degrees));
  ParallelTensorDimDegrees beta_degrees = PROPAGATE_ERR(
      get_beta_weights_parallel_dim_degrees(attrs, input_degrees));

  return std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>{
      {
          TensorSlotName::GAMMA,
          gamma_degrees,
      },
      {
          TensorSlotName::BETA,
          beta_degrees,
      },
  };
}

static std::optional<std::string>
    check_input_shape(LayerNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {
  {
    TensorShape reduced_shape = get_reduced_shape(input_shape);
    std::optional<std::string> maybe_err_msg =
        check_input_shape(attrs, reduced_shape);
    if (maybe_err_msg.has_value()) {
      return maybe_err_msg;
    }
  }

  {
    std::optional<std::string> maybe_err_msg =
        check_input_degrees(attrs, get_parallel_degrees(input_shape));
    if (maybe_err_msg.has_value()) {
      return maybe_err_msg;
    }
  }

  return std::nullopt;
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LayerNormAttrs const &attrs,
                     ParallelTensorShape const &input_shape) {
  TensorShape unpar =
      PROPAGATE_ERR(get_output_shape(attrs, get_reduced_shape(input_shape)));

  ParallelTensorDimDegrees degrees = PROPAGATE_ERR(
      get_output_parallel_dim_degrees(attrs, get_parallel_degrees(input_shape)));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_gamma_weights_shape(LayerNormAttrs const &attrs,
                            ParallelTensorShape const &input_shape) {
  TensorShape unpar = PROPAGATE_ERR(
      get_gamma_weights_shape(attrs, get_reduced_shape(input_shape)));

  ParallelTensorDimDegrees degrees =
      PROPAGATE_ERR(get_gamma_weights_parallel_dim_degrees(
          attrs, get_parallel_degrees(input_shape)));

  return lift_to_parallel_with_degrees(unpar, degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &attrs,
                           ParallelTensorShape const &input_shape) {
  if (!attrs.elementwise_affine) {
    return tl::unexpected(
        "No beta weights exist for attrs.elementwise_affine = false");
  }

  return get_gamma_weights_shape(attrs, input_shape);
}

tl::expected<std::unordered_map<TensorSlotName, ParallelTensorShape>,
             std::string>
    get_weight_shapes(LayerNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {

  ParallelTensorShape gamma_shape =
      PROPAGATE_ERR(get_gamma_weights_shape(attrs, input_shape));
  ParallelTensorShape beta_shape =
      PROPAGATE_ERR(get_beta_weights_shape(attrs, input_shape));

  return std::unordered_map<TensorSlotName, ParallelTensorShape>{
      {
          TensorSlotName::GAMMA,
          gamma_shape,
      },
      {
          TensorSlotName::BETA,
          beta_shape,
      },
  };
}

std::unordered_map<TensorSlotName, InitializerAttrs>
    get_initializers(LayerNormAttrs const &attrs) {
  if (attrs.elementwise_affine) {
    InitializerAttrs gamma_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{1}}}};

    InitializerAttrs beta_initializer =
        InitializerAttrs{ConstantInitializerAttrs{DataTypeValue{float{0}}}};

    return {
        {TensorSlotName::GAMMA, gamma_initializer},
        {TensorSlotName::BETA, beta_initializer},
    };
  } else {
    return {};
  }
}

OperatorTaskSpace
    get_operator_task_space(LayerNormAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      throw_if_unexpected(
          get_output_parallel_dim_degrees(attrs, input_degrees));

  return get_operator_task_space_matching_parallel_tensor_dim_degrees(
      output_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  return get_identity_mapping(get_operator_task_space(attrs, input_degrees),
                              input_degrees);
}

static ParallelTensorSpaceToParallelTensorSpaceMapping
    get_input_to_weight_mapping(LayerNormAttrs const &attrs,
                                ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees weight_degrees = throw_if_unexpected(
      get_gamma_weights_parallel_dim_degrees(attrs, input_degrees));

  DownProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>
      input_to_weight = make_empty_down_projection<parallel_tensor_dim_idx_t,
                                                   parallel_tensor_dim_idx_t>();

  project_dims(input_to_weight, {sum_dim_idx()}, sum_dim_idx());

  std::vector<ff_dim_t> layer_norm_dim_idxs =
      filter(vector_of(get_idxs(input_degrees.shard_degrees)),
             [&](ff_dim_t const &dim_idx) {
               return contains(attrs.axes, dim_idx);
             });

  for (int i = 0; i < layer_norm_dim_idxs.size(); i++) {
    project_dims(input_to_weight,
                 {shard_dim_idx(layer_norm_dim_idxs.at(i))},
                 shard_dim_idx(ff_dim_t{nonnegative_int{i}}));
  }

  std::unordered_set<parallel_tensor_dim_idx_t> non_layer_norm_dims =
      unordered_set_of(transform(
          filter(vector_of(get_idxs(input_degrees.shard_degrees)),
                 [&](ff_dim_t const &dim_idx) {
                   return !contains(attrs.axes, dim_idx);
                 }),
          [](ff_dim_t const &dim_idx) { return shard_dim_idx(dim_idx); }));
  non_layer_norm_dims.insert(discard_copy_dim_idx());
  project_dims(input_to_weight, non_layer_norm_dims, discard_copy_dim_idx());

  return parallel_tensor_space_mapping_from_projection(
      DimProjection{input_to_weight}, input_degrees, weight_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_gamma_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  return operator_ptensor_space_mapping_from_composition(
      get_operator_to_input_mapping(attrs, input_degrees),
      get_input_to_weight_mapping(attrs, input_degrees));
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_beta_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  return get_operator_to_gamma_mapping(attrs, input_degrees);
}

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees) {
  ParallelTensorDimDegrees output_degrees =
      throw_if_unexpected(
          get_output_parallel_dim_degrees(attrs, input_degrees));

  return get_identity_mapping(get_operator_task_space(attrs, input_degrees),
                              output_degrees);
}

} // namespace FlexFlow
