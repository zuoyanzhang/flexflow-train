#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LAYER_NORM_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_LAYER_NORM_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

std::unordered_map<TensorSlotName, IncomingTensorRole>
    get_layer_norm_incoming_tensor_roles(LayerNormAttrs const &);

tl::expected<TensorShape, std::string> get_output_shape(LayerNormAttrs const &,
                                                        TensorShape const &);
tl::expected<TensorShape, std::string>
    get_gamma_weights_shape(LayerNormAttrs const &, TensorShape const &);
tl::expected<TensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &, TensorShape const &);

tl::expected<std::unordered_map<TensorSlotName, TensorShape>, std::string>
    get_weight_shapes(LayerNormAttrs const &attrs,
                      TensorShape const &input_shape);

tl::expected<ParallelTensorDimDegrees, std::string>
    get_output_parallel_dim_degrees(LayerNormAttrs const &,
                                    ParallelTensorDimDegrees const &);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_gamma_weights_parallel_dim_degrees(LayerNormAttrs const &,
                                           ParallelTensorDimDegrees const &);
tl::expected<ParallelTensorDimDegrees, std::string>
    get_beta_weights_parallel_dim_degrees(LayerNormAttrs const &,
                                          ParallelTensorDimDegrees const &);

tl::expected<std::unordered_map<TensorSlotName, ParallelTensorDimDegrees>,
             std::string>
    get_weight_parallel_dim_degrees(
        LayerNormAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees);

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LayerNormAttrs const &, ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_gamma_weights_shape(LayerNormAttrs const &,
                            ParallelTensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_beta_weights_shape(LayerNormAttrs const &, ParallelTensorShape const &);

tl::expected<std::unordered_map<TensorSlotName, ParallelTensorShape>,
             std::string>
    get_weight_shapes(LayerNormAttrs const &attrs,
                      ParallelTensorShape const &input_shape);

/**
 * @brief Chosen to match pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/normalization.py#L210-L214
 */
std::unordered_map<TensorSlotName, InitializerAttrs>
    get_initializers(LayerNormAttrs const &attrs);

OperatorTaskSpace
    get_operator_task_space(LayerNormAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceMapping get_operator_to_input_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);
OperatorSpaceToParallelTensorSpaceMapping get_operator_to_gamma_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);
OperatorSpaceToParallelTensorSpaceMapping get_operator_to_beta_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);
OperatorSpaceToParallelTensorSpaceMapping get_operator_to_output_mapping(
    LayerNormAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
