#ifndef _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_TRANSFORMER_H
#define _FLEXFLOW_LIB_MODELS_INCLUDE_MODELS_TRANSFORMER_TRANSFORMER_H

#include "models/transformer/transformer_config.dtg.h"
#include "pcg/computation_graph_builder.h"

namespace FlexFlow {

// Helper functions to construct the Transformer model
tensor_guid_t create_transformer_feedforward_network(ComputationGraphBuilder &,
                                                     TransformerConfig const &,
                                                     tensor_guid_t const &);
tensor_guid_t create_transformer_encoder_layer(ComputationGraphBuilder &,
                                               TransformerConfig const &,
                                               tensor_guid_t const &);
tensor_guid_t create_transformer_decoder_layer(ComputationGraphBuilder &,
                                               TransformerConfig const &,
                                               tensor_guid_t const &,
                                               tensor_guid_t const &);

tensor_guid_t create_transformer_encoder(ComputationGraphBuilder &,
                                         TransformerConfig const &,
                                         tensor_guid_t const &);
tensor_guid_t create_transformer_decoder(ComputationGraphBuilder &,
                                         TransformerConfig const &,
                                         tensor_guid_t const &,
                                         tensor_guid_t const &);
tensor_guid_t create_llama2_7b_like_decoder_layer(ComputationGraphBuilder &,
                                                  TransformerConfig const &,
                                                  tensor_guid_t const &);

/**
 * @brief Get the base config from the Attention Is All You Need paper.
 *
 * @details See the first row of the Table 3 at the top of p. 9 in
 * https://arxiv.org/abs/1706.03762
 */
TransformerConfig get_default_transformer_config();

/**
 * @brief Get a Llama2-7B-shaped benchmark config.
 *
 * @details This config mirrors the main dimensions in
 * megatron-lm-nv/examples/llama/llama2_7b.sh. It is intentionally only a
 * benchmark shape: the graph uses only FlexFlow Train operators that currently
 * have machine-mapping support. It does not implement real attention,
 * LayerNorm/RMSNorm, RoPE, causal masking, BF16, checkpoint loading, or
 * tokenizer/data ingestion.
 */
TransformerConfig get_llama2_7b_like_config();

/**
 * @brief Get the Transformer computation graph.
 *
 * @param config The config of Transformer model.
 * @return The PCG of a Transformer model.
 */
ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &config);

/**
 * @brief Get a decoder-only Transformer-like computation graph.
 *
 * @details This graph is intended for benchmarking auto-parallel search against
 * Megatron-style decoder-only runs. It uses Q/K/V/O linear projections as an
 * attention parameter-count stand-in, a GELU-gated feed-forward network as a
 * SwiGLU stand-in, and an LM-head projection.
 */
ComputationGraph
    get_decoder_only_transformer_computation_graph(TransformerConfig const &);

/**
 * @brief Get the Llama2-7B-shaped decoder-only benchmark graph.
 */
ComputationGraph get_llama2_7b_like_computation_graph();

} // namespace FlexFlow

#endif
