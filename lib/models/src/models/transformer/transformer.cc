#include "models/transformer/transformer.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

TransformerConfig get_default_transformer_config() {
  return TransformerConfig{/*num_features=*/512_p,
                           /*sequence_length=*/512_p,
                           /*batch_size=*/64_p,
                           /*dim_feedforward=*/2048_p,
                           /*num_heads=*/8_p,
                           /*num_encoder_layers=*/6_p,
                           /*num_decoder_layers=*/6_p,
                           /*dropout=*/0.1,
                           /*layer_norm_eps=*/1e-05,
                           /*vocab_size=*/64_p};
}

TransformerConfig get_llama2_7b_like_config() {
  return TransformerConfig{
      /*num_features=*/4096_p,
      /*sequence_length=*/2048_p,
      /*batch_size=*/1_p,
      /*dim_feedforward=*/11008_p,
      /*num_heads=*/32_p,
      /*num_encoder_layers=*/1_p,
      /*num_decoder_layers=*/32_p,
      /*dropout=*/0.0,
      /*layer_norm_eps=*/1e-05,
      /*vocab_size=*/32000_p,
  };
}

tensor_guid_t create_feedforward_network(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input) {
  tensor_guid_t layer1_out = cgb.dense(
      input, config.dim_feedforward, Activation::RELU, /*use_bias=*/true);
  tensor_guid_t dropout_out = cgb.dropout(layer1_out, config.dropout);
  tensor_guid_t layer2_out = cgb.dense(dropout_out,
                                       config.num_features,
                                       /*activation=*/std::nullopt,
                                       /*use_bias=*/true);
  return cgb.dropout(layer2_out, config.dropout);
};

tensor_guid_t create_transformer_encoder_layer(ComputationGraphBuilder &cgb,
                                               TransformerConfig const &config,
                                               tensor_guid_t const &input) {
  std::set<relative_ff_dim_t> layer_norm_axis = {
      relative_ff_dim_t{-1}}; // Normalize the last dim
  positive_int kdim = positive_int{config.dim_feedforward / config.num_heads};
  positive_int vdim = positive_int{config.dim_feedforward / config.num_heads};
  tensor_guid_t self_attention =
      cgb.multihead_attention(/*query=*/input,
                              /*key=*/input,
                              /*value=*/input,
                              /*embed_dim=*/config.num_features,
                              /*num_heads=*/config.num_heads,
                              /*kdim=*/kdim,
                              /*vdim=*/vdim,
                              /*dropout=*/config.dropout,
                              /*bias=*/false);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t normalized = cgb.layer_norm(cgb.add(self_attention, input),
                                            layer_norm_axis,
                                            /*elementwise_affine=*/true,
                                            config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, normalized));

  tensor_guid_t feedforward_output =
      create_feedforward_network(cgb, config, normalized);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, feedforward_output));
  return cgb.layer_norm(cgb.add(normalized, feedforward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_transformer_encoder(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_encoder_layers; i++) {
    t = create_transformer_encoder_layer(cgb, config, t);
  }
  return t;
};

tensor_guid_t
    create_transformer_decoder_layer(ComputationGraphBuilder &cgb,
                                     TransformerConfig const &config,
                                     tensor_guid_t const &input,
                                     tensor_guid_t const &encoder_output) {
  std::set<relative_ff_dim_t> layer_norm_axis = {
      relative_ff_dim_t{-1}}; // Normalize the last dim
  positive_int kdim = positive_int{config.dim_feedforward / config.num_heads};
  positive_int vdim = positive_int{config.dim_feedforward / config.num_heads};
  tensor_guid_t self_attention =
      cgb.multihead_attention(/*query=*/input,
                              /*key=*/input,
                              /*value=*/input,
                              /*embed_dim=*/config.num_features,
                              /*num_heads=*/config.num_heads,
                              /*kdim=*/kdim,
                              /*vdim=*/vdim,
                              /*dropout=*/config.dropout,
                              /*bias=*/false);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention));

  tensor_guid_t self_attention_normalized =
      cgb.layer_norm(cgb.add(input, self_attention),
                     layer_norm_axis,
                     /*elementwise_affine=*/true,
                     config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, self_attention_normalized));

  tensor_guid_t mha =
      cgb.multihead_attention(/*query=*/self_attention_normalized,
                              /*key=*/encoder_output,
                              /*value=*/encoder_output,
                              /*embed_dim=*/config.num_features,
                              /*num_heads=*/config.num_heads,
                              /*kdim=*/kdim,
                              /*vdim=*/vdim,
                              /*dropout=*/config.dropout,
                              /*bias=*/false);
  assert(are_tensor_guid_shapes_equivalent(cgb.computation_graph, input, mha));

  tensor_guid_t mha_normalized =
      cgb.layer_norm(cgb.add(self_attention_normalized, mha),
                     layer_norm_axis,
                     /*elementwise_affine=*/true,
                     config.layer_norm_eps);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, mha_normalized));

  tensor_guid_t feedforward_output =
      create_feedforward_network(cgb, config, mha_normalized);
  assert(are_tensor_guid_shapes_equivalent(
      cgb.computation_graph, input, feedforward_output));

  return cgb.layer_norm(cgb.add(mha_normalized, feedforward_output),
                        layer_norm_axis,
                        /*elementwise_affine=*/true,
                        config.layer_norm_eps);
}

tensor_guid_t create_transformer_decoder(ComputationGraphBuilder &cgb,
                                         TransformerConfig const &config,
                                         tensor_guid_t const &input,
                                         tensor_guid_t const &encoder_output) {
  tensor_guid_t t = input;
  for (int i = 0; i < config.num_decoder_layers; i++) {
    t = create_transformer_decoder_layer(cgb, config, t, encoder_output);
  }
  return t;
}

static tensor_guid_t
    create_llama2_7b_like_feedforward_network(ComputationGraphBuilder &cgb,
                                              TransformerConfig const &config,
                                              tensor_guid_t const &input) {
  tensor_guid_t gate = cgb.dense(input,
                                 config.dim_feedforward,
                                 Activation::GELU,
                                 /*use_bias=*/false);
  tensor_guid_t up = cgb.dense(input,
                               config.dim_feedforward,
                               /*activation=*/std::nullopt,
                               /*use_bias=*/false);
  tensor_guid_t gated = cgb.multiply(gate, up);
  return cgb.dense(gated,
                   config.num_features,
                   /*activation=*/std::nullopt,
                   /*use_bias=*/false);
}

tensor_guid_t create_llama2_7b_like_decoder_layer(ComputationGraphBuilder &cgb,
                                                  TransformerConfig const &config,
                                                  tensor_guid_t const &input) {
  tensor_guid_t query = cgb.dense(input,
                                  config.num_features,
                                  /*activation=*/std::nullopt,
                                  /*use_bias=*/false);
  tensor_guid_t key = cgb.dense(input,
                                config.num_features,
                                /*activation=*/std::nullopt,
                                /*use_bias=*/false);
  tensor_guid_t value = cgb.dense(input,
                                  config.num_features,
                                  /*activation=*/std::nullopt,
                                  /*use_bias=*/false);
  tensor_guid_t qk = cgb.add(query, key);
  tensor_guid_t qkv = cgb.add(qk, value);
  tensor_guid_t self_attention = cgb.dense(qkv,
                                           config.num_features,
                                           /*activation=*/std::nullopt,
                                           /*use_bias=*/false);
  tensor_guid_t attention_residual = cgb.add(input, self_attention);

  tensor_guid_t feedforward_output =
      create_llama2_7b_like_feedforward_network(cgb, config, attention_residual);
  return cgb.add(attention_residual, feedforward_output);
}

ComputationGraph
    get_transformer_computation_graph(TransformerConfig const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{
          config.batch_size, config.sequence_length, config.num_features}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES, "input");
  tensor_guid_t target =
      cgb.create_input(input_shape, CreateGrad::YES, "target");

  tensor_guid_t encoder_output = create_transformer_encoder(cgb, config, input);
  tensor_guid_t decoder_output =
      create_transformer_decoder(cgb, config, target, encoder_output);

  tensor_guid_t out_prob = cgb.softmax(cgb.dense(decoder_output,
                                                 /*outDim=*/config.vocab_size,
                                                 Activation::RELU,
                                                 /*use_bias=*/true));
  return cgb.computation_graph;
}

ComputationGraph get_decoder_only_transformer_computation_graph(
    TransformerConfig const &config) {
  ComputationGraphBuilder cgb;

  TensorShape input_shape = TensorShape{
      TensorDims{FFOrdered<positive_int>{
          config.batch_size, config.sequence_length, config.num_features}},
      DataType::FLOAT,
  };
  tensor_guid_t input = cgb.create_input(input_shape, CreateGrad::YES, "input");

  tensor_guid_t hidden_states = input;
  for (int i = 0; i < config.num_decoder_layers; i++) {
    hidden_states =
        create_llama2_7b_like_decoder_layer(cgb, config, hidden_states);
  }

  tensor_guid_t logits = cgb.dense(hidden_states,
                                   /*outDim=*/config.vocab_size,
                                   /*activation=*/std::nullopt,
                                   /*use_bias=*/false,
                                   /*data_type=*/DataType::FLOAT,
                                   /*projection_initializer=*/std::nullopt,
                                   /*bias_initializer=*/std::nullopt,
                                   /*name=*/"lm_head");
  return cgb.computation_graph;
}

ComputationGraph get_llama2_7b_like_computation_graph() {
  return get_decoder_only_transformer_computation_graph(
      get_llama2_7b_like_config());
}

} // namespace FlexFlow
