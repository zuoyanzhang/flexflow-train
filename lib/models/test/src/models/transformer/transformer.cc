#include "models/transformer/transformer.h"
#include "pcg/computation_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_transformer_computation_graph") {
    TransformerConfig config = get_default_transformer_config();

    ComputationGraph result = get_transformer_computation_graph(config);

    SUBCASE("num layers") {
      int result_num_layers = get_layers(result).size();
      int correct_num_layers = 258;
      CHECK(result_num_layers == correct_num_layers);
    }
  }

  TEST_CASE("get_llama2_7b_like_config") {
    TransformerConfig config = get_llama2_7b_like_config();

    CHECK(config.num_encoder_layers == 32_p);
    CHECK(config.num_features == 4096_p);
    CHECK(config.dim_feedforward == 11008_p);
    CHECK(config.num_heads == 32_p);
    CHECK(config.sequence_length == 2048_p);
    CHECK(config.batch_size == 1_p);
    CHECK(config.vocab_size == 32000_p);
  }

  TEST_CASE("get_decoder_only_transformer_computation_graph") {
    TransformerConfig config = TransformerConfig{
        /*num_features=*/128_p,
        /*sequence_length=*/16_p,
        /*batch_size=*/2_p,
        /*dim_feedforward=*/256_p,
        /*num_heads=*/8_p,
        /*num_encoder_layers=*/2_p,
        /*num_decoder_layers=*/0_p,
        /*dropout=*/0.0,
        /*layer_norm_eps=*/1e-05,
        /*vocab_size=*/1024_p,
    };

    ComputationGraph result =
        get_decoder_only_transformer_computation_graph(config);

    CHECK(get_layers(result).size() > 0);
  }
}
