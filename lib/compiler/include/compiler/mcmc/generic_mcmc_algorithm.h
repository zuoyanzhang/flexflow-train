#ifndef _FLEXFLOW_COMPILER_MCMC_GENERIC_MCMC_ALGORITHM_H
#define _FLEXFLOW_COMPILER_MCMC_GENERIC_MCMC_ALGORITHM_H

#include "compiler/mcmc/generic_mcmc_config.dtg.h"
#include "utils/containers/transform.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/optional.h"
#include "utils/random_utils.h"

namespace FlexFlow {

// SamplingFn : State -> std::optional<State>
// CostFn : State -> float

template <typename State, typename SamplingFn, typename CostFn>
State run_mcmc(State const &starting_state,
               SamplingFn const &sampler,
               CostFn const &cost,
               GenericMCMCConfig const &search_config) {
  State best_state = starting_state;
  State current_state = best_state;
  float best_cost = cost(best_state);
  float current_cost = best_cost;
  for (nonnegative_int i : nonnegative_range(search_config.num_iterations)) {
    std::optional<State> maybe_new_state =
        transform(sampler(current_state), [&](State const &s) {
          float candidate_cost = cost(s);
          float delta = candidate_cost - current_cost;
          if (randf() < exp(-delta / search_config.temperature)) {
            current_cost = candidate_cost;
            if (candidate_cost < best_cost) {
              best_state = s;
              best_cost = candidate_cost;
            }
            return s;
          }
          return current_state;
        });
    current_state = or_else(maybe_new_state, [&]() { return current_state; });
  }
  return best_state;
}

} // namespace FlexFlow

#endif
