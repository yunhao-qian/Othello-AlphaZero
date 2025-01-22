/// @file self_play.cpp
/// @brief TODO

#include "self_play.h"

#include <cmath>
#include <stdexcept>

othello::SelfPlay::SelfPlay(
    int history_size,
    const std::string &torch_device,
    bool torch_pin_memory,
    int num_threads,
    int num_simulations,
    float c_puct_base,
    float c_puct_init,
    float dirichlet_epsilon,
    float dirichlet_alpha
)
    : _history_size(history_size),
      _torch_device(torch_device),
      _torch_pin_memory(torch_pin_memory),
      _num_threads(num_threads),
      _num_simulations(num_simulations),
      _c_puct_base(c_puct_base),
      _c_puct_init(c_puct_init),
      _dirichlet_epsilon(dirichlet_epsilon),
      _dirichlet_alpha(dirichlet_alpha) {
    if (history_size < 1) {
        throw std::invalid_argument(
            "Expected history_size >= 1, but got " +
            std::to_string(history_size) + "."
        );
    }
    if (num_threads < 1) {
        throw std::invalid_argument(
            "Expected num_threads >= 1, but got " +
            std::to_string(num_threads) + "."
        );
    }
    if (num_simulations < 1) {
        throw std::invalid_argument(
            "Expected num_simulations >= 1, but got " +
            std::to_string(num_simulations) + "."
        );
    }
    if (!(0.0f < c_puct_base && std::isfinite(c_puct_base))) {
        throw std::invalid_argument(
            "Expected 0.0 < c_puct_base < infinity, but got " +
            std::to_string(c_puct_base) + "."
        );
    }
    if (!(0.0f <= c_puct_init && std::isfinite(c_puct_init))) {
        throw std::invalid_argument(
            "Expected 0.0f <= c_puct_init < infinity, but got " +
            std::to_string(c_puct_init) + "."
        );
    }
    if (!(0.0f <= dirichlet_epsilon && dirichlet_epsilon <= 1.0f)) {
        throw std::invalid_argument(
            "Expected 0.0 <= dirichlet_epsilon <= 1.0, but got " +
            std::to_string(dirichlet_epsilon) + "."
        );
    }
    if (!(0.0f <= dirichlet_alpha && std::isfinite(dirichlet_alpha))) {
        throw std::invalid_argument(
            "Expected 0.0f <= dirichlet_alpha < infinity, but got " +
            std::to_string(dirichlet_alpha) + "."
        );
    }
}
