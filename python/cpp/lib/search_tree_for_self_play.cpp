/**
 * @file search_tree_for_self_play.cpp
 * @brief Implementation of the search tree for self-play class.
 */

#include "search_tree_for_self_play.h"

#include <cmath>
#include <format>
#include <ranges>
#include <stdexcept>

othello::SearchTreeForSelfPlay::SearchTreeForSelfPlay(
    const float c_puct_init,
    const float c_puct_base,
    const float dirichlet_epsilon,
    const float dirichlet_alpha
)
    : SearchTreeBase(c_puct_init, c_puct_base) {
    set_dirichlet_epsilon(dirichlet_epsilon);
    set_dirichlet_alpha(dirichlet_alpha);
}

auto othello::SearchTreeForSelfPlay::set_dirichlet_epsilon(const float value) -> void {
    if (!(0.f <= value && value <= 1.f)) {
        throw std::invalid_argument(
            std::format("Expected 0 <= dirichlet_epsilon <= 1, but got {}.", value)
        );
    }
    m_dirichlet_epsilon = value;
}

auto othello::SearchTreeForSelfPlay::set_dirichlet_alpha(const float value) -> void {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::format("Expected dirichlet_alpha to be finite, but got {}.", value)
        );
    }
    if (!(value >= 0.f)) {
        throw std::invalid_argument(std::format("Expected dirichlet_alpha >= 0, but got {}.", value)
        );
    }
    m_dirichlet_alpha = value;
}

auto othello::SearchTreeForSelfPlay::sample_dirichlet_noise(
    const std::size_t num_children, std::mt19937 &random_engine
) -> std::tuple<const float *, float> {
    std::gamma_distribution<float> gamma_distribution(m_dirichlet_alpha, 1.f);
    float noise_sum = 0.f;
    for (float &noise : m_dirichlet_noise | std::views::take(num_children)) {
        noise = gamma_distribution(random_engine);
        noise_sum += noise;
    }
    if (noise_sum == 0.f) {
        noise_sum = 1.f;
    }
    return std::make_tuple(m_dirichlet_noise.data(), noise_sum);
}
