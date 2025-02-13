/**
 * @file position_evaluation.cpp
 * @brief Implementation of the position evaluation class.
 */

#include "position_evaluation.h"

#include <algorithm>
#include <bit>
#include <ranges>
#include <tuple>

#include "search_tree_base.h"

namespace {

constexpr int transform_action(const int action, const int transformation) noexcept {
    if (action == 64) {
        return action;
    }
    int row = action / 8;
    int col = action % 8;
    if (const bool horizontal_flip = transformation % 2 == 1; horizontal_flip) {
        col = 7 - col;
    }
    const int num_rotations = transformation / 2;
    for (const int i : std::views::iota(0, num_rotations)) {
        std::tie(row, col) = std::make_tuple(col, 7 - row);
    }
    return row * 8 + col;
}

constexpr std::array<std::array<int, 65>, 8> compute_transformed_actions() noexcept {
    std::array<std::array<int, 65>, 8> transformed_actions;
    for (const int transformation : std::views::iota(0, 8)) {
        for (const int action : std::views::iota(0, 65)) {
            transformed_actions[transformation][action] = transform_action(action, transformation);
        }
    }
    return transformed_actions;
}

constexpr std::array<std::array<int, 65>, 8> TRANSFORMED_ACTIONS = compute_transformed_actions();

}  // namespace

othello::PositionEvaluation::PositionEvaluation(const int history_size)
    : m_history_size(history_size), m_input_features((history_size * 2 + 1) * 64) {}

bool othello::PositionEvaluation::set_position(
    const SearchTreeBase &search_tree, const std::uint32_t leaf_index, std::mt19937 &random_engine
) noexcept {
    m_leaf_index = leaf_index;

    const Position &leaf_position = search_tree.nodes()[leaf_index].position;
    if (leaf_position.is_terminal()) {
        m_is_result_ready = true;
        const int player1_count = std::popcount(leaf_position.player1_discs());
        const int player2_count = std::popcount(leaf_position.player2_discs());
        if (player1_count > player2_count) {
            m_player1_action_value = 1.f;
        } else if (player1_count < player2_count) {
            m_player1_action_value = -1.f;
        } else {
            m_player1_action_value = 0.f;
        }
        return false;
    }

    m_player = leaf_position.player();
    m_transformation = std::uniform_int_distribution<int>(0, 7)(random_engine);

    std::uint32_t node_index = leaf_index;
    const SearchNode *node = &search_tree.nodes()[node_index];

    auto input_feature_it = std::fill_n(m_input_features.begin(), 64, leaf_position.player() - 1.f);
    do {
        const std::uint64_t player1_discs = node->position.player1_discs();
        const std::uint64_t player2_discs = node->position.player2_discs();
        for (std::uint64_t square_mask = std::uint64_t(1) << 63;
             const int original_index : std::views::iota(0, 64)) {
            const int transformed_index = TRANSFORMED_ACTIONS[m_transformation][original_index];
            *(input_feature_it + transformed_index) = float((player1_discs & square_mask) != 0);
            *(input_feature_it + (64 + transformed_index)) =
                float((player2_discs & square_mask) != 0);
            square_mask >>= 1;
        }
        if (node_index == 0) {
            break;
        }
        node_index = node->parent_index;
        node = &search_tree.nodes()[node_index];
    } while ((input_feature_it += 128) != m_input_features.end());
    std::fill(input_feature_it, m_input_features.end(), 0.f);

    m_is_result_ready = false;

    return true;
}

void othello::PositionEvaluation::set_result(const float *const policy, const float value) {
    {
        std::lock_guard<std::mutex> lock(m_result_mutex);
        std::copy_n(policy, 64, m_policy.begin());
        m_player1_action_value = m_player == 1 ? value : -value;
        m_is_result_ready = true;
    }
    m_result_condition_variable.notify_one();
}

void othello::PositionEvaluation::wait_for_result() {
    std::unique_lock<std::mutex> lock(m_result_mutex);
    m_result_condition_variable.wait(lock, [this] { return m_is_result_ready; });
}

float othello::PositionEvaluation::get_prior_probability(const int action) const noexcept {
    const int transformed_action = TRANSFORMED_ACTIONS[m_transformation][action];
    return m_policy[transformed_action];
}
