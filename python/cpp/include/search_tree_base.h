/**
 * @file search_tree_base.h
 * @brief Search tree base class.
 */

#ifndef OTHELLO_ALPHAZERO_SEARCH_TREE_BASE_H
#define OTHELLO_ALPHAZERO_SEARCH_TREE_BASE_H

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>

#include "position.h"
#include "position_evaluation.h"
#include "search_node.h"
#include "thread_safe_queue.h"

namespace othello {

/**
 * @brief Base class for search trees.
 */
class SearchTreeBase {
public:
    /**
     * @brief Constructs a search tree.
     * @param c_puct_init `c_init` for the PUCT formula.
     * @param c_puct_base `c_base` for the PUCT formula.
     */
    SearchTreeBase(float c_puct_init, float c_puct_base);

    /**
     * @brief Gets the `c_init` for the PUCT formula.
     * @return `c_init` for the PUCT formula.
     */
    float c_puct_init() const noexcept {
        return m_c_puct_init;
    }

    /**
     * @brief Sets the `c_init` for the PUCT formula.
     * @param value `c_init` for the PUCT formula.
     */
    void set_c_puct_init(float value);

    /**
     * @brief Gets the `c_base` for the PUCT formula.
     * @return `c_base` for the PUCT formula.
     */
    float c_puct_base() const noexcept {
        return m_c_puct_base;
    }

    /**
     * @brief Sets the `c_base` for the PUCT formula.
     * @param value `c_base` for the PUCT formula.
     */
    void set_c_puct_base(float value);

    /**
     * @brief Resets the search tree to the initial position.
     */
    void reset_position();

    /**
     * @brief Gets the nodes in the search tree.
     * @return Reference to the vector of nodes.
     */
    const std::vector<SearchNode> &nodes() const noexcept {
        return m_nodes;
    }

    /**
     * @brief Gets the index of the root node.
     * @return Index of the root node.
     */
    std::uint32_t root_index() const noexcept {
        return m_root_index;
    }

    /**
     * @brief Applies an action to the current position and updates the search tree accordingly.
     */
    void apply_action(int action);

protected:
    /**
     * @brief Runs the forward selection phase and sets up a neural network evaluation if
     *     applicable.
     * @tparam Derived Type of the derived class.
     * @tparam DirichletNoise Whether Dirichlet noises are used.
     * @tparam VirtualLoss Whether virtual losses are used.
     * @param evaluation Position evaluation object.
     * @param queue Queue to push the position evaluation object to.
     * @param random_engine Random engine for sampling Dirichlet noises and transformations.
     */
    template <typename Derived, bool DirichletNoise, bool VirtualLoss>
    void forward_and_evaluate_impl(
        PositionEvaluation &evaluation,
        ThreadSafeQueue<PositionEvaluation *> &queue,
        std::mt19937 &random_engine
    );

    /**
     * @brief Expands a leaf node if applicable and runs the backward pass.
     * @tparam Derived Type of the derived class.
     * @tparam VirtualLoss Whether virtual losses are used.
     * @param evaluation Position evaluation object.
     */
    template <typename Derived, bool VirtualLoss>
    void expand_and_backward_impl(PositionEvaluation &evaluation);

private:
    template <typename Derived, typename... Args>
    std::uint32_t forward(Args &...args);

    template <typename Derived, typename... Args>
    const SearchNode &select_child(const SearchNode &node, Args &...args);

    void propagate_virtual_loss(std::uint32_t leaf_index) noexcept;

    template <std::invocable<int> Invocable>
    void expand(std::uint32_t leaf_index, Invocable get_prior_probability);

    template <bool VirtualLoss>
    void backward(std::uint32_t leaf_index, float player1_action_value) noexcept;

    std::uint32_t clone_nodes(
        const SearchNode &old_node,
        std::uint32_t new_parent_index,
        std::vector<SearchNode> &new_nodes
    ) const;

    float m_c_puct_init;
    float m_c_puct_base;

    std::vector<SearchNode> m_nodes;
    std::uint32_t m_root_index;

    std::array<int, 60> m_legal_actions;
};

}  // namespace othello

template <typename Derived, bool DirichletNoise, bool VirtualLoss>
void othello::SearchTreeBase::forward_and_evaluate_impl(
    PositionEvaluation &evaluation,
    ThreadSafeQueue<PositionEvaluation *> &queue,
    std::mt19937 &random_engine
) {
    {
        const auto search_tree_lock = static_cast<Derived *>(this)->lock_search_tree();
        std::uint32_t leaf_index;
        if constexpr (DirichletNoise) {
            leaf_index = forward<Derived>(random_engine);
        } else {
            leaf_index = forward<Derived>();
        }
        if (!evaluation.set_position(*this, leaf_index, random_engine)) {
            return;
        }
        if constexpr (VirtualLoss) {
            propagate_virtual_loss(leaf_index);
        }
    }
    queue.emplace(&evaluation);
}

template <typename Derived, bool VirtualLoss>
void othello::SearchTreeBase::expand_and_backward_impl(PositionEvaluation &evaluation) {
    evaluation.wait_for_result();
    const auto search_tree_lock = static_cast<Derived *>(this)->lock_search_tree();
    const auto leaf_index = evaluation.leaf_index();
    const SearchNode &leaf = m_nodes[leaf_index];
    if (!leaf.position.is_terminal() && leaf.child_indices.empty()) {
        expand(leaf_index, [&evaluation](const int action) {
            return evaluation.get_prior_probability(action);
        });
    }
    const float player1_action_value = evaluation.player1_action_value();
    if constexpr (VirtualLoss) {
        if (leaf.position.is_terminal()) {
            backward<false>(leaf_index, player1_action_value);
        } else {
            backward<true>(leaf_index, player1_action_value);
        }
    } else {
        backward<false>(leaf_index, player1_action_value);
    }
}

template <typename Derived, typename... Args>
std::uint32_t othello::SearchTreeBase::forward(Args &...args) {
    SearchNode &root = m_nodes[m_root_index];
    const SearchNode *node = &root;
    while (!(node->position.is_terminal() || node->child_indices.empty())) {
        node = &select_child<Derived>(*node, args...);
    }
    // The root node visit count is used for computing exploration rates.
    ++root.visit_count;
    return static_cast<std::uint32_t>(node - m_nodes.data());
}

template <typename Derived, typename... Args>
const othello::SearchNode &othello::SearchTreeBase::select_child(
    const SearchNode &node, Args &...args
) {
    const SearchNode &first_child = m_nodes[node.child_indices.front()];
    if (node.child_indices.size() == 1) {
        return first_child;
    }

    const float exploration_rate =
        std::log((1 + node.visit_count) / m_c_puct_base + 1.f) + m_c_puct_init;
    std::uint32_t total_visit_count = 0;
    for (const auto child_index : node.child_indices) {
        total_visit_count += m_nodes[child_index].visit_count;
    }
    const float u_multiplier = exploration_rate * std::sqrt(static_cast<float>(total_visit_count));

    if constexpr (sizeof...(args) > 0) {
        // The arguments are used for generating Dirichlet noise.
        static_assert(std::is_base_of_v<SearchTreeBase, Derived>);
        Derived &derived = static_cast<Derived &>(*this);

        if (&node == &m_nodes[m_root_index]) {
            const auto [dirichlet_noise, dirichlet_noise_sum] =
                derived.sample_dirichlet_noise(node.child_indices.size(), args...);
            const auto get_ucb = [u_multiplier,
                                  probability_multiplier(1.f - derived.m_dirichlet_epsilon),
                                  noise_multiplier(
                                      derived.m_dirichlet_epsilon / dirichlet_noise_sum
                                  )](const SearchNode &child, float noise) {
                const float probability =
                    child.prior_probability * probability_multiplier + noise * noise_multiplier;
                return child.mean_action_value +
                       u_multiplier * probability / static_cast<float>(1 + child.visit_count);
            };
            const SearchNode *max_child = &first_child;
            float max_ucb = get_ucb(first_child, dirichlet_noise[0]);
            for (const std::size_t i :
                 std::views::iota(std::size_t(1), node.child_indices.size())) {
                const SearchNode &child = m_nodes[node.child_indices[i]];
                if (const float ucb = get_ucb(child, dirichlet_noise[i]); ucb > max_ucb) {
                    max_child = &child;
                    max_ucb = ucb;
                }
            }
            return *max_child;
        }
    }

    const auto get_ucb = [u_multiplier](const SearchNode &child) {
        return child.mean_action_value +
               u_multiplier * child.prior_probability / static_cast<float>(1 + child.visit_count);
    };
    const SearchNode *max_child = &first_child;
    float max_ucb = get_ucb(first_child);
    for (const auto child_index : node.child_indices | std::views::drop(1)) {
        const SearchNode &child = m_nodes[child_index];
        if (const float ucb = get_ucb(child); ucb > max_ucb) {
            max_child = &child;
            max_ucb = ucb;
        }
    }
    return *max_child;
}

template <std::invocable<int> Invocable>
void othello::SearchTreeBase::expand(
    const std::uint32_t leaf_index, Invocable get_prior_probability
) {
    const int num_legal_moves = std::popcount(m_nodes[leaf_index].position.legal_moves());
    const std::size_t num_legal_actions = num_legal_moves == 0 ? 1 : num_legal_moves;
    m_nodes.reserve(m_nodes.size() + num_legal_actions);

    SearchNode &leaf = m_nodes[leaf_index];
    leaf.child_indices.reserve(num_legal_actions);

    for (const int action : std::ranges::subrange(
             m_legal_actions.begin(), leaf.position.legal_actions(m_legal_actions.begin())
         )) {
        leaf.child_indices.push_back(static_cast<std::uint32_t>(m_nodes.size()));
        m_nodes.push_back(SearchNode{
            .position = leaf.position.apply_action(action),
            .parent_index = leaf_index,
            .prior_probability = get_prior_probability(action)
        });
    }
}

template <bool VirtualLoss>
void othello::SearchTreeBase::backward(
    const std::uint32_t leaf_index, const float player1_action_value
) noexcept {
    const SearchNode *const root = &m_nodes[m_root_index];
    SearchNode *node = &m_nodes[leaf_index];
    if (node == root) {
        return;
    }

    // The sign of the action-value is with respect to the parent node.
    float action_value = m_nodes[node->parent_index].position.player() == 1 ? player1_action_value
                                                                            : -player1_action_value;

    do {
        if constexpr (VirtualLoss) {
            node->total_action_value += 1.f + action_value;
        } else {
            ++node->visit_count;
            node->total_action_value += action_value;
        }
        node->mean_action_value = node->total_action_value / node->visit_count;

        node = &m_nodes[node->parent_index];
        action_value = -action_value;
    } while (node != root);
}

#endif  // OTHELLO_ALPHAZERO_SEARCH_TREE_BASE_H
