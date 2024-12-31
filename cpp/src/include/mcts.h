/// @file mcts.h
/// @brief Declaration of the Monte Carlo Tree Search algorithm.

#ifndef OTHELLO_MCTS_MCTS_H
#define OTHELLO_MCTS_MCTS_H

#include <mutex>
#include <vector>

#include "position.h"

namespace othello {

/// @brief Node in the MCTS search tree.
///
class MCTSNode {
public:
    MCTSNode(
        const Position &position,
        int previous_action = 64,
        int visit_count = 0,
        float total_action_value = 0.0f,
        float mean_action_value = 0.0f,
        float prior_probability = 1.0f
    )
        : position(position),
          previous_action(previous_action),
          visit_count(visit_count),
          total_action_value(total_action_value),
          mean_action_value(mean_action_value),
          prior_probability(prior_probability) {}

    Position position;
    int previous_action;
    int visit_count;
    float total_action_value;
    float mean_action_value;
    float prior_probability;
    std::vector<unsigned> children;
};

/// @brief Monte Carlo Tree Search algorithm.
///
class MCTS {
public:
    MCTS(
        int num_simulations = 1600,
        int batch_size = 16,
        int num_threads = 16,
        float exploration_weight = 1.0f,
        float dirichlet_epsilon = 0.25f,
        float dirichlet_alpha = 0.03f
    );

    /// @brief Clears the search tree and resets the root node to the given
    ///     position.
    /// @param position Position to reset to.
    void reset_position(const Position &position);

    /// @brief Returns the position of the root node of the search tree.
    /// @return Position of the root node.
    Position root_position() const;

    /// @brief Applies an action to the current position and prunes the search
    ///     tree accordingly.
    /// @param action Action to apply.
    void apply_action(int action);

    int num_simulations() const noexcept {
        return _num_simulations;
    }

    void set_num_simulations(int value) noexcept;

    int batch_size() const noexcept {
        return _batch_size;
    }

    void set_batch_size(int value) noexcept;

    int num_threads() const noexcept {
        return _num_threads;
    }

    void set_num_threads(int value) noexcept;

    float exploration_weight() const noexcept {
        return _exploration_weight;
    }

    void set_exploration_weight(float value) noexcept;

    float dirichlet_epsilon() const noexcept {
        return _dirichlet_epsilon;
    }

    void set_dirichlet_epsilon(float value) noexcept;

    float dirichlet_alpha() const noexcept {
        return _dirichlet_alpha;
    }

    void set_dirichlet_alpha(float value) noexcept;

private:
    int _num_simulations;
    int _batch_size;
    int _num_threads;
    float _exploration_weight;
    float _dirichlet_epsilon;
    float _dirichlet_alpha;

    std::vector<MCTSNode> _search_tree;
    std::mutex _search_tree_mutex;
};

} // namespace othello

#endif // OTHELLO_MCTS_MCTS_H
