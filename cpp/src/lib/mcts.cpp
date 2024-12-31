/// @file mcts.cpp
/// @brief Implementation of the Monte Carlo Tree Search algorithm.

#include "mcts.h"

#include <algorithm>
#include <cmath>

othello::MCTS::MCTS(
    int num_simulations,
    int batch_size,
    int num_threads,
    float exploration_weight,
    float dirichlet_epsilon,
    float dirichlet_alpha
) {
    set_num_simulations(num_simulations);
    set_batch_size(batch_size);
    set_num_threads(num_threads);
    set_exploration_weight(exploration_weight);
    set_dirichlet_epsilon(dirichlet_epsilon);
    set_dirichlet_alpha(dirichlet_alpha);
    reset_position(Position::initial_position());
}

void othello::MCTS::reset_position(const Position &position) {
    _search_tree.clear();
    _search_tree.emplace_back(Position::initial_position());
}

othello::Position othello::MCTS::root_position() const {
    return _search_tree.front().position;
}

namespace {

using othello::MCTSNode;

/// @brief Collects the subtree rooted at the given node from the old tree to
///     the new tree.
/// @param old_tree Original search tree.
/// @param new_tree New search tree.
/// @param old_index Index of the root node in the old tree.
/// @return Index of the root node in the new tree.
unsigned collect_subtree(
    const std::vector<MCTSNode> &old_tree,
    std::vector<MCTSNode> &new_tree,
    unsigned old_index
) {
    const MCTSNode &old_node = old_tree[old_index];
    unsigned new_index = static_cast<unsigned>(new_tree.size());
    new_tree.emplace_back(
        old_node.position,
        old_node.previous_action,
        old_node.visit_count,
        old_node.total_action_value,
        old_node.mean_action_value,
        old_node.prior_probability
    );
    // This reference will not be invalidated because we reserve the capacity
    // of the new search tree.
    std::vector<unsigned> &new_children = new_tree.back().children;
    new_children.reserve(old_node.children.size());
    for (unsigned old_child_index : old_node.children) {
        new_children.push_back(
            collect_subtree(old_tree, new_tree, old_child_index)
        );
    }
    return new_index;
}

/// @brief Prunes the search tree to a new root node.
/// @param tree Original search tree.
/// @param new_root_index Index of the new root node.
/// @return Pruned search tree.
std::vector<MCTSNode>
prune_search_tree(const std::vector<MCTSNode> &tree, int new_root_index) {
    std::vector<MCTSNode> new_tree;
    new_tree.reserve(tree.size());
    collect_subtree(tree, new_tree, new_root_index);
    return new_tree;
}

} // namespace

void othello::MCTS::apply_action(int action) {
    MCTSNode &root = _search_tree.front();

    std::vector<int> legal_actions = root.position.legal_actions();
    if (std::find(legal_actions.begin(), legal_actions.end(), action) ==
        legal_actions.end()) {
        return;
    }

    if (root.children.empty()) {
        // The root node has not been expanded yet.
        Position next_position = root.position.apply_action(action);
        _search_tree.clear();
        _search_tree.emplace_back(next_position, action);
        return;
    }

    unsigned child_index = *std::find_if(
        root.children.begin(),
        root.children.end(),
        [this, action](unsigned child_index) {
            return _search_tree[child_index].previous_action == action;
        }
    );
    _search_tree = ::prune_search_tree(_search_tree, child_index);
}

void othello::MCTS::set_num_simulations(int value) noexcept {
    value = std::max(1, value);
    _num_simulations = value;
}

void othello::MCTS::set_batch_size(int value) noexcept {
    value = std::max(1, value);
    _batch_size = value;
}

void othello::MCTS::set_num_threads(int value) noexcept {
    value = std::max(1, value);
    _num_threads = value;
}

void othello::MCTS::set_exploration_weight(float value) noexcept {
    if (!std::isfinite(value)) {
        value = 0.0f;
    } else {
        value = std::max(0.0f, value);
    }
    _exploration_weight = value;
}

void othello::MCTS::set_dirichlet_epsilon(float value) noexcept {
    if (!std::isfinite(value)) {
        value = 0.0f;
    } else {
        value = std::max(0.0f, value);
        value = std::min(1.0f, value);
    }
    _dirichlet_epsilon = value;
}

void othello::MCTS::set_dirichlet_alpha(float value) noexcept {
    if (!std::isfinite(value)) {
        value = 0.0f;
    } else {
        value = std::max(0.0f, value);
    }
    _dirichlet_alpha = value;
}
