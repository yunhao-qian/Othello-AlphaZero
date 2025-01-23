/// @file search_node.h
/// @brief Search node structure.

#ifndef OTHELLO_MCTS_SEARCH_NODE_H
#define OTHELLO_MCTS_SEARCH_NODE_H

#include <memory>
#include <vector>

#include "position.h"

namespace othello {

/// @brief A node in the search tree.
///
struct SearchNode {
    /// @brief Game position.
    ///
    Position position;

    /// @brief Parent node.
    /// @details Unless the node corresponds to the initial position, this
    ///     pointer should always point to the node of the previous position,
    ///     even if the node is the root or is no longer in the search tree.
    SearchNode *parent = nullptr;

    /// @brief Child nodes.
    ///
    std::vector<std::unique_ptr<SearchNode>> children = {};

    /// @brief Visit count of the preceding edge.
    ///
    int visit_count = 0;

    /// @brief Total action-value of the preceding edge.
    ///
    float total_action_value = 0.0f;

    /// @brief Mean action-value of the preceding edge.
    ///
    float mean_action_value = 0.0f;

    /// @brief Prior probability of the preceding edge.
    ///
    float prior_probability = 1.0f;
};

} // namespace othello

#endif // OTHELLO_MCTS_SEARCH_NODE_H
