/**
 * @file search_node.h
 * @brief Search node structure.
 */

#ifndef OTHELLO_ALPHAZERO_SEARCH_NODE_H
#define OTHELLO_ALPHAZERO_SEARCH_NODE_H

#include <array>
#include <cstdint>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "position.h"

namespace othello {

/**
 * @brief Node in a search tree.
 */
struct SearchNode {
    /**
     * @brief Game position.
     */
    Position position;

    /**
     * @brief Index of the parent node.
     * @details The parent node may correspond to a history position that is no longer in the search
     *     tree. If the current node corresponds to the initial position, the parent index is 0.
     */
    std::uint32_t parent_index;

    /**
     * @brief Indices of the child nodes.
     * @note The maximum number of legal actions is 33 according to this paper:
     *     https://jxiv.jst.go.jp/index.php/jxiv/preprint/view/480/1498.
     */
    boost::container::static_vector<std::uint32_t, 33> child_indices = {};

    /**
     * @brief Visit count of the preceding edge.
     */
    std::uint32_t visit_count = 0;

    /**
     * @brief Total action-value of the preceding edge.
     */
    float total_action_value = 0.f;

    /**
     * @brief Mean action-value of the preceding edge.
     */
    float mean_action_value = 0.f;

    /**
     * @brief Prior probability of the preceding edge.
     */
    float prior_probability;
};

}  // namespace othello

#endif  // OTHELLO_ALPHAZERO_SEARCH_NODE_H
