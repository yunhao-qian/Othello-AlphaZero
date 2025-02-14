/**
 * @file search_tree_base.cpp
 * @brief Implementation of the search tree base class.
 */

#include "search_tree_base.h"

#include <format>
#include <stdexcept>
#include <utility>

othello::SearchTreeBase::SearchTreeBase(const float c_puct_init, const float c_puct_base) {
    set_c_puct_init(c_puct_init);
    set_c_puct_base(c_puct_base);
    reset_position();
}

void othello::SearchTreeBase::set_c_puct_init(const float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::format("Expected c_puct_init to be finite, but got {}.", value)
        );
    }
    if (!(value >= 0.f)) {
        throw std::invalid_argument(std::format("Expected c_puct_init >= 0, but got {}.", value));
    }
    m_c_puct_init = value;
}

void othello::SearchTreeBase::set_c_puct_base(const float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::format("Expected c_puct_base to be finite, but got {}.", value)
        );
    }
    if (!(value > 0.f)) {
        throw std::invalid_argument(std::format("Expected c_puct_base > 0, but got {}.", value));
    }
    m_c_puct_base = value;
}

void othello::SearchTreeBase::reset_position() {
    m_nodes.clear();
    m_nodes.push_back(
        SearchNode{.position = INITIAL_POSITION, .parent_index = 0, .prior_probability = 1.f}
    );
    m_root_index = 0;
}

void othello::SearchTreeBase::apply_action(const int action) {
    SearchNode &root = m_nodes[m_root_index];

    if (root.child_indices.empty()) {
        // The root node has not been expanded yet.
        m_nodes.push_back(SearchNode{
            .position = root.position.apply_action(action),
            .parent_index = m_root_index,
            .prior_probability = 1.f
        });
        ++m_root_index;
        return;
    }

    std::uint32_t child_index;
    if (action == 0 || root.child_indices.size() == 1) {
        child_index = root.child_indices.front();
    } else {
        // If action == 0, an uint64_t << 64 is undefined behavior. This case is handled separately.
        const std::uint64_t previous_moves_mask = ~std::uint64_t(0) << (64 - action);
        child_index =
            root.child_indices[std::popcount(root.position.legal_moves() & previous_moves_mask)];
    }
    root.child_indices.clear();
    root.child_indices.push_back(child_index);

    std::vector<SearchNode> new_nodes;
    // The vector is never reallocated when cloning the nodes, so all references to the nodes remain
    // valid.
    new_nodes.reserve(m_nodes.capacity());
    clone_nodes(m_nodes.front(), 0, new_nodes);
    m_nodes = std::move(new_nodes);

    ++m_root_index;
}

void othello::SearchTreeBase::propagate_virtual_loss(const std::uint32_t leaf_index) noexcept {
    const SearchNode *const root = &m_nodes[m_root_index];
    for (SearchNode *node = &m_nodes[leaf_index]; node != root;
         node = &m_nodes[node->parent_index]) {
        ++node->visit_count;
        node->total_action_value -= 1.f;
        node->mean_action_value = node->total_action_value / node->visit_count;
    }
}

std::uint32_t othello::SearchTreeBase::clone_nodes(
    const SearchNode &old_node,
    const std::uint32_t new_parent_index,
    std::vector<SearchNode> &new_nodes
) const {
    const std::uint32_t new_node_index = static_cast<std::uint32_t>(new_nodes.size());
    new_nodes.push_back(SearchNode{
        .position = old_node.position,
        .parent_index = new_parent_index,
        .visit_count = old_node.visit_count,
        .total_action_value = old_node.total_action_value,
        .mean_action_value = old_node.mean_action_value,
        .prior_probability = old_node.prior_probability
    });
    auto &new_child_indices = new_nodes.back().child_indices;
    new_child_indices.reserve(old_node.child_indices.size());
    for (const auto old_child_index : old_node.child_indices) {
        new_child_indices.push_back(clone_nodes(m_nodes[old_child_index], new_node_index, new_nodes)
        );
    }
    return new_node_index;
}
