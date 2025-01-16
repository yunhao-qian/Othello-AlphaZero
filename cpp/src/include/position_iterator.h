/// @file position_iterator.h
/// @brief Position iterator concept and implementation.

#ifndef OTHELLO_MCTS_POSITION_ITERATOR_H
#define OTHELLO_MCTS_POSITION_ITERATOR_H

#include <concepts>
#include <cstddef>
#include <iterator>

#include "search_node.h"

namespace othello {

/// @brief Concept for an iterator over the history positions.
///
template <typename T>
concept PositionIterator = std::input_iterator<T> && requires(T it) {
    { *it } -> std::convertible_to<const Position &>;
};

/// @brief Iterator over the history positions of a search node.
///
class SearchNodePositionIterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = Position;

    /// @brief Constructs a history position iterator.
    /// @param node Node to start from.
    SearchNodePositionIterator(othello::SearchNode *node) noexcept
        : _node(node) {}

    /// @brief Equality comparison.
    /// @param other Another iterator.
    /// @return True if the iterators are equal, false otherwise.
    bool operator==(const SearchNodePositionIterator &other) const noexcept {
        return _node == other._node;
    }

    /// @brief Dereference operator.
    /// @return Reference to the current position.
    othello::Position &operator*() const noexcept {
        return _node->position;
    }

    /// @brief Pre-increment operator moving to the previous position.
    /// @return Reference to the iterator.
    SearchNodePositionIterator &operator++() noexcept {
        _node = _node->parent;
        return *this;
    }

    /// @brief Post-increment operator moving to the previous position.
    /// @param Dummy integer to distinguish from the pre-increment operator.
    /// @return Position before the increment.
    othello::Position operator++(int) noexcept {
        othello::Position position = _node->position;
        _node = _node->parent;
        return position;
    }

    /// @brief Gets the past-the-end iterator.
    /// @return Past-the-end iterator.
    static SearchNodePositionIterator end() noexcept {
        return SearchNodePositionIterator(nullptr);
    }

private:
    othello::SearchNode *_node;
};

static_assert(PositionIterator<SearchNodePositionIterator>);

} // namespace othello

#endif // OTHELLO_MCTS_POSITION_ITERATOR_H
