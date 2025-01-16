/// @file transformation.h
/// @brief Position transformation functions.

#ifndef OTHELLO_MCTS_TRANSFORMATION_H
#define OTHELLO_MCTS_TRANSFORMATION_H

#include <algorithm>
#include <array>
#include <cstdint>

#include "position.h"
#include "position_iterator.h"

namespace othello {

/// @brief Transforms an action.
/// @param action Action to transform (0-63 for moves and 64 for pass).
/// @param transformation Transformation to apply (0-7).
/// @return Transformed action.
constexpr int transform_action(int action, int transformation) noexcept;

/// @brief Converts a sequence of positions to features.
/// @param first Iterator to the current position.
/// @param last Past-the-end iterator beyond the earliest position.
/// @param features Feature data of shape `(feature_channels, 8, 8)` to write
///     to, where `feature_channels = 1 + history_size * 2`.
/// @param history_size Number of history positions to include.
/// @param transformation Transformation to apply (0-7).
template <PositionIterator Iterator>
void positions_to_features(
    Iterator first,
    Iterator last,
    float *features,
    int history_size,
    int transformation
);

namespace internal {

constexpr int transform_action_impl(int action, int transformation) noexcept {
    if (action == 64) {
        return action;
    }
    int row = action / 8;
    int col = action % 8;
    bool horizontal_flip = transformation % 2 == 1;
    if (horizontal_flip) {
        col = 7 - col;
    }
    int num_rotations = transformation / 2;
    for (int i = 0; i < num_rotations; ++i) {
        int old_row = row;
        row = col;
        col = 7 - old_row;
    }
    return row * 8 + col;
}

constexpr std::array<std::array<int, 65>, 8>
create_transformed_actions() noexcept {
    std::array<std::array<int, 65>, 8> transformed_actions;
    for (int transformation = 0; transformation < 8; ++transformation) {
        for (int action = 0; action < 65; ++action) {
            transformed_actions[transformation][action] =
                transform_action_impl(action, transformation);
        }
    }
    return transformed_actions;
}

constexpr std::array<std::array<int, 65>, 8> TRANSFORMED_ACTIONS =
    create_transformed_actions();

} // namespace internal

} // namespace othello

constexpr int
othello::transform_action(int action, int transformation) noexcept {
    return internal::TRANSFORMED_ACTIONS[transformation][action];
}

template <othello::PositionIterator Iterator>
void othello::positions_to_features(
    Iterator first,
    Iterator last,
    float *features,
    int history_size,
    int transformation
) {
    std::fill_n(features, 64, (*first).player() - 1.0f);
    features += 64;
    const Position *position = nullptr;
    for (int i = 0; i < history_size; ++i) {
        if (!(first == last)) {
            position = &*first;
            ++first;
        }
        std::uint64_t player1_discs = position->player1_discs();
        std::uint64_t player2_discs = position->player2_discs();
        std::uint64_t square_mask = std::uint64_t(1) << 63;
        for (int original_index = 0; original_index < 64; ++original_index) {
            int transformed_index =
                transform_action(original_index, transformation);
            features[transformed_index] = (player1_discs & square_mask) != 0;
            features[64 + transformed_index] =
                (player2_discs & square_mask) != 0;
            square_mask >>= 1;
        }
        features += 128;
    }
}

#endif // OTHELLO_MCTS_TRANSFORMATION_H
