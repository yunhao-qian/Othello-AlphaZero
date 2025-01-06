/// @file position.cpp
/// @brief Implementation of the Othello game logic.

#include "position.h"

#include <algorithm>
#include <array>

#include "utility.h"

namespace {

/// @brief Left shifts the mask if the shift is negative, right shifts the mask
///     if otherwise.
std::uint64_t shift(std::uint64_t mask, int shift) noexcept {
    if (shift < 0) {
        return mask << -shift;
    }
    return mask >> shift;
}

/// @brief Returns the potential flips in a given direction.
/// @param player_discs Discs of the current player to consider.
/// @param masked_opponent_discs Discs of the opponent player that may be
///     flipped in the given direction.
/// @param stride Stride of the direction in the flat representation.
std::uint64_t get_potential_flips_in_direction(
    std::uint64_t player_discs, std::uint64_t masked_opponent_discs, int stride
) noexcept {
    std::uint64_t flips = masked_opponent_discs & ::shift(player_discs, stride);
    for (int i = 0; i < 5; ++i) {
        flips |= masked_opponent_discs & ::shift(flips, stride);
    }
    return flips;
}

constexpr std::array<int, 8> STRIDES = {-9, -8, -7, -1, 1, 7, 8, 9};

constexpr std::uint64_t MASK_NO_LEFT_RIGHT =
    0b01111110'01111110'01111110'01111110'01111110'01111110'01111110'01111110;

constexpr std::uint64_t MASK_NO_TOP_BOTTOM =
    0b00000000'11111111'11111111'11111111'11111111'11111111'11111111'00000000;

constexpr std::uint64_t MASK_NO_EDGES = MASK_NO_LEFT_RIGHT & MASK_NO_TOP_BOTTOM;

constexpr std::array<std::uint64_t, 8> MASKS = {
    MASK_NO_EDGES,
    MASK_NO_TOP_BOTTOM,
    MASK_NO_EDGES,
    MASK_NO_LEFT_RIGHT,
    MASK_NO_LEFT_RIGHT,
    MASK_NO_EDGES,
    MASK_NO_TOP_BOTTOM,
    MASK_NO_EDGES
};

} // namespace

othello::Position othello::Position::initial_position() noexcept {
    std::uint64_t p1_discs =
        0b00000000'00000000'00000000'00001000'00010000'00000000'00000000'00000000;
    std::uint64_t p2_discs =
        0b00000000'00000000'00000000'00010000'00001000'00000000'00000000'00000000;
    std::uint64_t legal_moves = get_legal_moves(p1_discs, p2_discs);
    return Position(1, p1_discs, p2_discs, legal_moves, 0);
}

std::vector<int> othello::Position::legal_actions() const {
    std::vector<int> actions;
    if (_legal_moves == 0) {
        actions.push_back(64);
    } else {
        actions.reserve(popcount(_legal_moves));
        std::uint64_t move_mask = 1ULL << 63;
        for (int action = 0; action < 64; ++action) {
            if ((move_mask & _legal_moves) != 0) {
                actions.push_back(action);
            }
            move_mask >>= 1;
        }
    }
    return actions;
}

othello::Position othello::Position::apply_action(int action) const noexcept {
    int player = 3 - _player;
    std::uint64_t p1_discs = _p1_discs;
    std::uint64_t p2_discs = _p2_discs;
    std::uint64_t legal_moves;
    std::uint64_t next_legal_moves = 0;

    if (action == 64) {
        // The action is a pass. The next legal moves have already been
        // computed.
        legal_moves = _next_legal_moves;
    } else {
        std::uint64_t *player_discs;
        std::uint64_t *opponent_discs;
        if (_player == 1) {
            player_discs = &p1_discs;
            opponent_discs = &p2_discs;
        } else {
            player_discs = &p2_discs;
            opponent_discs = &p1_discs;
        }

        std::uint64_t move_mask = 1ULL << (63 - action);
        std::uint64_t flips =
            get_flips(move_mask, *player_discs, *opponent_discs);
        *player_discs |= move_mask | flips;
        *opponent_discs &= ~flips;
        legal_moves = get_legal_moves(*opponent_discs, *player_discs);

        if (legal_moves == 0) {
            // The next player has no legal moves. If the current player has no
            // legal moves either, the game is over.
            next_legal_moves = get_legal_moves(*player_discs, *opponent_discs);
            if (next_legal_moves == 0) {
                player = 0;
            }
        }
    }

    return Position(player, p1_discs, p2_discs, legal_moves, next_legal_moves);
}

float othello::Position::action_value() const noexcept {
    int p1_num_discs = popcount(_p1_discs);
    int p2_num_discs = popcount(_p2_discs);
    if (p1_num_discs > p2_num_discs) {
        return 1.0f;
    }
    if (p1_num_discs < p2_num_discs) {
        return -1.0f;
    }
    return 0.0f;
}

std::vector<float> othello::Position::to_features() const {
    std::vector<float> features(3 * 64, 0.0f);
    std::uint64_t square_mask = 1ULL << 63;
    for (int i = 0; i < 64; ++i) {
        if ((_p1_discs & square_mask) != 0) {
            features[i] = 1.0f;
        }
        if ((_p2_discs & square_mask) != 0) {
            features[64 + i] = 1.0f;
        }
        square_mask >>= 1;
    }
    if (_player != 1) {
        std::fill(features.begin() + 2 * 64, features.end(), 1.0f);
    }
    return features;
}

int othello::Position::operator()(int row, int col) const noexcept {
    if (!(0 <= row && row < 8 && 0 <= col && col < 8)) {
        return -1;
    }
    int index = row * 8 + col;
    std::uint64_t square_mask = 1ULL << (63 - index);
    if ((_p1_discs & square_mask) != 0) {
        return 1;
    }
    if ((_p2_discs & square_mask) != 0) {
        return 2;
    }
    return 0;
}

bool othello::Position::is_legal_move(int row, int col) const noexcept {
    int index = row * 8 + col;
    std::uint64_t move_mask = 1ULL << (63 - index);
    return (_legal_moves & move_mask) != 0;
}

std::string othello::Position::to_string() const {
    std::string result;
    result.reserve(9 * 18);
    result.append("  a b c d e f g h\n");
    for (int row = 0; row < 8; ++row) {
        result.push_back(static_cast<char>('1' + row));
        for (int col = 0; col < 8; ++col) {
            result.push_back(' ');
            const char *square;
            switch ((*this)(row, col)) {
            case 1:
                square = "\u25cf"; // black circle
                break;
            case 2:
                square = "\u25cb"; // white circle
                break;
            default:
                if (is_legal_move(row, col)) {
                    square = "\u25cc"; // dotted circle
                } else {
                    square = "\u00b7"; // middle dot
                }
            }
            result.append(square);
        }
        result.push_back('\n');
    }
    // Remove the last newline character.
    result.pop_back();
    return result;
}

std::uint64_t othello::get_legal_moves(
    std::uint64_t player_discs, std::uint64_t opponent_discs
) noexcept {
    std::uint64_t legal_moves = 0;
    for (int i = 0; i < 8; ++i) {
        std::uint64_t potential_flips = ::get_potential_flips_in_direction(
            player_discs, opponent_discs & MASKS[i], STRIDES[i]
        );
        legal_moves |= ::shift(potential_flips, STRIDES[i]);
    }
    legal_moves &= ~(player_discs | opponent_discs);
    return legal_moves;
}

std::uint64_t othello::get_flips(
    std::uint64_t move_mask,
    std::uint64_t player_discs,
    std::uint64_t opponent_discs
) noexcept {
    std::uint64_t flips = 0;
    for (int i = 0; i < 8; ++i) {
        std::uint64_t potential_flips = ::get_potential_flips_in_direction(
            move_mask, opponent_discs & MASKS[i], STRIDES[i]
        );
        if ((::shift(potential_flips, STRIDES[i]) & player_discs) != 0) {
            flips |= potential_flips;
        }
    }
    return flips;
}
