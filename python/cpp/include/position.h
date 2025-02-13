/**
 * @file position.h
 * @brief Othello game logic.
 */

#ifndef OTHELLO_ALPHAZERO_POSITION_H
#define OTHELLO_ALPHAZERO_POSITION_H

#include <array>
#include <bit>
#include <cstdint>
#include <ranges>
#include <string>
#include <tuple>

namespace othello {

/**
 * @brief Position in an Othello game.
 */
class Position {
public:
    static constexpr Position initial_position() noexcept;

    /**
     * @brief Gets the current player.
     * @return 1 if Black, 2 if White, 0 if the position is terminal.
     */
    constexpr int player() const noexcept {
        return m_player;
    }

    /**
     * @brief Queries whether the position is terminal.
     * @return True if the position is terminal, or false if otherwise.
     */
    constexpr bool is_terminal() const noexcept {
        return m_player == 0;
    }

    /**
     * @brief Gets the discs of the Black player.
     * @return Bitboard of black discs.
     */
    constexpr std::uint64_t player1_discs() const noexcept {
        return m_player1_discs;
    }

    /**
     * @brief Gets the discs of the White player.
     * @return Bitboard of white discs.
     */
    constexpr std::uint64_t player2_discs() const noexcept {
        return m_player2_discs;
    }

    /**
     * @brief Gets the legal moves of the current player.
     * @return Bitboard of legal moves.
     */
    constexpr std::uint64_t legal_moves() const noexcept {
        return m_legal_moves;
    }

    /**
     * @brief Gets the legal actions of the current player.
     * @param output Output iterator of `int` to write the legal actions to.
     * @return One past the last written element.
     */
    template <typename OutputIt>
    constexpr OutputIt legal_actions(OutputIt output) const;

    /**
     * @brief Applies a move to the current position.
     * @param move_mask Bit mask of the move.
     * @return New position.
     */
    constexpr Position apply_move(std::uint64_t move_mask) const noexcept;

    /**
     * @brief Applies a pass to the current position.
     * @return New position.
     */
    constexpr Position apply_pass() const noexcept;

    /**
     * @brief Applies an action to the current position.
     * @param action Action to apply.
     * @return New position.
     */
    constexpr Position apply_action(int action) const noexcept;

    /**
     * @brief Gets the string representation of the position.
     * @return UFT-8 string.
     */
    constexpr std::string to_string() const;

private:
    constexpr Position(
        const int player,
        const std::uint64_t player1_discs,
        const std::uint64_t player2_discs,
        const std::uint64_t legal_moves,
        const std::uint64_t next_legal_moves
    ) noexcept
        : m_player(player),
          m_player1_discs(player1_discs),
          m_player2_discs(player2_discs),
          m_legal_moves(legal_moves),
          m_next_legal_moves(next_legal_moves) {}

    int m_player;
    std::uint64_t m_player1_discs;
    std::uint64_t m_player2_discs;
    std::uint64_t m_legal_moves;
    std::uint64_t m_next_legal_moves;
};

namespace internal {

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

template <int Direction>
constexpr std::uint64_t shift(const std::uint64_t mask) noexcept {
    constexpr int stride = STRIDES[Direction];
    if constexpr (stride > 0) {
        return mask >> stride;
    } else if constexpr (stride < 0) {
        return mask << -stride;
    } else {
        return mask;
    }
}

template <int Direction>
constexpr std::uint64_t get_potential_flips_in_direction(
    const std::uint64_t player_discs, std::uint64_t opponent_discs
) noexcept {
    opponent_discs &= MASKS[Direction];
    std::uint64_t flips = opponent_discs & shift<Direction>(player_discs);
    for ([[maybe_unused]] const int i : std::views::iota(0, 5)) {
        flips |= opponent_discs & shift<Direction>(flips);
    }
    return flips;
}

constexpr std::uint64_t get_legal_moves(
    const std::uint64_t player_discs, const std::uint64_t opponent_discs
) noexcept {
    std::uint64_t legal_moves = 0;

#define OTHELLO_ALPHAZERO_CHECK_DIRECTION(Direction)                                   \
    do {                                                                               \
        const std::uint64_t potential_flips =                                          \
            get_potential_flips_in_direction<Direction>(player_discs, opponent_discs); \
        legal_moves |= shift<Direction>(potential_flips);                              \
    } while (false)

    OTHELLO_ALPHAZERO_CHECK_DIRECTION(0);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(1);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(2);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(3);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(4);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(5);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(6);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(7);

#undef OTHELLO_ALPHAZERO_CHECK_DIRECTION

    legal_moves &= ~(player_discs | opponent_discs);
    return legal_moves;
}

constexpr std::uint64_t get_flips(
    const std::uint64_t move_mask,
    const std::uint64_t player_discs,
    const std::uint64_t opponent_discs
) noexcept {
    std::uint64_t flips = 0;

#define OTHELLO_ALPHAZERO_CHECK_DIRECTION(Direction)                                \
    do {                                                                            \
        const std::uint64_t potential_flips =                                       \
            get_potential_flips_in_direction<Direction>(move_mask, opponent_discs); \
        if ((shift<Direction>(potential_flips) & player_discs) != 0) {              \
            flips |= potential_flips;                                               \
        }                                                                           \
    } while (false)

    OTHELLO_ALPHAZERO_CHECK_DIRECTION(0);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(1);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(2);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(3);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(4);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(5);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(6);
    OTHELLO_ALPHAZERO_CHECK_DIRECTION(7);

#undef OTHELLO_ALPHAZERO_CHECK_DIRECTION

    return flips;
}

}  // namespace internal

}  // namespace othello

constexpr othello::Position othello::Position::initial_position() noexcept {
    constexpr std::uint64_t player1_discs =
        0b00000000'00000000'00000000'00001000'00010000'00000000'00000000'00000000;
    constexpr std::uint64_t player2_discs =
        0b00000000'00000000'00000000'00010000'00001000'00000000'00000000'00000000;
    constexpr std::uint64_t legal_moves = internal::get_legal_moves(player1_discs, player2_discs);
    return Position(1, player1_discs, player2_discs, legal_moves, 0);
}

template <typename OutputIt>
constexpr OutputIt othello::Position::legal_actions(OutputIt output) const {
    if (is_terminal()) {
        return output;
    }
    if (m_legal_moves == 0) {
        *output = 64;
        ++output;
        return output;
    }
    for (std::uint64_t move_mask = std::uint64_t(1) << 63;
         const int action : std::views::iota(0, 64)) {
        if ((move_mask & m_legal_moves) != 0) {
            *output = action;
            ++output;
        }
        move_mask >>= 1;
    }
    return output;
}

constexpr othello::Position othello::Position::apply_move(std::uint64_t move_mask) const noexcept {
    int player = 3 - m_player;
    std::uint64_t player1_discs = m_player1_discs;
    std::uint64_t player2_discs = m_player2_discs;

    auto [player_discs, opponent_discs] = m_player == 1 ? std::tie(player1_discs, player2_discs)
                                                        : std::tie(player2_discs, player1_discs);

    {
        const std::uint64_t flips = internal::get_flips(move_mask, player_discs, opponent_discs);
        player_discs |= move_mask | flips;
        opponent_discs &= ~flips;
    }

    const std::uint64_t legal_moves = internal::get_legal_moves(opponent_discs, player_discs);

    std::uint64_t next_legal_moves;
    if (legal_moves != 0) {
        next_legal_moves = 0;
    } else {
        // The next player has no legal moves. If the current player has no legal moves either, the
        // game is over.
        next_legal_moves = internal::get_legal_moves(player_discs, opponent_discs);
        if (next_legal_moves == 0) {
            player = 0;
        }
    }

    return Position(player, player1_discs, player2_discs, legal_moves, next_legal_moves);
}

constexpr othello::Position othello::Position::apply_pass() const noexcept {
    return Position(3 - m_player, m_player1_discs, m_player2_discs, m_next_legal_moves, 0);
}

constexpr othello::Position othello::Position::apply_action(const int action) const noexcept {
    if (action == 64) {
        return apply_pass();
    }
    return apply_move(std::uint64_t(1) << (63 - action));
}

constexpr std::string othello::Position::to_string() const {
    std::string result;
    // Black Circle and White Circle are 3 bytes long.
    // Multiplication Sign and Middle Dot are 2 bytes long.
    result.reserve(18 + 8 * 26 - 1 + std::popcount(m_player1_discs | m_player2_discs));

    result.append("  a b c d e f g h\n");

    for (std::uint64_t square_mask = std::uint64_t(1) << 63;
         const int row : std::views::iota(0, 8)) {
        result.push_back(static_cast<char>('1' + row));
        for ([[maybe_unused]] const int col : std::views::iota(0, 8)) {
            result.push_back(' ');
            const char *square;
            if (square_mask & m_player1_discs) {
                square = "\u25cf";  // Black Circle
            } else if (square_mask & m_player2_discs) {
                square = "\u25cb";  // White Circle
            } else if (square_mask & m_legal_moves) {
                square = "\u00d7";  // Multiplication Sign
            } else {
                square = "\u00b7";  // Middle Dot
            }
            result.append(square);
            square_mask >>= 1;
        }
        if (row < 7) {
            result.push_back('\n');
        }
    }

    return result;
}

#endif  // OTHELLO_ALPHAZERO_POSITION_H
