/// @file position.h
/// @brief Othello game logic.

#ifndef OTHELLO_MCTS_POSITION_H
#define OTHELLO_MCTS_POSITION_H

#include <array>
#include <bit>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace othello {

/// @brief Gets the legal moves of the current player.
/// @param player_discs Discs of the current player.
/// @param opponent_discs Discs of the opponent player.
/// @return Bitboard of legal moves.
constexpr std::uint64_t
get_legal_moves(std::uint64_t player_discs, std::uint64_t opponent_discs) noexcept;

/// @brief Gets the discs flipped by a move.
/// @param move_mask Bit mask of the move.
/// @param player_discs Discs of the current player.
/// @param opponent_discs Discs of the opponent player.
/// @return Bitboard of flipped discs.
constexpr std::uint64_t get_flips(
    std::uint64_t move_mask, std::uint64_t player_discs, std::uint64_t opponent_discs
) noexcept;

/// @brief Position of an Othello game.
///
class Position {
public:
    /// @brief Gets the initial position of a game.
    /// @return Initial position.
    static constexpr Position initial_position() noexcept;

    /// @brief Gets the current player.
    /// @return 1 if Black, 2 if White, 0 if the position is terminal.
    constexpr int player() const noexcept {
        return this->_player;
    }

    /// @brief Gets the discs of the Black player.
    /// @return Bitboard of black discs.
    constexpr std::uint64_t player1_discs() const noexcept {
        return this->_player1_discs;
    }

    /// @brief Gets the discs of the White player.
    /// @return Bitboard of white discs.
    constexpr std::uint64_t player2_discs() const noexcept {
        return this->_player2_discs;
    }

    /// @brief Gets the status of a square.
    /// @param index Flat index of the square.
    /// @return 1 if black, 2 if white, 0 if empty.
    constexpr int operator[](int index) const noexcept;

    /// @brief Checked version of Position::operator[].
    ///
    constexpr int at(int index) const;

    /// @brief Gets the legal moves of the current player.
    /// @return Bitboard of legal moves.
    constexpr std::uint64_t legal_moves() const noexcept {
        return this->_legal_moves;
    }

    /// @brief Queries whether a move is legal.
    /// @param index Flat index of the move.
    /// @return True if the move is legal, false otherwise.
    constexpr bool is_legal_move(int index) const noexcept;

    /// @brief Checked version of Position::is_legal_move.
    ///
    constexpr bool is_legal_move_checked(int index) const;

    /// @brief Gets the legal actions of the current player.
    /// @return Vector of legal actions.
    constexpr std::vector<int> legal_actions() const;

    /// @brief Applies a move to the current position and returns the new position.
    /// @param move_mask Bit mask of the move.
    /// @return New position.
    constexpr Position apply_move(std::uint64_t move_mask) const noexcept;

    /// @brief Checked version of Position::apply_move.
    ///
    constexpr Position apply_move_checked(std::uint64_t move_mask) const;

    /// @brief Applies a pass to the current position and returns the new position.
    /// @return New position.
    constexpr Position apply_pass() const noexcept;

    /// @brief Checked version of Position::apply_pass.
    ///
    constexpr Position apply_pass_checked() const;

    /// @brief Applies an action to the current position and returns the new position.
    /// @param action Action to apply.
    /// @return New position.
    constexpr Position apply_action(int action) const noexcept;

    /// @brief Checked version of Position::apply_action.
    ///
    constexpr Position apply_action_checked(int action) const;

    /// @brief Queries whether the position is terminal.
    /// @return True if the position is terminal, false otherwise.
    constexpr bool is_terminal() const noexcept {
        return this->_player == 0;
    }

    /// @brief Gets the string representation of the position.
    /// @return String representation.
    constexpr std::string to_string() const;

private:
    constexpr Position(
        int player,
        std::uint64_t player1_discs,
        std::uint64_t player2_discs,
        std::uint64_t legal_moves,
        std::uint64_t next_legal_moves
    ) noexcept
        : _player(player),
          _player1_discs(player1_discs),
          _player2_discs(player2_discs),
          _legal_moves(legal_moves),
          _next_legal_moves(next_legal_moves) {}

    int _player;
    std::uint64_t _player1_discs;
    std::uint64_t _player2_discs;
    std::uint64_t _legal_moves;
    std::uint64_t _next_legal_moves;
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
    for (const int i : std::views::iota(0, 5)) {
        flips |= opponent_discs & shift<Direction>(flips);
    }
    return flips;
}

} // namespace internal

} // namespace othello

constexpr std::uint64_t othello::get_legal_moves(
    const std::uint64_t player_discs, const std::uint64_t opponent_discs
) noexcept {
    std::uint64_t legal_moves = 0;

#define OTHELLO_MCTS_CHECK_DIRECTION(Direction)                                                    \
    do {                                                                                           \
        const std::uint64_t potential_flips =                                                      \
            internal::get_potential_flips_in_direction<Direction>(player_discs, opponent_discs);   \
        legal_moves |= internal::shift<Direction>(potential_flips);                                \
    } while (false)

    OTHELLO_MCTS_CHECK_DIRECTION(0);
    OTHELLO_MCTS_CHECK_DIRECTION(1);
    OTHELLO_MCTS_CHECK_DIRECTION(2);
    OTHELLO_MCTS_CHECK_DIRECTION(3);
    OTHELLO_MCTS_CHECK_DIRECTION(4);
    OTHELLO_MCTS_CHECK_DIRECTION(5);
    OTHELLO_MCTS_CHECK_DIRECTION(6);
    OTHELLO_MCTS_CHECK_DIRECTION(7);

#undef OTHELLO_MCTS_CHECK_DIRECTION

    legal_moves &= ~(player_discs | opponent_discs);
    return legal_moves;
}

constexpr std::uint64_t othello::get_flips(
    const std::uint64_t move_mask,
    const std::uint64_t player_discs,
    const std::uint64_t opponent_discs
) noexcept {
    std::uint64_t flips = 0;

#define OTHELLO_MCTS_CHECK_DIRECTION(Direction)                                                    \
    do {                                                                                           \
        const std::uint64_t potential_flips =                                                      \
            internal::get_potential_flips_in_direction<Direction>(move_mask, opponent_discs);      \
        if ((internal::shift<Direction>(potential_flips) & player_discs) != 0) {                   \
            flips |= potential_flips;                                                              \
        }                                                                                          \
    } while (false)

    OTHELLO_MCTS_CHECK_DIRECTION(0);
    OTHELLO_MCTS_CHECK_DIRECTION(1);
    OTHELLO_MCTS_CHECK_DIRECTION(2);
    OTHELLO_MCTS_CHECK_DIRECTION(3);
    OTHELLO_MCTS_CHECK_DIRECTION(4);
    OTHELLO_MCTS_CHECK_DIRECTION(5);
    OTHELLO_MCTS_CHECK_DIRECTION(6);
    OTHELLO_MCTS_CHECK_DIRECTION(7);

#undef OTHELLO_MCTS_CHECK_DIRECTION

    return flips;
}

constexpr othello::Position othello::Position::initial_position() noexcept {
    constexpr std::uint64_t player1_discs =
        0b00000000'00000000'00000000'00001000'00010000'00000000'00000000'00000000;
    constexpr std::uint64_t player2_discs =
        0b00000000'00000000'00000000'00010000'00001000'00000000'00000000'00000000;
    constexpr std::uint64_t legal_moves = get_legal_moves(player1_discs, player2_discs);
    return Position(1, player1_discs, player2_discs, legal_moves, 0);
}

constexpr int othello::Position::operator[](const int index) const noexcept {
    const std::uint64_t square_mask = std::uint64_t(1) << (63 - index);
    if ((this->_player1_discs & square_mask) != 0) {
        return 1;
    }
    if ((this->_player2_discs & square_mask) != 0) {
        return 2;
    }
    return 0;
}

constexpr int othello::Position::at(const int index) const {
    if (!(0 <= index && index < 64)) {
        throw std::out_of_range(std::format("Expected 0 <= index < 64, but got {}.", index));
    }
    return (*this)[index];
}

constexpr bool othello::Position::is_legal_move(const int index) const noexcept {
    const std::uint64_t move_mask = std::uint64_t(1) << (63 - index);
    return (this->_legal_moves & move_mask) != 0;
}

constexpr bool othello::Position::is_legal_move_checked(const int index) const {
    if (!(0 <= index && index < 64)) {
        throw std::out_of_range(std::format("Expected 0 <= index < 64, but got {}.", index));
    }
    return this->is_legal_move(index);
}

constexpr std::vector<int> othello::Position::legal_actions() const {
    if (this->is_terminal()) {
        return {};
    }
    if (this->_legal_moves == 0) {
        return {64};
    }

    std::vector<int> actions;
    actions.reserve(static_cast<std::size_t>(std::popcount(this->_legal_moves)));

    for (std::uint64_t move_mask = std::uint64_t(1) << 63;
         const int action : std::views::iota(0, 64)) {
        if ((move_mask & this->_legal_moves) != 0) {
            actions.push_back(action);
        }
        move_mask >>= 1;
    }
    return actions;
}

constexpr othello::Position othello::Position::apply_move(const std::uint64_t move_mask
) const noexcept {
    int player = 3 - this->_player;
    std::uint64_t player1_discs = this->_player1_discs;
    std::uint64_t player2_discs = this->_player2_discs;

    std::uint64_t *player_discs;
    std::uint64_t *opponent_discs;
    if (this->_player == 1) {
        player_discs = &player1_discs;
        opponent_discs = &player2_discs;
    } else {
        player_discs = &player2_discs;
        opponent_discs = &player1_discs;
    }

    const std::uint64_t flips = get_flips(move_mask, *player_discs, *opponent_discs);
    *player_discs |= move_mask | flips;
    *opponent_discs &= ~flips;
    const std::uint64_t legal_moves = get_legal_moves(*opponent_discs, *player_discs);

    std::uint64_t next_legal_moves;
    if (legal_moves != 0) {
        next_legal_moves = 0;
    } else {
        // The next player has no legal moves. If the current player has no legal moves either, the
        // game is over.
        next_legal_moves = get_legal_moves(*player_discs, *opponent_discs);
        if (next_legal_moves == 0) {
            player = 0;
        }
    }

    return Position(player, player1_discs, player2_discs, legal_moves, next_legal_moves);
}

constexpr othello::Position othello::Position::apply_move_checked(const std::uint64_t move_mask
) const {
    if (std::popcount(move_mask) != 1) {
        throw std::invalid_argument(std::format(
            "Expected a single bit in move_mask, but got 0b{}.",
            std::bitset<64>(move_mask).to_string()
        ));
    }
    if ((move_mask & this->_legal_moves) == 0) {
        throw std::invalid_argument(
            std::format("0b{} is not a legal move.", std::bitset<64>(move_mask).to_string())
        );
    }
    return this->apply_move(move_mask);
}

constexpr othello::Position othello::Position::apply_pass() const noexcept {
    return Position(
        3 - this->_player, this->_player1_discs, this->_player2_discs, this->_next_legal_moves, 0
    );
}

constexpr othello::Position othello::Position::apply_pass_checked() const {
    if (this->is_terminal()) {
        throw std::invalid_argument("Pass is not allowed in a terminal position.");
    }
    if (this->_legal_moves != 0) {
        throw std::invalid_argument("Pass is not allowed when there are legal moves.");
    }
    return this->apply_pass();
}

constexpr othello::Position othello::Position::apply_action(const int action) const noexcept {
    if (action == 64) {
        return this->apply_pass();
    }
    return this->apply_move(std::uint64_t(1) << (63 - action));
}

constexpr othello::Position othello::Position::apply_action_checked(const int action) const {
    if (!(0 <= action && action < 65)) {
        throw std::out_of_range(std::format("Expected 0 <= action < 65, but got {}.", action));
    }
    if (action == 64) {
        return this->apply_pass_checked();
    }
    const std::uint64_t move_mask = std::uint64_t(1) << (63 - action);
    if ((move_mask & this->_legal_moves) == 0) {
        throw std::invalid_argument(std::format("{} is not a legal action.", action));
    }
    return this->apply_move(move_mask);
}

constexpr std::string othello::Position::to_string() const {
    std::string result;
    result.reserve(9 * 18 - 1);
    result.append("  a b c d e f g h\n");

    for (std::uint64_t move_mask = std::uint64_t(1) << 63; const int row : std::views::iota(0, 8)) {
        result.push_back(static_cast<char>('1' + row));
        for (const int col : std::views::iota(0, 8)) {
            result.push_back(' ');
            const char *square;
            if (move_mask & this->_player1_discs) {
                square = "\u25cf"; // Black Circle
            } else if (move_mask & this->_player2_discs) {
                square = "\u25cb"; // White Circle
            } else if (move_mask & this->_legal_moves) {
                square = "\u00d7"; // Multiplication Sign
            } else {
                square = "\u00b7"; // Middle Dot
            }
            result.append(square);
            move_mask >>= 1;
        }
        if (row < 7) {
            result.push_back('\n');
        }
    }
    return result;
}

namespace othello {

/// @brief Initial position of an Othello game.
///
constexpr Position INITIAL_POSITION = Position::initial_position();

} // namespace othello

#endif // OTHELLO_MCTS_POSITION_H
