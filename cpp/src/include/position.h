/// @file position.h
/// @brief Declaration of the Othello game logic.

#ifndef OTHELLO_MCTS_POSITION_H
#define OTHELLO_MCTS_POSITION_H

#include <cstdint>
#include <string>
#include <vector>

namespace othello {

/// @brief Position of an Othello game.
///
class Position {
public:
    /// @brief Returns the initial position of a game.
    /// @return Initial position.
    static Position initial_position() noexcept;

    /// @brief Returns the legal actions of the current player.
    /// @return Vector of legal actions.
    std::vector<int> legal_actions() const;

    /// @brief Applies an action to the current position and returns the new
    ///     position.
    /// @param action Action to apply.
    /// @return New position.
    Position apply_action(int action) const noexcept;

    /// @brief Returns whether the position is a terminal position.
    /// @return True if the position is terminal, false otherwise.
    bool is_terminal() const noexcept {
        return _player == 0;
    }

    /// @brief Returns the current player.
    /// @return 1 if black, 2 if white, 0 if the game is over.
    int player() const noexcept {
        return _player;
    }

    /// @brief Returns the status of a square.
    /// @param row Row of the square.
    /// @param col Column of the square.
    /// @return 1 if black disc, 2 if white disc, 0 if empty, -1 if out of
    ///     bounds.
    int operator()(int row, int col) const noexcept;

    /// @brief Returns whether the given square is a legal move.
    /// @param row Row of the square.
    /// @param col Column of the square.
    /// @return True if the square is a legal move, false otherwise.
    bool is_legal_move(int row, int col) const noexcept;

    /// @brief Creates a string representation of the position.
    /// @return String representation.
    std::string to_string() const;

private:
    Position(
        int player,
        std::uint64_t p1_discs,
        std::uint64_t p2_discs,
        std::uint64_t legal_moves,
        std::uint64_t next_legal_moves
    ) noexcept
        : _player(player),
          _p1_discs(p1_discs),
          _p2_discs(p2_discs),
          _legal_moves(legal_moves),
          _next_legal_moves(next_legal_moves) {}

    int _player;
    std::uint64_t _p1_discs;
    std::uint64_t _p2_discs;
    std::uint64_t _legal_moves;
    std::uint64_t _next_legal_moves;
};

/// @brief Returns the legal moves of the current player.
/// @param player_discs Discs of the current player.
/// @param opponent_discs Discs of the opponent player.
/// @return Mask of legal moves.
std::uint64_t get_legal_moves(
    std::uint64_t player_discs, std::uint64_t opponent_discs
) noexcept;

/// @brief Returns the discs flipped by a move.
/// @param move_mask Mask of the move.
/// @param player_discs Discs of the current player.
/// @param opponent_discs Discs of the opponent player.
/// @return Mask of flipped discs.
std::uint64_t get_flips(
    std::uint64_t move_mask,
    std::uint64_t player_discs,
    std::uint64_t opponent_discs
) noexcept;

} // namespace othello

#endif // OTHELLO_MCTS_POSITION_H
