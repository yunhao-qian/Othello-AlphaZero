"""Declaration file for the Othello rules."""

from libc.stdint cimport uint64_t
from libcpp.vector cimport vector


"""Game position."""
cdef struct Position:
    int time_step
    int player
    uint64_t p1_discs
    uint64_t p2_discs
    uint64_t legal_moves
    uint64_t _next_legal_moves


cdef uint64_t get_flips(
    uint64_t move_mask, uint64_t player_discs, uint64_t opponent_discs
) noexcept nogil


cpdef Position get_initial_position() noexcept nogil


cpdef bint is_terminal_position(Position &position) noexcept nogil


cpdef vector[int] get_legal_actions(Position &position) noexcept nogil


cpdef Position apply_action(Position &position, int action) noexcept nogil
