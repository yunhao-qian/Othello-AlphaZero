# distutils: language = c++
"""Greedy player implementation."""

from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

from .othello cimport (
    apply_action,
    get_flips,
    get_initial_position,
    get_legal_actions,
    Position,
)
from .utility cimport bitwise_count


cdef class GreedyPlayer:
    """A player that always chooses the move that flips the most discs."""

    cdef Position position

    def __init__(self) -> None:
        self.position = get_initial_position()

    def get_position(self) -> Position:
        return self.position

    def get_best_action(self) -> None:
        cdef vector[int] actions = get_legal_actions(self.position)
        if actions.size() <= 1:
            return actions[0]

        cdef uint64_t player_discs
        cdef uint64_t opponent_discs
        if self.position.player == 1:
            player_discs = self.position.p1_discs
            opponent_discs = self.position.p2_discs
        else:
            player_discs = self.position.p2_discs
            opponent_discs = self.position.p1_discs

        cdef uint64_t move_mask
        cdef int num_flips
        cdef int max_num_flips = 0
        cdef int best_action = 0

        for action in actions:
            move_mask = <uint64_t>1 << (63 - action)
            num_flips = bitwise_count(
                get_flips(move_mask, player_discs, opponent_discs)
            )
            if num_flips > max_num_flips:
                max_num_flips = num_flips
                best_action = action

        return best_action

    def apply_action(self, action: int) -> None:
        self.position = apply_action(self.position, action)
