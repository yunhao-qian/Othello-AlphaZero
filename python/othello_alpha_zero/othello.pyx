# distutils: language = c++
"""Implementation file for the Othello rules."""

from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

from .othello cimport (
    apply_action,
    get_initial_position,
    get_legal_actions,
    is_terminal_position,
    Position,
)
from .utility cimport bitwise_count

from PIL import Image, ImageDraw


cdef inline uint64_t _shift(uint64_t mask, int stride) noexcept nogil:
    """Shifts the given 8x8 mask in the direction of `stride`."""

    if stride < 0:
        return mask << -stride
    return mask >> stride


cdef inline uint64_t _get_potential_flips_in_direction(
    uint64_t player_discs, uint64_t opponent_discs, int stride
) noexcept nogil:
    """Returns the potential flips starting from the given discs of the current player
    in the direction of `stride`."""

    cdef uint64_t result = opponent_discs & _shift(player_discs, stride)
    for _ in range(5):
        result |= opponent_discs & _shift(result, stride)
    return result


cdef inline uint64_t _get_flips_in_direction(
    uint64_t move_mask, uint64_t player_discs, uint64_t opponent_discs, int stride
) noexcept nogil:
    """Returns the flips starting from `move_mask` in the direction of `stride`."""

    cdef uint64_t potential_flips = _get_potential_flips_in_direction(
        move_mask, opponent_discs, stride
    )
    if _shift(potential_flips, stride) & player_discs:
        return potential_flips
    return 0


cdef int[8] _STRIDES = [-9, -8, -7, -1, 1, 7, 8, 9]

cdef uint64_t _NO_LEFT_RIGHT_MASK = (
    0b_01111110_01111110_01111110_01111110_01111110_01111110_01111110_01111110
)
cdef uint64_t _NO_TOP_BOTTOM_MASK = (
    0b_00000000_11111111_11111111_11111111_11111111_11111111_11111111_00000000
)
cdef uint64_t _NO_EDGES_MASK = _NO_LEFT_RIGHT_MASK & _NO_TOP_BOTTOM_MASK

cdef uint64_t[8] _MASKS = [
    _NO_EDGES_MASK,
    _NO_TOP_BOTTOM_MASK,
    _NO_EDGES_MASK,
    _NO_LEFT_RIGHT_MASK,
    _NO_LEFT_RIGHT_MASK,
    _NO_EDGES_MASK,
    _NO_TOP_BOTTOM_MASK,
    _NO_EDGES_MASK,
]


cdef uint64_t _get_legal_moves(
    uint64_t player_discs, uint64_t opponent_discs
) noexcept nogil:
    """Returns a mask of the legal moves."""

    cdef uint64_t legal_moves = 0

    cdef int stride
    cdef uint64_t mask
    cdef uint64_t potential_flips

    for i in range(8):
        stride = _STRIDES[i]
        mask = _MASKS[i]
        potential_flips = _get_potential_flips_in_direction(
            player_discs, opponent_discs & mask, stride
        )
        legal_moves |= _shift(potential_flips, stride)

    legal_moves &= ~(player_discs | opponent_discs)
    return legal_moves


cdef uint64_t get_flips(
    uint64_t move_mask, uint64_t player_discs, uint64_t opponent_discs
) noexcept nogil:
    """Returns the discs flipped by `move_mask`."""

    cdef uint64_t flips = 0

    cdef int stride
    cdef uint64_t mask

    for i in range(8):
        stride = _STRIDES[i]
        mask = _MASKS[i]
        flips |= _get_flips_in_direction(
            move_mask, player_discs, opponent_discs & mask, stride
        )

    return flips


cpdef Position get_initial_position() noexcept nogil:
    """Returns the initial position of the game."""

    cdef Position position
    position.time_step = 0
    position.player = 1
    position.p1_discs = (
        0b_00000000_00000000_00000000_00001000_00010000_00000000_00000000_00000000
    )
    position.p2_discs = (
        0b_00000000_00000000_00000000_00010000_00001000_00000000_00000000_00000000
    )
    position.legal_moves = _get_legal_moves(position.p1_discs, position.p2_discs)
    position._next_legal_moves = 0
    return position


cpdef bint is_terminal_position(Position &position) noexcept nogil:
    """Returns the given position is the terminal position of a game."""

    return position.player == 0


cpdef vector[int] get_legal_actions(Position &position) noexcept nogil:
    """Returns a vector of legal actions for the given position."""

    # Theoretically, push_back() has a rare chance of throwing C++ exceptions. However,
    # we still declare the function as noexcept because it is required for use with
    # nogil.

    cdef vector[int] actions

    cdef uint64_t move_mask
    if position.legal_moves == 0:
        actions.push_back(64)
    else:
        actions.reserve(<size_t>bitwise_count(position.legal_moves))
        move_mask = <uint64_t>1 << 63
        for action in range(64):
            if move_mask & position.legal_moves:
                actions.push_back(action)
            move_mask >>= 1

    return actions


cpdef Position apply_action(Position &position, int action) noexcept nogil:
    """Applies an action to the position and returns the new position."""

    cdef Position new_position
    new_position.time_step = position.time_step + 1
    new_position.player = 3 - position.player

    cdef uint64_t player_discs
    cdef uint64_t opponent_discs
    if position.player == 1:
        player_discs = position.p1_discs
        opponent_discs = position.p2_discs
    else:
        player_discs = position.p2_discs
        opponent_discs = position.p1_discs

    cdef uint64_t move_mask
    cdef uint64_t flips
    if action == 64:
        # Pass. The next legal moves have been stored in the previous position.
        new_position.legal_moves = position._next_legal_moves
    else:
        move_mask = <uint64_t>1 << (63 - action)
        flips = get_flips(move_mask, player_discs, opponent_discs)
        player_discs |= move_mask | flips
        opponent_discs &= ~flips

        new_position.legal_moves = _get_legal_moves(opponent_discs, player_discs)

    if position.player == 1:
        new_position.p1_discs = player_discs
        new_position.p2_discs = opponent_discs
    else:
        new_position.p1_discs = opponent_discs
        new_position.p2_discs = player_discs

    if new_position.legal_moves == 0:
        # The next player has no legal moves. If the current player has no legal moves
        # either, the game is over.
        new_position._next_legal_moves = _get_legal_moves(player_discs, opponent_discs)
        if new_position._next_legal_moves == 0:
            # Set player to 0 to indicate that the game is over.
            new_position.player = 0
    else:
        # If not a pass, we do not know the next legal moves yet.
        new_position._next_legal_moves = 0

    return new_position


def visualize_position(position: Position) -> Image.Image:
    """Visualizes the given position as an image."""

    image_size = 512
    square_size = 60
    margin_width = (image_size - 8 * square_size) // 2
    border_width = 4
    grid_line_width = 1
    dot_radius = 2
    disc_radius = 25
    disc_line_width = 1

    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (
            margin_width,
            margin_width,
            image_size - margin_width,
            image_size - margin_width,
        ),
        radius=border_width,
        outline="black",
        width=border_width,
    )

    for i in range(1, 8):
        line_position = margin_width + i * square_size
        line_start = margin_width
        line_end = image_size - margin_width
        draw.line(
            (line_start, line_position, line_end, line_position),
            fill="black",
            width=grid_line_width,
        )
        draw.line(
            (line_position, line_start, line_position, line_end),
            fill="black",
            width=grid_line_width,
        )

    for row in 2, 6:
        y = margin_width + row * square_size
        for col in 2, 6:
            x = margin_width + col * square_size
            draw.circle((x, y), dot_radius, fill="black")

    cdef uint64_t square_mask = <uint64_t>1 << 63
    cdef uint64_t discs

    for row in range(8):
        y = margin_width + (row + 0.5) * square_size
        for col in range(8):
            x = margin_width + (col + 0.5) * square_size

            for discs, color in (
                (position.p1_discs, "black"),
                (position.p2_discs, "white"),
            ):
                if discs & square_mask:
                    draw.circle(
                        (x, y),
                        disc_radius,
                        fill=color,
                        outline="black",
                        width=disc_line_width,
                    )

            if position.legal_moves & square_mask:
                draw.polygon(
                    [
                        (x + disc_radius, y),
                        (x, y + disc_radius),
                        (x - disc_radius, y),
                        (x, y - disc_radius),
                    ],
                    fill="gray" if position.player == 1 else "white",
                    outline="gray",
                    width=disc_line_width,
                )

            square_mask >>= 1

    return image
