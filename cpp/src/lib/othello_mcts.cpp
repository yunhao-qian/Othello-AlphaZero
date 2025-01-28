/// @file othello_mcts.cpp
/// @brief Pybind11 bindings for the othello_mcts package.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "position.h"

namespace nb = nanobind;

NB_MODULE(_othello_mcts_impl, m) {
    using othello::Position;

    nb::class_<Position>(m, "Position")
        .def_static("initial_position", []() { return othello::INITIAL_POSITION; })
        .def("player", &Position::player)
        .def("player1_discs", &Position::player1_discs)
        .def("player2_discs", &Position::player2_discs)
        .def("__getitem__", &Position::at)
        .def("legal_moves", &Position::legal_moves)
        .def("is_legal_move", &Position::is_legal_move_checked)
        .def("legal_actions", &Position::legal_actions)
        .def("apply_move", &Position::apply_move_checked)
        .def("apply_pass", &Position::apply_pass_checked)
        .def("apply_action", &Position::apply_action_checked)
        .def("is_terminal", &Position::is_terminal)
        .def("__str__", &Position::to_string);
}
