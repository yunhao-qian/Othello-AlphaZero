/// @file othello_mcts.cpp
/// @brief Definition of Python bindings.

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mcts.h"
#include "position.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_othello_mcts_impl, m) {
    using othello::MCTS;
    using othello::Position;

    nb::class_<Position>(m, "Position")
        .def_static("initial_position", &Position::initial_position)
        .def("legal_actions", &Position::legal_actions)
        .def("apply_action", &Position::apply_action, "action"_a)
        .def("is_terminal", &Position::is_terminal)
        .def("action_value", &Position::action_value)
        .def("to_features", &Position::to_features)
        .def("player", &Position::player)
        .def("__call__", &Position::operator(), "row"_a, "col"_a)
        .def("is_legal_move", &Position::is_legal_move, "row"_a, "col"_a)
        .def("__str__", &Position::to_string);

    nb::class_<MCTS>(m, "MCTS")
        .def(
            nb::init<int, int, int, float, float, float>(),
            "num_simulations"_a = 1600,
            "batch_size"_a = 16,
            "num_threads"_a = 16,
            "exploration_weight"_a = 1.0f,
            "dirichlet_epsilon"_a = 0.25f,
            "dirichlet_alpha"_a = 0.03f
        )
        .def("reset_position", &MCTS::reset_position, "position"_a)
        .def("root_position", &MCTS::root_position)
        .def("apply_action", &MCTS::apply_action, "action"_a)
        .def_prop_rw(
            "num_simulations",
            [](MCTS &t) { return t.num_simulations(); },
            [](MCTS &t, int value) { t.set_num_simulations(value); }
        )
        .def_prop_rw(
            "batch_size",
            [](MCTS &t) { return t.batch_size(); },
            [](MCTS &t, int value) { t.set_batch_size(value); }
        )
        .def_prop_rw(
            "num_threads",
            [](MCTS &t) { return t.num_threads(); },
            [](MCTS &t, int value) { t.set_num_threads(value); }
        )
        .def_prop_rw(
            "exploration_weight",
            [](MCTS &t) { return t.exploration_weight(); },
            [](MCTS &t, float value) { t.set_exploration_weight(value); }
        )
        .def_prop_rw(
            "dirichlet_epsilon",
            [](MCTS &t) { return t.dirichlet_epsilon(); },
            [](MCTS &t, float value) { t.set_dirichlet_epsilon(value); }
        )
        .def_prop_rw(
            "dirichlet_alpha",
            [](MCTS &t) { return t.dirichlet_alpha(); },
            [](MCTS &t, float value) { t.set_dirichlet_alpha(value); }
        );
}
