/// @file othello_mcts.cpp
/// @brief Pybind11 bindings for the othello_mcts package.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "mcts.h"
#include "neural_net.h"
#include "position.h"

namespace py = pybind11;
using namespace py::literals;

namespace {

/// @brief Pybind11 neural network wrapper.
///
class Pybind11NeuralNet {
public:
    /// @brief Constructs a pybind11 neural network wrapper.
    /// @param neural_net Neural network object that takes a tensor and returns
    ///     a dictionary with keys "policy" and "value".
    Pybind11NeuralNet(pybind11::object neural_net) : _neural_net(neural_net) {}

    /// @brief Calls the neural network with the given features.
    /// @param features Feature tensor of shape
    ///     `(batch_size, feature_channels, 8, 8)`.
    /// @return Neural network output.
    othello::NeuralNetOutput operator()(torch::Tensor features);

private:
    pybind11::object _neural_net;
};

othello::NeuralNetOutput Pybind11NeuralNet::operator()(torch::Tensor features) {
    py::dict output = _neural_net(py::cast(features));

    // Without detach(), the tensors will hold references to Python objects and
    // lead to unexpected GIL acquisitions.
    return othello::NeuralNetOutput{
        .policy = output["policy"].cast<torch::Tensor>().detach(),
        .value = output["value"].cast<torch::Tensor>().detach()
    };
}

} // namespace

PYBIND11_MODULE(_othello_mcts_impl, m) {
    using othello::MCTS;
    using othello::Position;

    m.def(
        "get_legal_moves",
        &othello::get_legal_moves,
        "player_discs"_a,
        "opponent_discs"_a
    );

    m.def(
        "get_flips",
        &othello::get_flips,
        "move_mask"_a,
        "player_discs"_a,
        "opponent_discs"_a
    );

    py::class_<Position>(m, "Position")
        .def_static(
            "initial_position",
            []() {
                constexpr Position position = Position::initial_position();
                return position;
            }
        )
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

    py::class_<MCTS>(m, "MCTS")
        .def(
            py::init<
                int,
                const std::string &,
                bool,
                int,
                int,
                int,
                float,
                float,
                float>(),
            "history_size"_a = 4,
            "torch_device"_a = "cpu",
            "torch_pin_memory"_a = false,
            "num_simulations"_a = 800,
            "num_threads"_a = 2,
            "batch_size"_a = 16,
            "exploration_weight"_a = 1.0f,
            "dirichlet_epsilon"_a = 0.25f,
            "dirichlet_alpha"_a = 0.5f
        )
        .def("reset_position", &MCTS::reset_position)
        .def("position", &MCTS::position)
        .def(
            "search",
            [](MCTS &mcts, py::object neural_net) {
                mcts.search(Pybind11NeuralNet(neural_net));
            }
        )
        .def("visit_counts", &MCTS::visit_counts)
        .def("mean_action_values", &MCTS::mean_action_values)
        .def(
            "self_play_data",
            [](MCTS &mcts) {
                othello::SelfPlayData data = mcts.self_play_data();
                return py::dict(
                    "features"_a = data.features, "policy"_a = data.policy
                );
            }
        )
        .def("apply_action", &MCTS::apply_action)
        .def("history_size", &MCTS::history_size)
        .def("set_history_size", &MCTS::set_history_size)
        .def("torch_device", &MCTS::torch_device)
        .def("set_torch_device", &MCTS::set_torch_device)
        .def("num_simulations", &MCTS::num_simulations)
        .def("set_num_simulations", &MCTS::set_num_simulations)
        .def("num_threads", &MCTS::num_threads)
        .def("set_num_threads", &MCTS::set_num_threads)
        .def("batch_size", &MCTS::batch_size)
        .def("set_batch_size", &MCTS::set_batch_size)
        .def("exploration_weight", &MCTS::exploration_weight)
        .def("set_exploration_weight", &MCTS::set_exploration_weight)
        .def("dirichlet_epsilon", &MCTS::dirichlet_epsilon)
        .def("set_dirichlet_epsilon", &MCTS::set_dirichlet_epsilon)
        .def("dirichlet_alpha", &MCTS::dirichlet_alpha)
        .def("set_dirichlet_alpha", &MCTS::set_dirichlet_alpha);
}
