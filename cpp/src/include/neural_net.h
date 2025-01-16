/// @file neural_net.h
/// @brief Interface for interacting with neural networks.

#ifndef OTHELLO_MCTS_NEURAL_NET_H
#define OTHELLO_MCTS_NEURAL_NET_H

#include <concepts>

#include <torch/torch.h>

namespace othello {

/// @brief Output of the neural network.
///
struct NeuralNetOutput {
    /// @brief Policy tensor of shape `(batch_size, 65)`.
    ///
    torch::Tensor policy;

    /// @brief Value tensor of shape `(batch_size,)`.
    torch::Tensor value;
};

/// @brief Concept for a neural network.
///
template <typename T>
concept NeuralNet = requires(T neural_net, torch::Tensor features) {
    { neural_net(features) } -> std::same_as<NeuralNetOutput>;
};

} // namespace othello

#endif // OTHELLO_MCTS_NEURAL_NET_H
