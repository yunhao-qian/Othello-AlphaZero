/// @file neural_net_wrapper.h
/// @brief Declaration of the neural network wrapper.

#ifndef NEURAL_NET_WRAPPER_H
#define NEURAL_NET_WRAPPER_H

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "search_thread.h"

namespace othello {

/// @brief Wrapper for a PyTorch neural network.
///
class NeuralNetWrapper {
public:
    /// @brief Constructs a neural network wrapper.
    /// @param neural_net PyTorch neural network.
    NeuralNetWrapper(pybind11::object neural_net) : _neural_net(neural_net) {}

    /// @brief Calls the neural network with the given features.
    /// @param features Feature tensor of shape
    ///     `(batch_size, feature_channels, 8, 8)`.
    /// @return Neural network output.
    NeuralNetOutput operator()(torch::Tensor features);

private:
    pybind11::object _neural_net;
};

} // namespace othello

#endif // NEURAL_NET_WRAPPER_H
