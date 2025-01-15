/// @file neural_net_wrapper.cpp
/// @brief Implementation of the neural network wrapper.

#include "neural_net_wrapper.h"

#include <torch/extension.h>

othello::NeuralNetOutput
othello::NeuralNetWrapper::operator()(torch::Tensor features) {
    py::dict output = _neural_net(py::cast(features));

    // Without detach(), the tensors will hold references to Python objects and
    // lead to unexpected GIL acquisitions.
    return othello::NeuralNetOutput{
        .policy = output["policy"].cast<torch::Tensor>().detach(),
        .value = output["value"].cast<torch::Tensor>().detach()
    };
}
