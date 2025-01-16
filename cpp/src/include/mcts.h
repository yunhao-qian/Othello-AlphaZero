/// @file mcts.h
/// @brief Declaration of the Monte Carlo Tree Search algorithm.

#ifndef OTHELLO_MCTS_MCTS_H
#define OTHELLO_MCTS_MCTS_H

#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "neural_net.h"
#include "position.h"
#include "queue"
#include "search_node.h"
#include "search_thread.h"

namespace othello {

/// @brief Self-play data for training neural networks.
///
struct SelfPlayData {
    /// @brief Vector of 8 feature tensors, each of shape
    ///     `(feature_channels, 8, 8)`.
    std::vector<torch::Tensor> features;

    /// @brief Vector of 8 policy tensors, each of shape `(65,)`.
    ///
    std::vector<torch::Tensor> policy;
};

/// @brief Monte Carlo Tree Search algorithm.
///
class MCTS {
public:
    /// @brief Constructs a reusable object for Monte Carlo Tree Search.
    /// @param history_size Number of history positions included in the neural
    ///     network input features.
    /// @param torch_device PyTorch device for neural network inference.
    /// @param torch_pin_memory Whether to use pinned memory for PyTorch tensors
    ///     on the CPU.
    /// @param num_simulations Total number of simulations for a single search.
    /// @param num_threads Number of threads for parallel search.
    /// @param batch_size Batch size for neural network inference.
    /// @param exploration_weight Exploration weight for the upper confidence
    ///     bound. It is called `c_puct` in the AlphaGo Zero paper.
    /// @param dirichlet_epsilon Epsilon for Dirichlet noise.
    /// @param dirichlet_alpha Alpha for Dirichlet noise.
    MCTS(
        int history_size = 4,
        const std::string &torch_device = "cpu",
        bool torch_pin_memory = false,
        int num_simulations = 800,
        int num_threads = 2,
        int batch_size = 16,
        float exploration_weight = 1.0f,
        float dirichlet_epsilon = 0.25f,
        float dirichlet_alpha = 0.5f
    );

    /// @brief Resets the search tree to the initial position.
    ///
    void reset_position();

    /// @brief Gets the current position.
    /// @return Current position.
    Position position() const noexcept {
        return _search_tree->position;
    }

    /// @brief Performs a Monte Carlo Tree Search using the given neural
    ///     network.
    /// @tparam T Type of the neural network satisfying the othello::NeuralNet
    ///     concept.
    /// @param neural_net Neural network.
    template <NeuralNet T>
    void search(T &&neural_net);

    /// @brief Returns the visit counts of the edges from the root node.
    /// @return Vector of visit counts.
    std::vector<int> visit_counts();

    /// @brief Returns the mean action-values of the edges from the root node.
    /// @return Vector of mean action-values.
    std::vector<float> mean_action_values();

    /// @brief Returns the self-play data for training neural networks.
    /// @return Self-play data.
    SelfPlayData self_play_data();

    /// @brief Applies an action to the current position and updates the search
    ///     tree accordingly.
    /// @param action Action to apply.
    void apply_action(int action);

    /// @brief Gets the history size.
    /// @return History size.
    int history_size() const noexcept {
        return _history_size;
    }

    /// @brief Sets the history size.
    /// @param value History size.
    void set_history_size(int value);

    /// @brief Gets the PyTorch device.
    /// @return PyTorch device.
    std::string torch_device() const {
        return _torch_device;
    }

    /// @brief Sets the PyTorch device.
    /// @param value PyTorch device.
    void set_torch_device(const std::string &value) {
        _torch_device = value;
    }

    /// @brief Gets the PyTorch pin memory flag.
    /// @return PyTorch pin memory flag.
    bool torch_pin_memory() const noexcept {
        return _torch_pin_memory;
    }

    /// @brief Sets the PyTorch pin memory flag.
    /// @param value PyTorch pin memory flag.
    void set_torch_pin_memory(bool value) {
        _torch_pin_memory = value;
    }

    /// @brief Gets the number of simulations.
    /// @return Number of simulations.
    int num_simulations() const noexcept {
        return _num_simulations;
    }

    /// @brief Sets the number of simulations.
    /// @param value Number of simulations.
    void set_num_simulations(int value);

    /// @brief Gets the number of threads.
    /// @return Number of threads.
    int num_threads() const noexcept {
        return _num_threads;
    }

    /// @brief Sets the number of threads.
    /// @param value Number of threads.
    void set_num_threads(int value);

    /// @brief Gets the batch size.
    /// @return Batch size.
    int batch_size() const noexcept {
        return _batch_size;
    }

    /// @brief Sets the batch size.
    /// @param value Batch size.
    void set_batch_size(int value);

    /// @brief Gets the exploration weight.
    /// @return Exploration weight.
    float exploration_weight() const noexcept {
        return _exploration_weight;
    }

    /// @brief Sets the exploration weight.
    /// @param value Exploration weight.
    void set_exploration_weight(float value);

    /// @brief Gets the Dirichlet noise epsilon.
    /// @return Dirichlet noise epsilon.
    float dirichlet_epsilon() const noexcept {
        return _dirichlet_epsilon;
    }

    /// @brief Sets the Dirichlet noise epsilon.
    /// @param value Dirichlet noise epsilon.
    void set_dirichlet_epsilon(float value);

    /// @brief Gets the Dirichlet noise alpha.
    /// @return Dirichlet noise alpha.
    float dirichlet_alpha() const noexcept {
        return _dirichlet_alpha;
    }

    /// @brief Sets the Dirichlet noise alpha.
    /// @param value Dirichlet noise alpha.
    void set_dirichlet_alpha(float value);

private:
    int _history_size;
    std::string _torch_device;
    bool _torch_pin_memory;
    int _num_simulations;
    int _num_threads;
    int _batch_size;
    float _exploration_weight;
    float _dirichlet_epsilon;
    float _dirichlet_alpha;

    std::unique_ptr<SearchNode> _search_tree;
    std::vector<std::unique_ptr<SearchNode>> _history;
};

} // namespace othello

template <othello::NeuralNet T>
void othello::MCTS::search(T &&neural_net) {
    std::mutex search_tree_mutex;
    Queue<NeuralNetInput> neural_net_input_queue;

    std::size_t num_threads = static_cast<std::size_t>(_num_threads);
    std::vector<std::unique_ptr<SearchThread>> search_threads;
    search_threads.reserve(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < _num_threads; ++i) {
        search_threads.push_back(std::make_unique<SearchThread>(
            this,
            _search_tree.get(),
            &search_tree_mutex,
            &neural_net_input_queue
        ));
        threads.emplace_back(&SearchThread::run, search_threads[i].get());
    }

    int num_running_threads = _num_threads;
    while (true) {
        NeuralNetInput input = neural_net_input_queue.pop();
        if (input.output_queue == nullptr) {
            if (--num_running_threads == 0) {
                break;
            }
            continue;
        }
        input.output_queue->push(neural_net(input.features));
    }

    for (std::thread &thread : threads) {
        thread.join();
    }
}

#endif // OTHELLO_MCTS_MCTS_H
