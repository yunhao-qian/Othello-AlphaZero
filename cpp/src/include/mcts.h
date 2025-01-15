/// @file mcts.h
/// @brief Declaration of the Monte Carlo Tree Search algorithm.

#ifndef OTHELLO_MCTS_MCTS_H
#define OTHELLO_MCTS_MCTS_H

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "position.h"
#include "queue.h"
#include "search_thread.h"

namespace othello {

/// @brief A node in the search tree.
///
struct SearchNode {
    /// @brief Game position.
    ///
    Position position;

    /// @brief Parent node.
    /// @details Unless the node corresponds to the initial position, this
    ///     pointer should always point to the node of the previous position,
    ///     even if the node is the root or is no longer in the search tree.
    SearchNode *parent = nullptr;

    /// @brief Child nodes.
    ///
    std::vector<std::unique_ptr<SearchNode>> children = {};

    /// @brief Visit count of the preceding edge.
    ///
    int visit_count = 0;

    /// @brief Total action-value of the preceding edge.
    ///
    float total_action_value = 0.0f;

    /// @brief Mean action-value of the preceding edge.
    ///
    float mean_action_value = 0.0f;

    /// @brief Prior probability of the preceding edge.
    ///
    float prior_probability = 1.0f;
};

/// @brief Result of a Monte Carlo Tree Search.
///
struct MCTSResult {
    /// @brief Legal actions of the current position.
    ///
    std::vector<int> actions;

    /// @brief Visit counts of the edges from the root node.
    ///
    std::vector<int> visit_counts;

    /// @brief Mean action-values of the edges from the root node.
    ///
    std::vector<float> mean_action_values;
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

    /// @brief Performs a Monte Carlo Tree Search.
    /// @tparam T Type of the neural network, which should be callable with a
    ///     torch::Tensor and return an othello::NeuralNetOutput.
    /// @param neural_net Neural network.
    /// @return Result of the search.
    template <typename T>
    MCTSResult search(T &&neural_net);

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

template <typename T>
othello::MCTSResult othello::MCTS::search(T &&neural_net) {
    std::mutex search_tree_mutex;
    Queue<NeuralNetInput> neural_net_input_queue;

    std::size_t num_threads = static_cast<std::size_t>(_num_threads);
    std::vector<std::unique_ptr<SearchThread>> search_threads(num_threads);
    std::vector<std::thread> threads(num_threads);

    for (int i = 0; i < _num_threads; ++i) {
        search_threads[i] = std::make_unique<SearchThread>(
            this,
            _search_tree.get(),
            &search_tree_mutex,
            &neural_net_input_queue
        );
        threads[i] = std::thread(&SearchThread::run, search_threads[i].get());
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

    MCTSResult result;
    result.actions = _search_tree->position.legal_actions();
    result.visit_counts.resize(result.actions.size());
    result.mean_action_values.resize(result.actions.size());
    for (std::size_t i = 0; i < _search_tree->children.size(); ++i) {
        SearchNode *child = _search_tree->children[i].get();
        result.visit_counts[i] = child->visit_count;
        result.mean_action_values[i] = child->mean_action_value;
    }
    return result;
}

#endif // OTHELLO_MCTS_MCTS_H
