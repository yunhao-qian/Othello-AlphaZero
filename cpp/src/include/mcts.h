/// @file mcts.h
/// @brief Declaration of the Monte Carlo Tree Search algorithm.

#ifndef OTHELLO_MCTS_MCTS_H
#define OTHELLO_MCTS_MCTS_H

#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "position.h"
#include "queue.h"

namespace othello {

/// @brief Node in the search tree.
///
class SearchNode {
public:
    SearchNode(
        const Position &position,
        int previous_action = 64,
        int visit_count = 0,
        float total_action_value = 0.0f,
        float mean_action_value = 0.0f,
        float prior_probability = 1.0f
    )
        : position(position),
          previous_action(previous_action),
          visit_count(visit_count),
          total_action_value(total_action_value),
          mean_action_value(mean_action_value),
          prior_probability(prior_probability) {}

    Position position;
    int previous_action;
    int visit_count;
    float total_action_value;
    float mean_action_value;
    float prior_probability;
    std::vector<unsigned> children;
};

/// @brief Input from search threads to the neural net input thread.
///
struct NeuralNetInput {
    int thread_id;
    std::vector<float> features;
};

/// @brief Output from the neural net output thread to a search thread.
///
struct NeuralNetOutput {
    std::vector<float> policy;
    float value;
};

/// @brief Batched input from the neural net input thread to the main
///     thread.
struct BatchedNeuralNetInput {
    std::vector<int> thread_ids;
    torch::Tensor features;
};

/// @brief Batched output from the main thread to the neural net output thread.
///
struct BatchedNeuralNetOutput {
    std::vector<int> thread_ids;
    torch::Tensor policies;
    torch::Tensor values;
};

class MCTS;

/// @brief Local logic and data for a single search thread.
///
class SearchThread {
public:
    SearchThread(MCTS &mcts, int thread_id);

    /// @brief Runs the search thread.
    ///
    void run();

private:
    friend class MCTS;

    /// @brief Runs a single simulation.
    ///
    void _simulate();

    /// @brief Chooses the child with the highest Upper Confidence Bound (UCB).
    /// @param node_index Index of the parent node.
    /// @return Index of the chosen child.
    unsigned _choose_best_child(unsigned node_index);

    MCTS *_mcts;
    int _thread_id;
    Queue<NeuralNetOutput> _neural_net_output_queue;
    std::vector<unsigned> _search_path;
    std::mt19937 _random_engine;
    std::gamma_distribution<float> _gamma_distribution;
};

/// @brief Monte Carlo Tree Search algorithm.
///
class MCTS {
public:
    MCTS(
        const std::string &torch_device = "cpu",
        int num_simulations = 800,
        int batch_size = 16,
        int num_threads = 16,
        float exploration_weight = 1.0f,
        float dirichlet_epsilon = 0.25f,
        float dirichlet_alpha = 0.3f
    );

    /// @brief Clears the search tree and resets the root node to the given
    ///     position.
    /// @param position Position to reset to.
    void reset_position(const Position &position);

    /// @brief Returns the position of the root node of the search tree.
    /// @return Position of the root node.
    Position root_position() const;

    /// @brief  Performs a Monte Carlo Tree Search using the given neural net.
    /// @param neural_net PyTorch neural net.
    /// @return Python dictionary of `actions`, `visit_counts`, and
    ///     `mean_action_values`.
    pybind11::object search(pybind11::object neural_net);

    /// @brief Applies an action to the current position and prunes the search
    ///     tree accordingly.
    /// @param action Action to apply.
    void apply_action(int action);

    std::string torch_device() const {
        return _torch_device;
    }

    void set_torch_device(const std::string &value) {
        _torch_device = value;
    }

    int num_simulations() const noexcept {
        return _num_simulations;
    }

    void set_num_simulations(int value) noexcept;

    int batch_size() const noexcept {
        return _batch_size;
    }

    void set_batch_size(int value) noexcept;

    int num_threads() const noexcept {
        return _num_threads;
    }

    void set_num_threads(int value) noexcept;

    float exploration_weight() const noexcept {
        return _exploration_weight;
    }

    void set_exploration_weight(float value) noexcept;

    float dirichlet_epsilon() const noexcept {
        return _dirichlet_epsilon;
    }

    void set_dirichlet_epsilon(float value) noexcept;

    float dirichlet_alpha() const noexcept {
        return _dirichlet_alpha;
    }

    void set_dirichlet_alpha(float value) noexcept;

private:
    friend class SearchThread;

    /// @brief Entry point for the neural net input thread.
    ///
    void _neural_net_input_thread();

    /// @brief Entry point for the neural net output thread.
    ///
    void _neural_net_output_thread();

    std::string _torch_device;
    int _num_simulations;
    int _batch_size;
    int _num_threads;
    float _exploration_weight;
    float _dirichlet_epsilon;
    float _dirichlet_alpha;

    std::vector<SearchNode> _search_tree;
    std::mutex _search_tree_mutex;

    Queue<NeuralNetInput> _neural_net_input_queue;
    Queue<BatchedNeuralNetInput> _batched_neural_net_input_queue;
    Queue<BatchedNeuralNetOutput> _batched_neural_net_output_queue;
    std::vector<std::unique_ptr<SearchThread>> _search_threads;
};

} // namespace othello

#endif // OTHELLO_MCTS_MCTS_H
