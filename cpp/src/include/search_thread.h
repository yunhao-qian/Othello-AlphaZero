/// @file search_thread.h
/// @brief Declaration of the search thread.

#ifndef OTHELLO_MCTS_SEARCH_THREAD_H
#define OTHELLO_MCTS_SEARCH_THREAD_H

#include <mutex>
#include <random>

#include <torch/torch.h>

#include "queue.h"

namespace othello {

class SearchNode;
class MCTS;

/// @brief Output of the neural network.
///
struct NeuralNetOutput {
    /// @brief Policy tensor of shape `(batch_size, 65)`.
    ///
    torch::Tensor policy;

    /// @brief Value tensor of shape `(batch_size,)`.
    torch::Tensor value;
};

/// @brief Input to the neural network.
///
struct NeuralNetInput {
    /// @brief Feature tensor of shape `(batch_size, feature_channels, 8, 8)`.
    ///
    torch::Tensor features;

    /// @brief Queue to push the neural network output to.
    ///
    Queue<NeuralNetOutput> *output_queue;
};

/// @brief Search thread.
///
class SearchThread {
public:
    /// @brief Constructs a search thread.
    /// @param mcts MCTS object to get the parameters from.
    /// @param search_tree Root of the search tree.
    /// @param search_tree_mutex Mutex to lock the search tree.
    /// @param neural_net_input_queue Queue to push the neural network inputs
    ///     to.
    SearchThread(
        const MCTS *mcts,
        SearchNode *seach_tree,
        std::mutex *search_tree_mutex,
        Queue<NeuralNetInput> *neural_net_input_queue
    );

    /// @brief Runs the search thread.
    ///
    void run();

private:
    /// @brief Runs a batch of simulations simultaneously.
    ///
    void _simulate_batch();

    /// @brief Expands the leaf node if it is not terminal, and back-propagates
    ///     the action-value.
    /// @param leaf Leaf node.
    /// @param transformation Transformation applied to the positions.
    /// @param policy Policy data of shape `(65,)`.
    /// @param value Value data of shape `()`.
    void _expand_and_backward(
        SearchNode *leaf, int transformation, float *policy, float *value
    );

    /// @brief Chooses the best child node according to the UCB formula.
    /// @param node Parent node.
    /// @return Best child node.
    SearchNode *_choose_best_child(const SearchNode *node);

    const MCTS *_mcts;
    SearchNode *_search_tree;
    std::mutex *_search_tree_mutex;
    Queue<NeuralNetInput> *_neural_net_input_queue;
    Queue<NeuralNetOutput> _neural_net_output_queue;
    std::mt19937 _random_engine;
    std::gamma_distribution<float> _gamma_distribution;
    std::uniform_int_distribution<int> _transformation_distribution;

    std::vector<SearchNode *> _leaves;
    std::vector<int> _transformations;
    torch::Tensor _features_cpu;
    torch::Tensor _features_device;
    torch::Tensor _policy_cpu;
    torch::Tensor _value_cpu;
};

} // namespace othello

#endif // OTHELLO_MCTS_SEARCH_THREAD_H
