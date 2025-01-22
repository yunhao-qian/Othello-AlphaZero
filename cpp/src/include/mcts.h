/// @file mcts.h
/// @brief Declaration of the Monte Carlo Tree Search algorithm.

#ifndef OTHELLO_MCTS_MCTS_H
#define OTHELLO_MCTS_MCTS_H

#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "neural_net.h"
#include "position.h"
#include "queue.h"
#include "search_node.h"

namespace othello {

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
    /// @param c_puct_init \f$c_\text{init}\f$ for the PUCT formula.
    /// @param c_puct_base \f$c_\text{base}\f$ for the PUCT formula.
    MCTS(
        int history_size = 4,
        const std::string &torch_device = "cpu",
        bool torch_pin_memory = false,
        int num_simulations = 800,
        int num_threads = 2,
        int batch_size = 16,
        float c_puct_init = 20000.0f,
        float c_puct_base = 2.5f
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

    /// @brief Gets the visit counts of the edges from the root node.
    /// @return Vector of visit counts.
    std::vector<int> visit_counts();

    /// @brief Gets the mean action-values of the edges from the root node.
    /// @return Vector of mean action-values.
    std::vector<float> mean_action_values();

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

    /// @brief Gets the \f$c_\text{base}\f$ for the PUCT formula.
    /// @return \f$c_\text{base}\f$ for the PUCT formula.
    float c_puct_base() const noexcept {
        return _c_puct_base;
    }

    /// @brief Sets the \f$c_\text{base}\f$ for the PUCT formula.
    /// @param value \f$c_\text{base}\f$ for the PUCT formula.
    void set_c_puct_base(float value);

    /// @brief Gets the \f$c_\text{init}\f$ for the PUCT formula.
    /// @return \f$c_\text{init}\f$ for the PUCT formula.
    float c_puct_init() const noexcept {
        return _c_puct_init;
    }

    /// @brief Sets the \f$c_\text{init}\f$ for the PUCT formula.
    /// @param value \f$c_\text{init}\f$ for the PUCT formula.
    void set_c_puct_init(float value);

private:
    int _history_size;
    std::string _torch_device;
    bool _torch_pin_memory;
    int _num_simulations;
    int _num_threads;
    int _batch_size;
    float _c_puct_base;
    float _c_puct_init;

    std::unique_ptr<SearchNode> _search_tree;
    std::vector<std::unique_ptr<SearchNode>> _history;
};

/// @brief Input to the neural network.
///
struct MCTSNeuralNetInput {
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
        Queue<MCTSNeuralNetInput> *neural_net_input_queue
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
    Queue<MCTSNeuralNetInput> *_neural_net_input_queue;
    Queue<NeuralNetOutput> _neural_net_output_queue;
    std::mt19937 _random_engine;
    std::uniform_int_distribution<int> _transformation_distribution;

    std::vector<SearchNode *> _leaves;
    std::vector<int> _transformations;
    torch::Tensor _features_cpu;
    torch::Tensor _features_device;
    torch::Tensor _policy_cpu;
    torch::Tensor _value_cpu;
};

} // namespace othello

template <othello::NeuralNet T>
void othello::MCTS::search(T &&neural_net) {
    std::mutex search_tree_mutex;
    Queue<MCTSNeuralNetInput> neural_net_input_queue;

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
        MCTSNeuralNetInput input = neural_net_input_queue.pop();
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
