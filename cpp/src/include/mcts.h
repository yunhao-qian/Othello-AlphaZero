/// @file mcts.h
/// @brief Declaration of the Monte Carlo Tree Search algorithm.

#ifndef OTHELLO_MCTS_MCTS_H
#define OTHELLO_MCTS_MCTS_H

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "position.h"
#include "queue.h"
#include "search_thread.h"
#include "transformation.h"

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

/// @brief Iterator over the history positions.
///
class HistoryPositionIterator {
public:
    /// @brief Constructs a history position iterator.
    /// @param node Node to start from.
    HistoryPositionIterator(othello::SearchNode *node) noexcept : _node(node) {}

    /// @brief Equality comparison.
    /// @param other Another iterator.
    /// @return True if the iterators are equal, false otherwise.
    bool operator==(const HistoryPositionIterator &other) const noexcept {
        return _node == other._node;
    }

    /// @brief Inequality comparison.
    /// @param other Another iterator.
    /// @return True if the iterators are not equal, false otherwise.
    bool operator!=(const HistoryPositionIterator &other) const noexcept {
        return _node != other._node;
    }

    /// @brief Dereference operator.
    /// @return Reference to the current position.
    othello::Position &operator*() const noexcept {
        return _node->position;
    }

    /// @brief Member access operator.
    /// @return Pointer to the current position.
    othello::Position *operator->() const noexcept {
        return &_node->position;
    }

    /// @brief Prefix increment operator moving to the previous position.
    /// @return Reference to the iterator.
    HistoryPositionIterator &operator++() noexcept {
        _node = _node->parent;
        return *this;
    }

    /// @brief Gets the past-the-end iterator.
    /// @return Past-the-end iterator.
    static HistoryPositionIterator end() noexcept {
        return HistoryPositionIterator(nullptr);
    }

private:
    othello::SearchNode *_node;
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

/// @brief Data for self-play.
///
struct SelfPlayData {
    /// @brief Legal actions of the current position.
    ///
    std::vector<int> actions;

    /// @brief Visit counts of the edges from the root node.
    ///
    std::vector<int> visit_counts;

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

    /// @brief Performs a Monte Carlo Tree Search.
    /// @tparam T Type of the neural network, which should be callable with a
    ///     torch::Tensor and return an othello::NeuralNetOutput.
    /// @param neural_net Neural network.
    /// @return Result of the search.
    template <typename T>
    MCTSResult search(T &&neural_net);

    /// @brief Performs a Monte Carlo Tree Search and returns the self-play
    ///     data.
    /// @tparam T Type of the neural network, which should be callable with a
    ///     torch::Tensor and return an othello::NeuralNetOutput.
    /// @param neural_net Neural network.
    /// @return Self-play data.
    template <typename T>
    SelfPlayData search_for_self_play(T &&neural_net);

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
    template <typename T>
    void _search_impl(T &&neural_net);

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
    _search_impl<T>(std::forward<T>(neural_net));

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

template <typename T>
othello::SelfPlayData othello::MCTS::search_for_self_play(T &&neural_net) {
    _search_impl<T>(std::forward<T>(neural_net));

    SelfPlayData data;

    data.actions = _search_tree->position.legal_actions();

    data.visit_counts.resize(_search_tree->children.size());
    int visit_count_sum = 0;
    for (std::size_t i = 0; i < _search_tree->children.size(); ++i) {
        data.visit_counts[i] = _search_tree->children[i]->visit_count;
        visit_count_sum += data.visit_counts[i];
    }
    float visit_count_divisor = visit_count_sum;
    if (visit_count_sum == 0) {
        visit_count_divisor = 1.0f;
    }

    data.features.resize(8);
    data.policy.resize(8);
    for (int transformation = 0; transformation < 8; ++transformation) {
        torch::Tensor features =
            torch::empty({1 + _history_size * 2, 8, 8}, torch::kFloat32);
        positions_to_features(
            HistoryPositionIterator(_search_tree.get()),
            HistoryPositionIterator::end(),
            features.data_ptr<float>(),
            _history_size,
            transformation
        );
        data.features[transformation] = features;

        torch::Tensor policy = torch::zeros({65}, torch::kFloat32);
        for (std::size_t i = 0; i < data.actions.size(); ++i) {
            int original_action = data.actions[i];
            int transformed_action =
                transform_action(original_action, transformation);
            policy.data_ptr<float>()[transformed_action] =
                data.visit_counts[i] / visit_count_divisor;
        }
        data.policy[transformation] = policy;
    }

    return data;
}

template <typename T>
void othello::MCTS::_search_impl(T &&neural_net) {
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
}

#endif // OTHELLO_MCTS_MCTS_H
