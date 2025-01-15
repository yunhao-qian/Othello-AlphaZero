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

struct SearchNode {
    Position position;
    SearchNode *parent = nullptr;
    std::vector<std::unique_ptr<SearchNode>> children = {};
    int visit_count = 0;
    float total_action_value = 0.0f;
    float mean_action_value = 0.0f;
    float prior_probability = 1.0f;
};

struct MCTSResult {
    std::vector<int> actions;
    std::vector<int> visit_counts;
    std::vector<float> mean_action_values;
};

class MCTS {
public:
    MCTS(
        int history_size = 4,
        const std::string &torch_device = "cpu",
        int num_simulations = 800,
        int num_threads = 2,
        int batch_size = 16,
        float exploration_weight = 1.0f,
        float dirichlet_epsilon = 0.25f,
        float dirichlet_alpha = 0.5f
    );

    void reset_position();

    Position position() const noexcept {
        return _search_tree->position;
    }

    template <typename T>
    MCTSResult search(T &&neural_net);

    void apply_action(int action);

    int history_size() const noexcept {
        return _history_size;
    }

    void set_history_size(int value);

    std::string torch_device() const {
        return _torch_device;
    }

    void set_torch_device(const std::string &value) {
        _torch_device = value;
    }

    int num_simulations() const noexcept {
        return _num_simulations;
    }

    void set_num_simulations(int value);

    int num_threads() const noexcept {
        return _num_threads;
    }

    void set_num_threads(int value);

    int batch_size() const noexcept {
        return _batch_size;
    }

    void set_batch_size(int value);

    float exploration_weight() const noexcept {
        return _exploration_weight;
    }

    void set_exploration_weight(float value);

    float dirichlet_epsilon() const noexcept {
        return _dirichlet_epsilon;
    }

    void set_dirichlet_epsilon(float value);

    float dirichlet_alpha() const noexcept {
        return _dirichlet_alpha;
    }

    void set_dirichlet_alpha(float value);

private:
    int _history_size;
    std::string _torch_device;
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
