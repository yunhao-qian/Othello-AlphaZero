/// @file mcts.cpp
/// @brief Implementation of the Monte Carlo Tree Search algorithm.

#include "mcts.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>

#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;
using namespace py::literals;

othello::SearchThread::SearchThread(MCTS &mcts, int thread_id)
    : _mcts(&mcts),
      _thread_id(thread_id),
      _random_engine(std::random_device()()),
      _gamma_distribution(mcts.dirichlet_alpha(), 1.0f) {}

void othello::SearchThread::run() {
    int num_simulations =
        (_mcts->num_simulations() + _mcts->num_threads() - 1) /
        _mcts->num_threads();
    for (int i = 0; i < num_simulations; ++i) {
        _simulate();
    }
    _mcts->_neural_net_input_queue.push(NeuralNetInput{_thread_id, {}});
}

void othello::SearchThread::_simulate() {
    std::vector<unsigned> &search_path = _search_path;
    search_path.clear();

    std::vector<SearchNode> &search_tree = _mcts->_search_tree;

    std::mutex &search_tree_mutex = _mcts->_search_tree_mutex;
    search_tree_mutex.lock();

    for (unsigned node_index = 0;;
         node_index = _choose_best_child(node_index)) {
        search_path.push_back(node_index);
        if (SearchNode &node = search_tree[node_index];
            node.position.is_terminal() || node.children.empty()) {
            // It is a terminal position or the node has not been expanded yet.
            break;
        }
    }

    unsigned leaf_index = search_path.back();
    Position leaf_position = search_tree[leaf_index].position;

    float action_value;
    int visit_count_increment;
    float total_action_value_offset;

    if (leaf_position.is_terminal()) {
        visit_count_increment = 1;
        total_action_value_offset = 0.0f;

        // If the game is over, we do not need neural network evaluation.
        action_value = leaf_position.action_value();
    } else {
        visit_count_increment = 0;
        total_action_value_offset = 1.0f;

        // Use virtual losses to ensure each thread evaluates different nodes.
        for (unsigned child_index : search_path) {
            // The root node do not need to be updated, but it does not matter.
            SearchNode &child = search_tree[child_index];
            child.visit_count += 1;
            child.total_action_value -= 1.0f;
            child.mean_action_value =
                child.total_action_value / child.visit_count;
        }

        search_tree_mutex.unlock();

        _mcts->_neural_net_input_queue.push(
            NeuralNetInput{_thread_id, leaf_position.to_features()}
        );
        NeuralNetOutput neural_net_output = _neural_net_output_queue.pop();

        search_tree_mutex.lock();

        // There is a small chance that the leaf node has already been expanded
        // by another thread, in which case we should not overwrite the
        // children.
        if (std::vector<unsigned> *leaf_children =
                &search_tree[leaf_index].children;
            leaf_children->empty()) {
            std::vector<int> legal_actions = leaf_position.legal_actions();
            leaf_children->reserve(legal_actions.size());
            for (int action : legal_actions) {
                leaf_children->push_back(static_cast<unsigned>(search_tree.size(
                )));
                search_tree.emplace_back(
                    leaf_position.apply_action(action),
                    action,
                    0,                               // visit_count
                    0.0f,                            // total_action_value
                    0.0f,                            // mean_action_value
                    neural_net_output.policy[action] // prior_probability
                );

                // The leaf_children pointer may be invalidated by the
                // reallocation of the search_tree vector.
                leaf_children = &search_tree[leaf_index].children;
            }
        }

        action_value = neural_net_output.value;
        if (leaf_position.player() != 1) {
            action_value = -action_value;
        }
    }

    // Backward pass to update the visit counts and action-values.
    for (unsigned child_index : search_path) {
        SearchNode &child = search_tree[child_index];
        child.visit_count += visit_count_increment;
        child.total_action_value +=
            total_action_value_offset +
            (child.position.player() == 1 ? action_value : -action_value);
        child.mean_action_value = child.total_action_value / child.visit_count;
    }

    search_tree_mutex.unlock();
}

unsigned othello::SearchThread::_choose_best_child(unsigned node_index) {
    std::vector<unsigned> &children = _mcts->_search_tree[node_index].children;
    if (children.size() == 1) {
        return children.front();
    }

    int sum_visit_count = 0;
    for (unsigned child_index : children) {
        sum_visit_count += _mcts->_search_tree[child_index].visit_count;
    }
    float sqrt_sum_visit_count = std::sqrt(static_cast<float>(sum_visit_count));

    bool is_exploration = node_index == 0 && _mcts->_dirichlet_epsilon > 0.0f;
    // UCB is at least the minimum action-value, which is -1.
    float best_ucb = -10.0f;
    unsigned best_child_index = children.front();

    for (unsigned child_index : children) {
        SearchNode &child = _mcts->_search_tree[child_index];
        float probability = child.prior_probability;
        if (is_exploration) {
            float noise = _gamma_distribution(_random_engine);
            probability *= 1.0f - _mcts->_dirichlet_epsilon;
            probability += _mcts->_dirichlet_epsilon * noise;
        }
        float ucb = child.mean_action_value +
                    _mcts->_exploration_weight * probability *
                        sqrt_sum_visit_count / (1.0f + child.visit_count);
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_child_index = child_index;
        }
    }

    return best_child_index;
}

othello::MCTS::MCTS(
    const std::string &torch_device,
    int num_simulations,
    int batch_size,
    int num_threads,
    float exploration_weight,
    float dirichlet_epsilon,
    float dirichlet_alpha
) {
    set_torch_device(torch_device);
    set_num_simulations(num_simulations);
    set_batch_size(batch_size);
    set_num_threads(num_threads);
    set_exploration_weight(exploration_weight);
    set_dirichlet_epsilon(dirichlet_epsilon);
    set_dirichlet_alpha(dirichlet_alpha);
    reset_position(Position::initial_position());
}

void othello::MCTS::reset_position(const Position &position) {
    _search_tree.clear();
    _search_tree.emplace_back(Position::initial_position());
}

othello::Position othello::MCTS::root_position() const {
    return _search_tree.front().position;
}

py::object othello::MCTS::search(py::object neural_net) {
    _neural_net_input_queue.clear();
    _batched_neural_net_input_queue.clear();
    _batched_neural_net_output_queue.clear();
    _search_threads.clear();
    _search_threads.reserve(_num_threads);

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(2 + _num_threads));

    threads.emplace_back(&MCTS::_neural_net_input_thread, this);
    threads.emplace_back(&MCTS::_neural_net_output_thread, this);
    for (int thread_id = 0; thread_id < _num_threads; ++thread_id) {
        _search_threads.push_back(
            std::make_unique<SearchThread>(*this, thread_id)
        );
        threads.emplace_back(&SearchThread::run, _search_threads.back().get());
    }

    while (true) {
        BatchedNeuralNetInput batched_input =
            _batched_neural_net_input_queue.pop();
        if (batched_input.thread_ids.empty()) {
            // All search threads have finished. Propagate the signal to the
            // neural network output thread.
            _batched_neural_net_output_queue.push(BatchedNeuralNetOutput{
                {},
                torch::empty({0, 65}, torch::kFloat32),
                torch::empty({0}, torch::kFloat32)
            });
            break;
        }
        py::tuple policies_and_values =
            neural_net(py::cast(batched_input.features));
        torch::Tensor policies = policies_and_values[0].cast<torch::Tensor>();
        torch::Tensor values = policies_and_values[1].cast<torch::Tensor>();
        // Without detach(), the tensors will hold references to Python objects
        // and lead to unexpected GIL acquisitions.
        _batched_neural_net_output_queue.push(BatchedNeuralNetOutput{
            std::move(batched_input.thread_ids),
            policies.detach(),
            values.detach()
        });
    }

    for (std::thread &thread : threads) {
        thread.join();
    }

    _neural_net_input_queue.clear();
    _batched_neural_net_input_queue.clear();
    _batched_neural_net_output_queue.clear();
    _search_threads.clear();

    std::vector<int> actions;
    std::vector<int> visit_counts;
    std::vector<float> mean_action_values;

    std::vector<unsigned> &root_children = _search_tree.front().children;
    actions.reserve(root_children.size());
    visit_counts.reserve(root_children.size());
    mean_action_values.reserve(root_children.size());

    for (unsigned child_index : root_children) {
        SearchNode &child = _search_tree[child_index];
        actions.push_back(child.previous_action);
        visit_counts.push_back(child.visit_count);
        mean_action_values.push_back(child.mean_action_value);
    }

    return py::dict(
        "actions"_a = actions,
        "visit_counts"_a = visit_counts,
        "mean_action_values"_a = mean_action_values
    );
}

namespace {

using othello::SearchNode;

/// @brief Collects the subtree rooted at the given node from the old tree to
///     the new tree.
/// @param old_tree Original search tree.
/// @param new_tree New search tree.
/// @param old_index Index of the root node in the old tree.
/// @return Index of the root node in the new tree.
unsigned collect_subtree(
    const std::vector<SearchNode> &old_tree,
    std::vector<SearchNode> &new_tree,
    unsigned old_index
) {
    const SearchNode &old_node = old_tree[old_index];
    unsigned new_index = static_cast<unsigned>(new_tree.size());
    new_tree.emplace_back(
        old_node.position,
        old_node.previous_action,
        old_node.visit_count,
        old_node.total_action_value,
        old_node.mean_action_value,
        old_node.prior_probability
    );
    // This reference will not be invalidated because we reserve the capacity
    // of the new search tree.
    std::vector<unsigned> &new_children = new_tree.back().children;
    new_children.reserve(old_node.children.size());
    for (unsigned old_child_index : old_node.children) {
        new_children.push_back(
            collect_subtree(old_tree, new_tree, old_child_index)
        );
    }
    return new_index;
}

/// @brief Prunes the search tree to a new root node.
/// @param tree Original search tree.
/// @param new_root_index Index of the new root node.
/// @return Pruned search tree.
std::vector<SearchNode>
prune_search_tree(const std::vector<SearchNode> &tree, int new_root_index) {
    std::vector<SearchNode> new_tree;
    new_tree.reserve(tree.size());
    collect_subtree(tree, new_tree, new_root_index);
    return new_tree;
}

} // namespace

void othello::MCTS::apply_action(int action) {
    SearchNode &root = _search_tree.front();

    std::vector<int> legal_actions = root.position.legal_actions();
    if (std::find(legal_actions.begin(), legal_actions.end(), action) ==
        legal_actions.end()) {
        return;
    }

    if (root.children.empty()) {
        // The root node has not been expanded yet.
        Position next_position = root.position.apply_action(action);
        _search_tree.clear();
        _search_tree.emplace_back(next_position, action);
        return;
    }

    unsigned child_index = *std::find_if(
        root.children.begin(),
        root.children.end(),
        [this, action](unsigned child_index) {
            return _search_tree[child_index].previous_action == action;
        }
    );
    _search_tree = ::prune_search_tree(_search_tree, child_index);
}

void othello::MCTS::set_num_simulations(int value) noexcept {
    value = std::max(1, value);
    _num_simulations = value;
}

void othello::MCTS::set_batch_size(int value) noexcept {
    value = std::max(1, value);
    _batch_size = value;
}

void othello::MCTS::set_num_threads(int value) noexcept {
    value = std::max(1, value);
    _num_threads = value;
}

void othello::MCTS::set_exploration_weight(float value) noexcept {
    if (!std::isfinite(value)) {
        value = 0.0f;
    } else {
        value = std::max(0.0f, value);
    }
    _exploration_weight = value;
}

void othello::MCTS::set_dirichlet_epsilon(float value) noexcept {
    if (!std::isfinite(value)) {
        value = 0.0f;
    } else {
        value = std::clamp(value, 0.0f, 1.0f);
    }
    _dirichlet_epsilon = value;
}

void othello::MCTS::set_dirichlet_alpha(float value) noexcept {
    if (!std::isfinite(value)) {
        value = 0.0f;
    } else {
        value = std::max(0.0f, value);
    }
    _dirichlet_alpha = value;
}

void othello::MCTS::_neural_net_input_thread() {
    int max_batch_size = std::min(_batch_size, _num_threads);

    std::vector<int> thread_ids;
    thread_ids.reserve(max_batch_size);

    std::vector<float> features;
    features.reserve(max_batch_size * 3 * 8 * 8);

    int num_running_threads = _num_threads;
    int actual_batch_size = max_batch_size;

    torch::Device device(_torch_device);

    while (true) {
        NeuralNetInput input = _neural_net_input_queue.pop();
        if (input.features.empty()) {
            // An empty feature vector signals the end of the search thread.
            if (--num_running_threads == 0) {
                break;
            }
            actual_batch_size = std::min(_batch_size, num_running_threads);
        } else {
            thread_ids.push_back(input.thread_id);
            features.insert(
                features.end(), input.features.begin(), input.features.end()
            );
        }

        if (static_cast<int>(thread_ids.size()) < actual_batch_size) {
            continue;
        }

        // Some compiled models expect fix-sized input tensors, so we always pad
        // the input to the maximum batch size.
        thread_ids.resize(max_batch_size, -1);
        features.resize(max_batch_size * 3 * 8 * 8, 0.0f);

        torch::Tensor feature_tensor = torch::from_blob(
            features.data(), {max_batch_size, 3, 8, 8}, torch::kFloat32
        );
        feature_tensor = feature_tensor.to(
            device,          // device
            torch::kFloat32, // dtype
            false,           // non_blocking
            true             // copy
        );
        _batched_neural_net_input_queue.push(
            BatchedNeuralNetInput{thread_ids, std::move(feature_tensor)}
        );

        thread_ids.clear();
        features.clear();
    }

    // Put an empty batch to signal the end of the thread.
    _batched_neural_net_input_queue.push(
        BatchedNeuralNetInput{{}, torch::empty({0, 3, 8, 8}, torch::kFloat32)}
    );
}

void othello::MCTS::_neural_net_output_thread() {
    while (true) {
        BatchedNeuralNetOutput batched_output =
            _batched_neural_net_output_queue.pop();
        if (batched_output.thread_ids.empty()) {
            // An empty thread ID vector indicates that all search threads have
            // finished.
            break;
        }

        torch::Tensor policies =
            batched_output.policies.to(torch::kCPU, torch::kFloat32)
                .contiguous();
        torch::Tensor values =
            batched_output.values.to(torch::kCPU, torch::kFloat32).contiguous();
        float *policy_data = policies.data_ptr<float>();
        float *value_data = values.data_ptr<float>();

        for (int thread_id : batched_output.thread_ids) {
            if (thread_id < 0) {
                // Skip dummy threads for padding.
                continue;
            }

            float *policy_data_end = policy_data + 65;
            std::vector<float> policy(policy_data, policy_data_end);
            policy_data = policy_data_end;

            float value = *value_data++;

            _search_threads[thread_id]->_neural_net_output_queue.push(
                NeuralNetOutput{std::move(policy), value}
            );
        }
    }
}
