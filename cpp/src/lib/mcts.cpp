/// @file mcts.cpp
/// @brief Implementation of the Monte Carlo Tree Search algorithm.

#include "mcts.h"

#include <algorithm>
#include <cmath>

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
    _mcts->_neural_net_input_queue.push(NeuralNetInput{
        _thread_id, // thread_id
        true,       // is_finished
        {}          // features
    });
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

        _mcts->_neural_net_input_queue.push(NeuralNetInput{
            _thread_id,                 // thread_id
            false,                      // is_finished
            leaf_position.to_features() // features
        });
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

    int sum_visit_count = 0;
    for (unsigned child_index : children) {
        sum_visit_count += _mcts->_search_tree[child_index].visit_count;
    }
    float sqrt_sum_visit_count = std::sqrt(static_cast<float>(sum_visit_count));

    bool is_exploration = node_index == 0 && _mcts->_dirichlet_epsilon > 0.0f;
    // UCB is at least the minimum action-value, which is -1.
    float best_ucb = -10.0f;
    unsigned best_child_index = 0;

    for (unsigned child_index : children) {
        const SearchNode &child = _mcts->_search_tree[child_index];
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
    int num_simulations,
    int batch_size,
    int num_threads,
    float exploration_weight,
    float dirichlet_epsilon,
    float dirichlet_alpha
) {
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
