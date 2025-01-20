/// @file search_thread.cpp
/// @brief Implementation of the search thread.

#include "search_thread.h"

#include <cmath>

#include "mcts.h"
#include "position_iterator.h"
#include "transformation.h"

othello::SearchThread::SearchThread(
    const MCTS *mcts,
    SearchNode *seach_tree,
    std::mutex *search_tree_mutex,
    Queue<NeuralNetInput> *neural_net_input_queue
)
    : _mcts(mcts),
      _search_tree(seach_tree),
      _search_tree_mutex(search_tree_mutex),
      _neural_net_input_queue(neural_net_input_queue),
      _random_engine(std::random_device()()),
      _gamma_distribution(mcts->dirichlet_alpha(), 1.0f),
      _transformation_distribution(0, 7),
      _leaves(mcts->batch_size()),
      _transformations(mcts->batch_size()) {

    torch::TensorOptions cpu_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (mcts->torch_pin_memory()) {
        cpu_options = cpu_options.pinned_memory(true);
    }
    torch::TensorOptions device_options = torch::TensorOptions()
                                              .dtype(torch::kFloat32)
                                              .device(mcts->torch_device());

    _features_cpu = torch::zeros(
        {mcts->batch_size(), 1 + mcts->history_size() * 2, 8, 8}, cpu_options
    );
    _features_device = torch::empty(
        {mcts->batch_size(), 1 + mcts->history_size() * 2, 8, 8}, device_options
    );
    _policy_cpu = torch::empty({mcts->batch_size(), 65}, cpu_options);
    _value_cpu = torch::empty({mcts->batch_size()}, cpu_options);
}

void othello::SearchThread::run() {
    int batch_size = _mcts->num_threads() * _mcts->batch_size();
    int num_simulations =
        (_mcts->num_simulations() + batch_size - 1) / batch_size;
    for (int i = 0; i < num_simulations; ++i) {
        _simulate_batch();
    }
    _neural_net_input_queue->push(
        NeuralNetInput{.features = _features_device, .output_queue = nullptr}
    );
}

void othello::SearchThread::_simulate_batch() {
    _search_tree_mutex->lock();

    for (int i = 0; i < _mcts->batch_size(); ++i) {
        SearchNode *node;
        for (node = _search_tree;
             !(node->position.is_terminal() || node->children.empty());
             node = _choose_best_child(node)) {
        }
        _leaves[i] = node;
        // Use virtual losses to ensure each thread evaluates different nodes.
        for (SearchNode *child = node; child != _search_tree;
             child = child->parent) {
            child->visit_count += 1;
            child->total_action_value -= 1.0f;
            child->mean_action_value =
                child->total_action_value / child->visit_count;
        }
        // The root node visit count is used for computing exploration rates.
        _search_tree->visit_count += 1;
    }

    _search_tree_mutex->unlock();

    bool all_terminal = true;
    float *features = _features_cpu.data_ptr<float>();
    int num_features = (1 + _mcts->history_size() * 2) * 64;

    for (int i = 0; i < _mcts->batch_size(); ++i) {
        if (_leaves[i]->position.is_terminal()) {
            continue;
        }
        all_terminal = false;
        _transformations[i] = _transformation_distribution(_random_engine);
        positions_to_features(
            SearchNodePositionIterator(_leaves[i]),
            SearchNodePositionIterator::end(),
            features + i * num_features,
            _mcts->history_size(),
            _transformations[i]
        );
    }

    if (!all_terminal) {
        _features_device.copy_(_features_cpu);
        _neural_net_input_queue->push(NeuralNetInput{
            .features = _features_device,
            .output_queue = &_neural_net_output_queue
        });
        NeuralNetOutput output = _neural_net_output_queue.pop();
        _policy_cpu.copy_(output.policy);
        _value_cpu.copy_(output.value);
    }

    float *policy_data = _policy_cpu.data_ptr<float>();
    float *value_data = _value_cpu.data_ptr<float>();

    _search_tree_mutex->lock();

    for (int i = 0; i < _mcts->batch_size(); ++i) {
        _expand_and_backward(
            _leaves[i],
            _transformations[i],
            policy_data + i * 65,
            value_data + i
        );
    }

    _search_tree_mutex->unlock();
}

void othello::SearchThread::_expand_and_backward(
    othello::SearchNode *leaf, int transformation, float *policy, float *value
) {
    // There is a small chance that the leaf node has already been expanded by
    // another thread, in which case we should not expand it again.
    if (!leaf->position.is_terminal() && leaf->children.empty()) {
        std::vector<int> legal_actions = leaf->position.legal_actions();
        leaf->children.reserve(legal_actions.size());
        for (int original_action : legal_actions) {
            int transformed_action =
                transform_action(original_action, transformation);
            leaf->children.push_back(std::make_unique<SearchNode>(
                leaf->position.apply_action(original_action), // position
                leaf,                                         // parent
                std::vector<std::unique_ptr<SearchNode>>(),   // children
                0,                                            // visit_count
                0.0f,                      // total_action_value
                0.0f,                      // mean_action_value
                policy[transformed_action] // prior_probability
            ));
        }
    }

    float action_value;
    // The action-value is with respect to the parent node, so the sign should
    // be flipped.
    if (!leaf->position.is_terminal()) {
        action_value = -value[0];
    } else {
        std::uint64_t player_discs;
        std::uint64_t opponent_discs;
        if (leaf->parent->position.player() == 1) {
            player_discs = leaf->position.player1_discs();
            opponent_discs = leaf->position.player2_discs();
        } else {
            player_discs = leaf->position.player2_discs();
            opponent_discs = leaf->position.player1_discs();
        }
        int player_count = std::popcount(player_discs);
        int opponent_count = std::popcount(opponent_discs);
        if (player_count > opponent_count) {
            action_value = 1.0f;
        } else if (player_count < opponent_count) {
            action_value = -1.0f;
        } else {
            action_value = 0.0f;
        }
    }

    // Backward pass to update the visit counts and action-values.
    for (SearchNode *child = leaf; child != _search_tree;
         child = child->parent) {
        // visit_count has already been incremented by the virtual loss.
        // +1.0 to cancel the virtual loss.
        child->total_action_value += 1.0f + action_value;
        child->mean_action_value =
            child->total_action_value / child->visit_count;

        action_value = -action_value;
    }
}

othello::SearchNode *
othello::SearchThread::_choose_best_child(const SearchNode *node) {
    if (node->children.size() == 1) {
        return node->children.front().get();
    }

    float exploration_rate =
        std::log(
            (1 + node->visit_count + _mcts->c_puct_base()) /
            _mcts->c_puct_base()
        ) +
        _mcts->c_puct_init();
    int total_visit_count = 0;
    for (auto &child : node->children) {
        total_visit_count += child->visit_count;
    }
    float ucb_multiplier =
        exploration_rate * std::sqrt(static_cast<float>(total_visit_count));

    if (!(node == _search_tree && _mcts->dirichlet_epsilon() > 0.0f)) {
        auto get_ucb = [ucb_multiplier](const auto &child) {
            return child->mean_action_value + ucb_multiplier *
                                                  child->prior_probability /
                                                  (1.0f + child->visit_count);
        };
        SearchNode *best_child = node->children.front().get();
        float best_ucb = get_ucb(best_child);
        for (std::size_t i = 1; i < node->children.size(); ++i) {
            SearchNode *child = node->children[i].get();
            float ucb = get_ucb(child);
            if (ucb > best_ucb) {
                best_child = child;
                best_ucb = ucb;
            }
        }
        return best_child;
    }

    std::vector<float> noises(node->children.size());
    float noise_sum = 0.0f;
    for (float &noise : noises) {
        noise = _gamma_distribution(_random_engine);
        noise_sum += noise;
    }
    if (noise_sum == 0.0f) {
        noise_sum = 1.0f;
    }
    auto get_ucb = [&children(node->children),
                    &noises,
                    probability_multiplier(1.0f - _mcts->dirichlet_epsilon()),
                    noise_multiplier(_mcts->dirichlet_epsilon() / noise_sum),
                    ucb_multiplier](std::size_t i) {
        auto &child = children[i];
        float probability = child->prior_probability * probability_multiplier +
                            noises[i] * noise_multiplier;
        return child->mean_action_value +
               ucb_multiplier * probability / (1.0f + child->visit_count);
    };
    std::size_t best_child_index = 0;
    float best_ucb = get_ucb(0);
    for (std::size_t i = 1; i < node->children.size(); ++i) {
        float ucb = get_ucb(i);
        if (ucb > best_ucb) {
            best_child_index = i;
            best_ucb = ucb;
        }
    }
    return node->children[best_child_index].get();
}
