/// @file mcts.cpp
/// @brief Implementation of the Monte Carlo Tree Search algorithm.

#include "mcts.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "position_iterator.h"
#include "transformation.h"

othello::MCTS::MCTS(
    int history_size,
    const std::string &torch_device,
    bool torch_pin_memory,
    int num_simulations,
    int num_threads,
    int batch_size,
    float c_puct_base,
    float c_puct_init,
    float dirichlet_epsilon,
    float dirichlet_alpha
) {
    set_history_size(history_size);
    set_torch_device(torch_device);
    set_torch_pin_memory(torch_pin_memory);
    set_num_simulations(num_simulations);
    set_num_threads(num_threads);
    set_batch_size(batch_size);
    set_c_puct_base(c_puct_base);
    set_c_puct_init(c_puct_init);
    set_dirichlet_epsilon(dirichlet_epsilon);
    set_dirichlet_alpha(dirichlet_alpha);
    reset_position();
}

void othello::MCTS::reset_position() {
    _search_tree = std::make_unique<SearchNode>(Position::initial_position());
    _history.clear();
}

std::vector<int> othello::MCTS::visit_counts() {
    std::vector<int> counts;
    counts.reserve(_search_tree->children.size());
    for (auto &child : _search_tree->children) {
        counts.push_back(child->visit_count);
    }
    return counts;
}

std::vector<float> othello::MCTS::mean_action_values() {
    std::vector<float> values;
    values.reserve(_search_tree->children.size());
    for (auto &child : _search_tree->children) {
        values.push_back(child->mean_action_value);
    }
    return values;
}

othello::SelfPlayData othello::MCTS::self_play_data() {
    if (_search_tree->position.is_terminal()) {
        throw std::invalid_argument(
            "Self-play data cannot be generated from a terminal position."
        );
    }
    if (_search_tree->children.empty()) {
        throw std::invalid_argument("The root node has not been expanded yet.");
    }

    SelfPlayData data;
    data.features.reserve(8);
    data.policy.reserve(8);

    std::vector<int> legal_actions = _search_tree->position.legal_actions();

    int visit_count_sum = 0;
    for (auto &child : _search_tree->children) {
        visit_count_sum += child->visit_count;
    }
    if (visit_count_sum == 0) {
        visit_count_sum = 1;
    }

    for (int transformation = 0; transformation < 8; ++transformation) {
        torch::Tensor features =
            torch::empty({1 + _history_size * 2, 8, 8}, torch::kFloat32);
        positions_to_features(
            SearchNodePositionIterator(_search_tree.get()),
            SearchNodePositionIterator::end(),
            features.data_ptr<float>(),
            _history_size,
            transformation
        );
        data.features.push_back(features);

        torch::Tensor policy = torch::zeros({65}, torch::kFloat32);
        for (std::size_t i = 0; i < legal_actions.size(); ++i) {
            int original_action = legal_actions[i];
            int transformed_action =
                transform_action(original_action, transformation);
            policy.data_ptr<float>()[transformed_action] =
                static_cast<float>(_search_tree->children[i]->visit_count) /
                static_cast<float>(visit_count_sum);
        }
        data.policy.push_back(policy);
    }

    return data;
}

void othello::MCTS::apply_action(int action) {
    if (!(0 <= action && action < 65)) {
        throw std::out_of_range(
            "Expected 0 <= action < 65, but got " + std::to_string(action) + "."
        );
    }
    if (action == 64) {
        if (_search_tree->position.is_terminal()) {
            throw std::invalid_argument(
                "Pass is not allowed in a terminal position."
            );
        }
        if (_search_tree->position.legal_moves() != 0) {
            throw std::invalid_argument(
                "Pass is not allowed when there are legal moves."
            );
        }
    } else if (std::uint64_t move_mask = std::uint64_t(1) << (63 - action);
               (move_mask & _search_tree->position.legal_moves()) == 0) {
        throw std::invalid_argument(
            std::to_string(action) + " is not a legal action."
        );
    }

    if (_search_tree->children.empty()) {
        // The root node has not been expanded yet.
        _history.push_back(std::move(_search_tree));
        _search_tree = std::make_unique<SearchNode>(
            _history.back()->position.apply_action(action),
            _history.back().get()
        );
        return;
    }

    int child_index = 0;
    if (action != 0 && _search_tree->children.size() > 1) {
        // An uint64_t << 64 is undefined behavior, so we explicitly handle the
        // case where action == 0.
        std::uint64_t previous_moves_mask = ~std::uint64_t(0) << (64 - action);
        child_index = std::popcount(
            _search_tree->position.legal_moves() & previous_moves_mask
        );
    }
    std::unique_ptr<SearchNode> next_root =
        std::move(_search_tree->children[child_index]);

    // All nodes not belonging to the new root node are deleted, while all
    // ancestor nodes are moved to _history and remain valid.
    _search_tree->children.clear();
    _history.push_back(std::move(_search_tree));
    _search_tree = std::move(next_root);
}

void othello::MCTS::set_history_size(int value) {
    if (value < 1) {
        throw std::invalid_argument(
            "Expected history_size >= 1, but got " + std::to_string(value) + "."
        );
    }
    _history_size = value;
}

void othello::MCTS::set_num_simulations(int value) {
    if (value < 1) {
        throw std::invalid_argument(
            "Expected num_simulations >= 1, but got " + std::to_string(value) +
            "."
        );
    }
    _num_simulations = value;
}

void othello::MCTS::set_num_threads(int value) {
    if (value < 1) {
        throw std::invalid_argument(
            "Expected num_threads >= 1, but got " + std::to_string(value) + "."
        );
    }
    _num_threads = value;
}

void othello::MCTS::set_batch_size(int value) {
    if (value < 1) {
        throw std::invalid_argument(
            "Expected batch_size >= 1, but got " + std::to_string(value) + "."
        );
    }
    _batch_size = value;
}

void othello::MCTS::set_c_puct_base(float value) {
    if (!(value > 0.0f)) {
        throw std::invalid_argument(
            "Expected c_puct_base > 0.0, but got " + std::to_string(value) + "."
        );
    }
    _c_puct_base = value;
}

void othello::MCTS::set_c_puct_init(float value) {
    if (!(value >= 0.0f)) {
        throw std::invalid_argument(
            "Expected c_puct_init >= 0.0, but got " + std::to_string(value) +
            "."
        );
    }
    _c_puct_init = value;
}

void othello::MCTS::set_dirichlet_epsilon(float value) {
    if (!(0.0f <= value && value <= 1.0f)) {
        throw std::invalid_argument(
            "Expected 0.0 <= dirichlet_epsilon <= 1.0, but got " +
            std::to_string(value) + "."
        );
    }
    _dirichlet_epsilon = value;
}

void othello::MCTS::set_dirichlet_alpha(float value) {
    if (!(value >= 0.0f)) {
        throw std::invalid_argument(
            "Expected dirichlet_alpha >= 0.0, but got " +
            std::to_string(value) + "."
        );
    }
    _dirichlet_alpha = value;
}
