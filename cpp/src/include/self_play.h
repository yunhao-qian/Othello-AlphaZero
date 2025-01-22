/// @file self_play.h
/// @brief TODO

#ifndef OTHELLO_MCTS_SELF_PLAY_H
#define OTHELLO_MCTS_SELF_PLAY_H

#include <atomic>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "neural_net.h"
#include "queue.h"
#include "search_node.h"

namespace othello {

class SelfPlay {
public:
    SelfPlay(
        int history_size = 4,
        const std::string &torch_device = "cpu",
        bool torch_pin_memory = false,
        int num_threads = 16,
        int num_simulations = 800,
        float c_puct_base = 20000.0f,
        float c_puct_init = 2.5f,
        float dirichlet_epsilon = 0.25f,
        float dirichlet_alpha = 0.5f
    );

    int history_size() const noexcept {
        return _history_size;
    }

    std::string torch_device() const {
        return _torch_device;
    }

    bool torch_pin_memory() const noexcept {
        return _torch_pin_memory;
    }

    int num_threads() const noexcept {
        return _num_threads;
    }

    int num_simulations() const noexcept {
        return _num_simulations;
    }

    float c_puct_base() const noexcept {
        return _c_puct_base;
    }

    float c_puct_init() const noexcept {
        return _c_puct_init;
    }

    float dirichlet_epsilon() const noexcept {
        return _dirichlet_epsilon;
    }

    float dirichlet_alpha() const noexcept {
        return _dirichlet_alpha;
    }

private:
    int _history_size;
    std::string _torch_device;
    bool _torch_pin_memory;
    int _num_threads;
    int _num_simulations;
    float _c_puct_base;
    float _c_puct_init;
    float _dirichlet_epsilon;
    float _dirichlet_alpha;
};

struct SelfPlayNeuralNetInput {
    torch::Tensor features;
    Queue<NeuralNetOutput> *output_queue;
};

struct SelfPlayData {
    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> policies;
    std::vector<torch::Tensor> values;
};

class SelfPlayThread {
public:
    SelfPlayThread(
        const SelfPlay *self_play,
        std::atomic<int> *num_remaining_games,
        Queue<SelfPlayNeuralNetInput> *neural_net_input_queue,
        Queue<SelfPlayData> *self_play_data_queue
    );

    void run();
};

} // namespace othello

#endif // OTHELLO_MCTS_SELF_PLAY_H
