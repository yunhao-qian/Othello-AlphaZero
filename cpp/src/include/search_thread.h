#ifndef OTHELLO_MCTS_SEARCH_THREAD_H
#define OTHELLO_MCTS_SEARCH_THREAD_H

#include <mutex>
#include <random>

#include <torch/torch.h>

#include "queue.h"

namespace othello {

class SearchNode;
class MCTS;

struct NeuralNetOutput {
    torch::Tensor policy;
    torch::Tensor value;
};

struct NeuralNetInput {
    torch::Tensor features;
    Queue<NeuralNetOutput> *output_queue;
};

class SearchThread {
public:
    SearchThread(
        const MCTS *mcts,
        SearchNode *seach_tree,
        std::mutex *search_tree_mutex,
        Queue<NeuralNetInput> *neural_net_input_queue
    );

    void run();

private:
    void _simulate_batch();

    void _expand_and_backward(
        SearchNode *leaf, int transformation, float *policy, float *value
    );

    SearchNode *_choose_best_child(const SearchNode *node);

    const MCTS *_mcts;
    SearchNode *_search_tree;
    std::mutex *_search_tree_mutex;
    Queue<NeuralNetInput> *_neural_net_input_queue;
    Queue<NeuralNetOutput> _neural_net_output_queue;
    std::mt19937 _random_engine;
    std::gamma_distribution<float> _gamma_distribution;
    std::uniform_int_distribution<int> _transformation_distribution;
};

} // namespace othello

#endif // OTHELLO_MCTS_SEARCH_THREAD_H
