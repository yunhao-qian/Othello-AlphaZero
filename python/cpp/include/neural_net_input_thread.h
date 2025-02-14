/**
 * @file neural_net_input_thread.h
 * @brief Neural network input thread function.
 */

#ifndef OTHELLO_ALPHAZERO_NEURAL_NET_INPUT_THREAD_H
#define OTHELLO_ALPHAZERO_NEURAL_NET_INPUT_THREAD_H

#include <vector>

#include "position_evaluation.h"
#include "thread_safe_queue.h"

namespace othello {

/**
 * @brief Structure for a batch of input data to the neural network.
 */
struct InputBatch {
    /**
     * @brief Input features of shape `(batch_size, feature_channels, 8, 8)`.
     */
    std::vector<float> input_features;

    /**
     * @brief Position evaluations corresponding to the input features.
     */
    std::vector<PositionEvaluation *> evaluations;
};

/**
 * @brief Runs the neural network input thread.
 * @param input_queue Queue of individual position evaluations.
 * @param output_queue Queue of batches of input data.
 * @param history_size Number of history positions to include in the input features.
 * @param batch_size Number of positions to evaluate in each batch.
 * @param num_evaluations Total number of running position evaluations.
 */
void neural_net_input_thread(
    ThreadSafeQueue<PositionEvaluation *> &input_queue,
    ThreadSafeQueue<InputBatch> &output_queue,
    int history_size,
    int batch_size,
    int num_evaluations
);

}  // namespace othello

#endif  // OTHELLO_ALPHAZERO_NEURAL_NET_INPUT_THREAD_H
