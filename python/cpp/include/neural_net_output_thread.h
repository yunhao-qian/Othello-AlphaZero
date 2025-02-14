/**
 * @file neural_net_output_thread.h
 * @brief Neural network output thread function.
 */

#ifndef OTHELLO_ALPHAZERO_NEURAL_NET_OUTPUT_THREAD_H
#define OTHELLO_ALPHAZERO_NEURAL_NET_OUTPUT_THREAD_H

#include <vector>

#include "position_evaluation.h"
#include "thread_safe_queue.h"

namespace othello {

/**
 * @brief Structure for a batch of output data from the neural network.
 */
struct OutputBatch {
    /**
     * @brief Policy tensor of shape `(batch_size, 65)`.
     */
    std::vector<float> policy;

    /**
     * @brief Value tensor of shape `(batch_size,)`.
     */
    std::vector<float> value;

    /**
     * @brief Position evaluations corresponding to the output data.
     */
    std::vector<PositionEvaluation *> evaluations;
};

/**
 * @brief Runs the neural network output thread.
 * @param queue Queue of batches of output data.
 */
void neural_network_output_thread(ThreadSafeQueue<OutputBatch> &queue);

}  // namespace othello

#endif  // OTHELLO_ALPHAZERO_NEURAL_NET_OUTPUT_THREAD_H
