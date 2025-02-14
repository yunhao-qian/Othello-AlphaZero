/**
 * @file neural_net_input_thread.cpp
 * @brief Implementation of the neural network input thread function.
 */

#include "neural_net_input_thread.h"

#include <algorithm>
#include <cstddef>
#include <utility>

auto othello::neural_net_input_thread(
    ThreadSafeQueue<PositionEvaluation *> &input_queue,
    ThreadSafeQueue<InputBatch> &output_queue,
    const int history_size,
    const int batch_size,
    int num_evaluations
) -> void {
    InputBatch batch;
    const std::size_t num_input_features = batch_size * (history_size * 2 + 1) * 64;
    batch.input_features.reserve(num_input_features);
    batch.evaluations.reserve(batch_size);
    std::size_t effective_batch_size = std::min(batch_size, num_evaluations);

    while (true) {
        PositionEvaluation *const evaluation = input_queue.pop();
        if (evaluation == nullptr) {
            if (--num_evaluations == 0) {
                break;
            }
            effective_batch_size = std::min(batch_size, num_evaluations);
        } else {
            batch.input_features.insert(
                batch.input_features.end(),
                evaluation->input_features().cbegin(),
                evaluation->input_features().cend()
            );
            batch.evaluations.push_back(evaluation);
        }
        if (batch.evaluations.size() < effective_batch_size) {
            continue;
        }
        batch.input_features.resize(num_input_features);
        output_queue.emplace(std::move(batch));
        batch.input_features.clear();
        batch.evaluations.clear();
        batch.input_features.reserve(num_input_features);
        batch.evaluations.reserve(batch_size);
    }

    // Send an empty batch to signal the end of the input thread.
    batch.input_features.clear();
    batch.evaluations.clear();
    output_queue.emplace(std::move(batch));
}
