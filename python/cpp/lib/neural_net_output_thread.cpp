#include "neural_net_output_thread.h"

auto othello::neural_network_output_thread(ThreadSafeQueue<OutputBatch> &queue) -> void {
    while (true) {
        OutputBatch batch = queue.pop();
        if (batch.evaluations.empty()) {
            // An empty batch is a signal to stop the thread.
            break;
        }
        const float *policy_data = batch.policy.data();
        const float *value_data = batch.value.data();
        for (PositionEvaluation *const evaluation : batch.evaluations) {
            evaluation->set_result(policy_data, *value_data);
            policy_data += 65;
            ++value_data;
        }
    }
}
