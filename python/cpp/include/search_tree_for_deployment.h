/**
 * @file search_tree_for_deployment.h
 * @brief Search tree for deployment class.
 */

#ifndef OTHELLO_ALPHAZERO_SEARCH_TREE_FOR_DEPLOYMENT_H
#define OTHELLO_ALPHAZERO_SEARCH_TREE_FOR_DEPLOYMENT_H

#include <mutex>

#include "position_evaluation.h"
#include "search_tree_base.h"
#include "thread_safe_queue.h"

namespace othello {

/**
 * @brief Search tree for deployment.
 */
class SearchTreeForDeployment : public SearchTreeBase {
public:
    using SearchTreeBase::SearchTreeBase;

    /**
     * @brief Runs the forward selection phase and sets up a neural network evaluation if
     *     applicable.
     * @tparam VirtualLoss Whether virtual losses are used.
     * @param evaluation Position evaluation object.
     * @param queue Queue to push the position evaluation object to.
     * @param random_engine Random engine for sampling transformations.
     */
    template <bool VirtualLoss>
    void forward_and_evaluate(
        PositionEvaluation &evaluation,
        ThreadSafeQueue<PositionEvaluation *> &queue,
        std::mt19937 &random_engine
    );

    /**
     * @brief Expands a leaf node if applicable and runs the backward pass.
     * @tparam VirtualLoss Whether virtual losses are used.
     * @param evaluation Position evaluation object.
     */
    template <bool VirtualLoss>
    void expand_and_backward(PositionEvaluation &evaluation);

private:
    friend class SearchTreeBase;

    std::unique_lock<std::mutex> lock_search_tree() {
        return std::unique_lock<std::mutex>(m_search_tree_mutex);
    }

    std::mutex m_search_tree_mutex;
};

}  // namespace othello

template <bool VirtualLoss>
void othello::SearchTreeForDeployment::forward_and_evaluate(
    PositionEvaluation &evaluation,
    ThreadSafeQueue<PositionEvaluation *> &queue,
    std::mt19937 &random_engine
) {
    forward_and_evaluate_impl<SearchTreeForDeployment, false, VirtualLoss>(
        evaluation, queue, random_engine
    );
}

template <bool VirtualLoss>
void othello::SearchTreeForDeployment::expand_and_backward(PositionEvaluation &evaluation) {
    expand_and_backward_impl<SearchTreeForDeployment, VirtualLoss>(evaluation);
}

#endif  // OTHELLO_ALPHAZERO_SEARCH_TREE_FOR_DEPLOYMENT_H
