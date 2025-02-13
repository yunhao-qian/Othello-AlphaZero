/**
 * @file search_tree_for_self_play.h
 * @brief Search tree for self-play class.
 */

#ifndef OTHELLO_ALPHAZERO_SEARCH_TREE_FOR_SELF_PLAY_H
#define OTHELLO_ALPHAZERO_SEARCH_TREE_FOR_SELF_PLAY_H

#include <array>
#include <cstddef>
#include <random>
#include <tuple>

#include "search_tree_base.h"

namespace othello {

/**
 * @brief Search tree for self-play.
 */
class SearchTreeForSelfPlay : public SearchTreeBase {
public:
    /**
     * @brief Constructs a search tree.
     * @param c_puct_init `c_init` for the PUCT formula.
     * @param c_puct_base `c_base` for the PUCT formula.
     * @param dirichlet_epsilon Epsilon for the Dirichlet noise.
     * @param dirichlet_alpha Alpha for the Dirichlet noise.
     */
    SearchTreeForSelfPlay(
        float c_puct_init, float c_puct_base, float dirichlet_epsilon, float dirichlet_alpha
    );

    /**
     * @brief Gets the epsilon for the Dirichlet noise.
     * @return Epsilon for the Dirichlet noise.
     */
    float dirichlet_epsilon() const noexcept {
        return m_dirichlet_epsilon;
    }

    /**
     * @brief Sets the epsilon for the Dirichlet noise.
     * @param value Epsilon for the Dirichlet noise.
     */
    void set_dirichlet_epsilon(float value);

    /**
     * @brief Gets the alpha for the Dirichlet noise.
     * @return Alpha for the Dirichlet noise.
     */
    float dirichlet_alpha() const noexcept {
        return m_dirichlet_alpha;
    }

    /**
     * @brief Sets the alpha for the Dirichlet noise.
     * @param value Alpha for the Dirichlet noise.
     */
    void set_dirichlet_alpha(float value);

    /**
     * @brief Runs the forward selection phase and sets up a neural network evaluation if
     *     applicable.
     * @tparam VirtualLoss Whether virtual losses are used.
     * @param evaluation Position evaluation object.
     * @param queue Queue to push the position evaluation object to.
     * @param random_engine Random engine for sampling Dirichlet noises and transformations.
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

    std::tuple<> lock_search_tree() {
        // In self-play, we do not search the tree from multiple threads, so we return a dummy lock.
        return std::tuple<>();
    }

    std::tuple<const float *, float> sample_dirichlet_noise(
        std::size_t num_children, std::mt19937 &random_engine
    );

    float m_dirichlet_epsilon;
    float m_dirichlet_alpha;

    std::array<float, 60> m_dirichlet_noise;
};

template <bool VirtualLoss>
void othello::SearchTreeForSelfPlay::forward_and_evaluate(
    PositionEvaluation &evaluation,
    ThreadSafeQueue<PositionEvaluation *> &queue,
    std::mt19937 &random_engine
) {
    forward_and_evaluate_impl<SearchTreeForSelfPlay, true, VirtualLoss>(
        evaluation, queue, random_engine
    );
}

template <bool VirtualLoss>
void othello::SearchTreeForSelfPlay::expand_and_backward(PositionEvaluation &evaluation) {
    expand_and_backward_impl<SearchTreeForSelfPlay, VirtualLoss>(evaluation);
}

}  // namespace othello

#endif  // OTHELLO_ALPHAZERO_SEARCH_TREE_FOR_SELF_PLAY_H
