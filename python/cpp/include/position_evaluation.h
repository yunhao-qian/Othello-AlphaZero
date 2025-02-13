/**
 * @file position_evaluation.h
 * @brief Position evaluation class.
 */

#ifndef OTHELLO_ALPHAZERO_POSITION_EVALUATION_H
#define OTHELLO_ALPHAZERO_POSITION_EVALUATION_H

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <vector>

namespace othello {

class SearchTreeBase;

/**
 * @brief Class for setting up neural network evaluations of positions.
 */
class PositionEvaluation {
public:
    /**
     * @brief Constructs a position evaluation object.
     * @param history_size Number of history positions to include in the input features.
     */
    PositionEvaluation(int history_size);

    /**
     * @brief Sets the position to evaluate.
     * @param search_tree Search tree to get the position from.
     * @param leaf_index Index of the leaf node corresponding to the position to evaluate.
     * @param random_engine Random engine for sampling a transformation.
     * @return True if the position should be evaluated by the neural network, or false if the
     *     position is terminal.
     */
    bool set_position(
        const SearchTreeBase &search_tree, std::uint32_t leaf_index, std::mt19937 &random_engine
    ) noexcept;

    /**
     * @brief Gets the index of the leaf node corresponding to the evaluated position.
     * @return Index of the leaf node.
     */
    std::uint32_t leaf_index() const noexcept {
        return m_leaf_index;
    }

    /**
     * @brief Gets the input features.
     * @return Reference to the vector of input features.
     */
    const std::vector<float> &input_features() const noexcept {
        return m_input_features;
    }

    /**
     * @brief Sets the result of the evaluation.
     * @param policy Pointer to the policy data (i.e., prior probabilities of actions).
     * @param value Value (i.e., action-value with respect to the current player).
     */
    void set_result(const float *policy, float value);

    /**
     * @brief Waits for the result to be ready.
     */
    void wait_for_result();

    /**
     * @brief Gets the prior probability of an action.
     * @param action Action to get the probability for.
     * @return Prior probability of the action.
     */
    float get_prior_probability(int action) const noexcept;

    /**
     * @brief Gets the action-value with respect to the Black player.
     * @return Action-value.
     */
    float player1_action_value() const noexcept {
        return m_player1_action_value;
    }

private:
    int m_history_size;
    std::uint32_t m_leaf_index;
    int m_player;
    int m_transformation;
    std::vector<float> m_input_features;
    std::mutex m_result_mutex;
    std::condition_variable m_result_condition_variable;
    bool m_is_result_ready;
    std::array<float, 64> m_policy;
    float m_player1_action_value;
};

}  // namespace othello

#endif  // OTHELLO_ALPHAZERO_POSITION_EVALUATION_H
