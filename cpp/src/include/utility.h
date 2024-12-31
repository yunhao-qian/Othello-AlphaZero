/// @file utility.h
/// @brief Declaration of utility functions.

#ifndef OTHELLO_MCTS_UTILITY_H

#include <cstdint>

namespace othello {

/// @brief Returns the number 1 bits in an integer.
/// @param x Integer.
/// @return Number of 1 bits.
int popcount(std::uint64_t x) noexcept;

} // namespace othello

#endif // OTHELLO_MCTS_UTILITY_H
