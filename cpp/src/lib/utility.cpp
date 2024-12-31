/// @file utility.cpp
/// @brief Implementation of utility functions.

#include "utility.h"

int othello::popcount(std::uint64_t x) noexcept {
    int count = 0;
    while (x != 0) {
        x &= x - 1;
        ++count;
    }
    return count;
}
