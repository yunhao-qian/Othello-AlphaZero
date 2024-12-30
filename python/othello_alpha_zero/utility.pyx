# distutils: language = c++
"""Implementation file for utility functions."""

from libc.stdint cimport uint64_t


cdef int bitwise_count(uint64_t mask) noexcept nogil:
    """Counts the number of 1-bits in the given mask."""

    # popcount() has been available in C++20, but it is still too new.

    cdef int count = 0
    while mask:
        mask &= mask - 1
        count += 1
    return count
