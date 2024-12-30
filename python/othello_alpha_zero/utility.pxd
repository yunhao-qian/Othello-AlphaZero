"""Declaration file for utility functions."""

from libc.stdint cimport uint64_t


cdef int bitwise_count(uint64_t mask) noexcept nogil
