"""C++ std::mutex declaration."""

cdef extern from "<mutex>" namespace "std" nogil:
    cdef cppclass mutex:
        void lock()
        void unlock()
