# The `othello-mcts` Package

This subdirectory contains the C++ implementation of the Monte Carlo Tree Search (MCTS) algorithm for the game of Othello. MCTS enables the AlphaZero agent to search the game tree and select optimal moves during self-play and evaluation.

- C++ headers are located in `src/include` and are intended for internal use only.
- C++ source files reside in `src/lib` and contain the core implementation.
- `src/othello_mcts` defines the Python module using pybind11 and includes type annotations for C++ bindings.
- Note that `CMakeLists.txt` is called by scikit-build during `pip install` and not intended for direct CMake builds by users.

Main components:

- `position.h` and `position.cpp`: Define board representation and manipulation.
- `mcts.h` and `mcts.cpp`: Implement the Monte Carlo Tree Search algorithm.
- `othello_mcts.cpp`: Provide Pybind11 bindings for the Position and MCTS classes.
- `queue.h`: Implement a thread-safe queue for managing data transfer between threads.
