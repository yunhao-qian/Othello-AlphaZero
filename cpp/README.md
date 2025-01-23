# The `othello-mcts` Package

This subdirectory contains the C++ implementation of the Monte Carlo Tree Search (MCTS) algorithm for the game of Othello. MCTS enables the AlphaZero agent to search the game tree and select optimal moves during self-play and evaluation.

- C++ headers are located in `src/include` and are intended for internal use only.
- C++ source files reside in `src/lib` and contain most of the implementation.
- The Python module in `src/othello_mcts` provides bindings with type annotations.
- Note that `CMakeLists.txt` is invoked by scikit-build during `pip install` and is not intended for direct CMake builds by users.

Main components:

- `position.h`: Represents and manipulates game positions.
- `mcts.h` and `mcts.cpp`: Implements the Monte Carlo Tree Search (MCTS) algorithm.
- `othello_mcts.cpp`: Defines pybind11 bindings for Python integration.

Utility components:

- `neural_net.h`:  Provides an interface for interacting with neural networks.
- `position_iterator.h`: Iterates over historical game positions.
- `queue.h`: Implements a thread-safe queue for data transfer between threads.
- `search_node.h`: Represents a node in the search tree.
- `search_thread.h` and `search_thread.cpp`: Defines worker threads for MCTS execution.
- `transformation.h`: Handles transformations of game positions.
