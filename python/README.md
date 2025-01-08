# The `othello-alphazero` Package

This subdirectory houses the Python-based components of the Othello-AlphaZero project. Ensure the `othello-mcts` package from the `cpp` subdirectory is installed before installing this package.

Main components:

- `neural_net.py`: PyTorch implementation of the neural network architecture used by the AlphaZero agent.
- `player.py`: Player classes for both human and agent players, along with a `play_game()` function for running a game between two players.
- `train.py`: Script for training the AlphaZero agent.
- `play.py`: Command-line script for playing games directly in the terminal.
