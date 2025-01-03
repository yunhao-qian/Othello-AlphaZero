# The CMake project declares libtorch.so as a dynamic library dependency, so we need to
# import torch before importing the bindings. Otherwise, there will be a "cannot open
# shared object file" error.
import torch

from ._othello_mcts_impl import MCTS, Position
