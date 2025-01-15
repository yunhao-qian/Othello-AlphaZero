# Set the minimum CMake version and policies for highest tested version.
cmake_minimum_required(VERSION 3.15...3.29)

# Set up the project and ensure there is a working C++ compiler.
project(othello_mcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# We are now ready to compile the actual extension module.
pybind11_add_module(
    # Name of the extension
    _othello_mcts_impl

    # Source code goes here.
    src/lib/mcts.cpp
    src/lib/neural_net_wrapper.cpp
    src/lib/othello_mcts.cpp
    src/lib/search_thread.cpp
)
target_include_directories(_othello_mcts_impl PRIVATE src/include)

target_include_directories(_othello_mcts_impl PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(_othello_mcts_impl PRIVATE "${TORCH_LIBRARIES}")

# https://github.com/pytorch/pytorch/issues/108041#issuecomment-1753895665
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(_othello_mcts_impl PRIVATE "${TORCH_PYTHON_LIBRARY}")

# Install directive for scikit-build-core.
install(TARGETS _othello_mcts_impl LIBRARY DESTINATION othello_mcts)
