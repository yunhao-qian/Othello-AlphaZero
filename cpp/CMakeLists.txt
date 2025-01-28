# Set the minimum CMake version and policies for highest tested version.
cmake_minimum_required(VERSION 3.18...3.29)

# Set up the project and ensure there is a working C++ compiler.
project(othello_mcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(ONNXRUNTIME_ROOT_DIR "$ENV{ONNXRUNTIME_ROOT_DIR}")
if (NOT ONNXRUNTIME_ROOT_DIR)
    message(
        FATAL_ERROR
        "Please set the environment variable ONNXRUNTIME_ROOT_DIR to the path "
        "of the ONNX Runtime installation."
    )
endif()

# Try to import all Python components potentially needed by nanobind.
find_package(
    Python 3.10 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

# Detect the installed nanobind package and import it into CMake.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    # Name of the extension
    _othello_mcts_impl

    # Target the stable ABI for Python 3.12+, which reduces the number of binary
    # wheels that must be built. This does nothing on older Python versions.
    STABLE_ABI

    # Source code
    src/lib/othello_mcts.cpp
)

target_include_directories(_othello_mcts_impl PRIVATE src/include)

# CMake support of onnxruntime is broken, so we specify the include and link
# directories manually following the official examples:
# https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/CMakeLists.txt
target_include_directories(
    _othello_mcts_impl PRIVATE "${ONNXRUNTIME_ROOT_DIR}/include"
)
target_link_directories(
    _othello_mcts_impl PRIVATE "${ONNXRUNTIME_ROOT_DIR}/lib"
)

nanobind_add_stub(
    _othello_mcts_impl_stub
    MODULE _othello_mcts_impl
    OUTPUT _othello_mcts_impl.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_othello_mcts_impl>
    DEPENDS _othello_mcts_impl
    MARKER_FILE py.typed
)

# Install directive for scikit-build-core
install(TARGETS _othello_mcts_impl LIBRARY DESTINATION othello_mcts)
install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/py.typed"
    "${CMAKE_CURRENT_BINARY_DIR}/_othello_mcts_impl.pyi"
    DESTINATION othello_mcts
)
