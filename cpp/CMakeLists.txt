# Set the minimum CMake version and policies for highest tested version.
cmake_minimum_required(VERSION 3.15...3.28)

# Set up the project and ensure there is a working C++ compiler.
project(othello_mcts LANGUAGES CXX)

# Warn if the user invokes CMake directly.
if (NOT SKBUILD)
    message(
    WARNING "\
    This CMake file is meant to be executed using 'scikit-build-core'.
    Running it directly will almost certainly not produce the desired
    result. If you are a user trying to install this package, use the
    command below, which will install all necessary build dependencies,
    compile the package in an isolated environment, and then install it.
    =====================================================================
    $ pip install .
    =====================================================================
    If you are a software developer, and this is your own package, then
    it is usually much more efficient to install the build dependencies
    in your environment once and use the following command that avoids
    a costly creation of a new virtual environment at every compilation:
    =====================================================================
    $ pip install nanobind scikit-build-core[pyproject]
    $ pip install --no-build-isolation -ve .
    =====================================================================
    You may optionally add -Ceditable.rebuild=true to auto-rebuild when
    the package is imported. Otherwise, you need to rerun the above
    after editing C++ files."
    )
endif()

# Try to import all Python components potentially needed by nanobind.
find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

# Import nanobind through CMake's find_package mechanism.
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module.
nanobind_add_module(
    # Name of the extension
    _othello_mcts_impl

    # Target the stable ABI for Python 3.12+, which reduces
    # the number of binary wheels that must be built. This
    # does nothing on older Python versions.
    STABLE_ABI

    # Source code goes here.
    src/lib/mcts.cpp
    src/lib/othello_mcts.cpp
    src/lib/position.cpp
    src/lib/utility.cpp
)
target_include_directories(_othello_mcts_impl PRIVATE src/include)

nanobind_add_stub(
    _othello_mcts_impl_stub
    MODULE _othello_mcts_impl
    OUTPUT _othello_mcts_impl.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_othello_mcts_impl>
    DEPENDS _othello_mcts_impl
    MARKER_FILE py.typed
)

# Install directive for scikit-build-core.
install(TARGETS _othello_mcts_impl LIBRARY DESTINATION othello_mcts)
install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/py.typed"
    "${CMAKE_CURRENT_BINARY_DIR}/_othello_mcts_impl.pyi"
    DESTINATION othello_mcts
)
