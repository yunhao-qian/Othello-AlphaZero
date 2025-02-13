cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Boost 1.83 REQUIRED COMPONENTS container CONFIG)

add_library(
    othello_alphazero_core
    STATIC
    cpp/lib/position_evaluation.cpp
    cpp/lib/search_tree_base.cpp
    cpp/lib/search_tree_for_self_play.cpp
)
target_include_directories(othello_alphazero_core PUBLIC cpp/include)
