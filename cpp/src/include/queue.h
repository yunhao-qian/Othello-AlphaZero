/// @file queue.h
/// @brief Implementation of a multi-threaded queue.

#ifndef OTHELLO_MCTS_QUEUE_H
#define OTHELLO_MCTS_QUEUE_H

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace othello {

/// @brief Multi-threaded queue.
/// @tparam T Type of the elements in the queue.
template <typename T>
class Queue {
public:
    /// @brief Pushes a new element to the queue.
    /// @param value Lvalue or rvalue reference to the element.
    template <typename U>
    void push(U &&value);

    /// @brief Pops an element from the queue.
    /// @return Popped element.
    T pop();

    /// @brief Clears the queue.
    ///
    void clear();

private:
    std::queue<T> _queue;
    std::mutex _mutex;
    std::condition_variable _condition_variable;
};

} // namespace othello

template <typename T>
template <typename U>
void othello::Queue<T>::push(U &&value) {
    std::lock_guard<std::mutex> lock(_mutex);
    _queue.push(std::forward<U>(value));

    // We expect the queue to be single/multi-in single-out, so notify_one is
    // sufficient.
    _condition_variable.notify_one();
}

template <typename T>
T othello::Queue<T>::pop() {
    std::unique_lock<std::mutex> lock(_mutex);
    _condition_variable.wait(lock, [this] { return !_queue.empty(); });
    T value = std::move(_queue.front());
    _queue.pop();
    return value;
}

template <typename T>
void othello::Queue<T>::clear() {
    std::queue<T> new_queue;
    std::lock_guard<std::mutex> lock(_mutex);
    std::swap(_queue, new_queue);
}

#endif // OTHELLO_MCTS_QUEUE_H
