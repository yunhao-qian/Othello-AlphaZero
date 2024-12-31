/// @file queue.h
/// @brief Implementation of a multi-threaded queue.

#ifndef OTHELLO_MCTS_QUEUE_H
#define OTHELLO_MCTS_QUEUE_H

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
    /// @param value Element to push.
    void push(const T &value);

    /// @brief Pops an element from the queue.
    /// @return Popped element.
    T pop();

private:
    std::queue<T> _queue;
    std::mutex _mutex;
    std::condition_variable _condition_variable;
};

} // namespace othello

template <typename T>
void othello::Queue<T>::push(const T &value) {
    std::lock_guard<std::mutex> lock(_mutex);
    _queue.push(value);
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

#endif // OTHELLO_MCTS_QUEUE_H
