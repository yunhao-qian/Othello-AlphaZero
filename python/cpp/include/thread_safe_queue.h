/**
 * @file thread_safe_queue.h
 * @brief Thread-safe queue class.
 */

#ifndef OTHELLO_ALPHAZERO_THREAD_SAFE_QUEUE_H
#define OTHELLO_ALPHAZERO_THREAD_SAFE_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace othello {

/**
 * @brief Queue that is safe to use from multiple threads.
 * @tparam T Element type.
 */
template <typename T>
class ThreadSafeQueue {
public:
    /**
     * @brief Pushes a new element to the queue.
     * @param args Arguments to construct the element.
     */
    template <typename... Args>
    void emplace(Args &&...args);

    /**
     * @brief Pops an element from the queue.
     * @return Popped element.
     */
    T pop();

private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_condition_variable;
};

}  // namespace othello

template <typename T>
template <typename... Args>
void othello::ThreadSafeQueue<T>::emplace(Args &&...args) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.emplace(std::forward<Args>(args)...);
    // We expect the queue to be multi-in single-out, so notify_one is enough.
    m_condition_variable.notify_one();
}

template <typename T>
T othello::ThreadSafeQueue<T>::pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condition_variable.wait(lock, [this] { return !m_queue.empty(); });
    T value = std::move(m_queue.front());
    m_queue.pop();
    return value;
}

#endif  // OTHELLO_ALPHAZERO_THREAD_SAFE_QUEUE_H
