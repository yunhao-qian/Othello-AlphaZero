# distutils: language = c++
"""Monte Carlo Tree Search player implementation."""

from cython cimport view
from cython.operator cimport dereference as deref
from libc.math cimport sqrt
from libc.stdint cimport uint64_t
from libcpp.random cimport gamma_distribution, mt19937, random_device
from libcpp.vector cimport vector

from .mutex cimport mutex
from .othello cimport (
    apply_action,
    get_initial_position,
    get_legal_actions,
    is_terminal_position,
    Position,
)
from .utility cimport bitwise_count

from queue import Queue
from threading import Thread

import numpy as np
import torch

from .resnet import AlphaZeroResNet


"""Node in the Monte Carlo Tree Search algorithm."""
cdef struct MCTSNode:
    Position position
    int previous_action
    int visit_count
    double total_action_value
    double mean_action_value
    double prior_probability
    vector[size_t] children


"""Part of the MCTS context that can be described in a C struct."""
cdef struct MCTSContext:
    int num_simulations
    int batch_size
    int num_threads
    int leaf_depth
    double c_puct
    double dirichlet_epsilon
    double dirichlet_alpha

    vector[MCTSNode] search_tree
    mutex search_tree_mutex


cdef size_t _get_best_child(
    MCTSContext &context, vector[size_t] &children, bint is_root, mt19937 &rng
) noexcept nogil:
    """Returns the child with the highest Upper Confidence Bound (UCB)."""

    cdef size_t child_index
    cdef int sum_visit_count = 0
    for child_index in children:
        sum_visit_count += context.search_tree[child_index].visit_count
    cdef double sqrt_sum_visit_count = sqrt(<double>sum_visit_count)

    # Compute Dirichlet noises which encourage additional exploration.

    cdef vector[double] noises
    noises.reserve(children.size())

    cdef double noise
    cdef gamma_distribution[double] gamma_dist = gamma_distribution[double](
        context.dirichlet_alpha, 1.0
    )
    cdef double noise_sum = 0.0

    if is_root and context.dirichlet_epsilon > 0:
        for _ in range(children.size()):
            noise = gamma_dist(rng)
            noise_sum += noise
            noises.push_back(noise)

        if noise_sum > 0:
            for i in range(children.size()):
                noises[i] = noises[i] / noise_sum
    else:
        for _ in range(children.size()):
            noises.push_back(0.0)

    cdef MCTSNode *child
    cdef double probability
    cdef double ucb

    # UCB is at least the minimum action-value, which is -1.
    cdef double highest_ucb = -10.0
    cdef size_t best_child = 0

    for i in range(children.size()):
        child_index = children[i]
        child = &context.search_tree[child_index]
        probability = (
            (1 - context.dirichlet_epsilon) * child.prior_probability
            + context.dirichlet_epsilon * noises[i]
        )
        ucb = (
            child.mean_action_value
            + context.c_puct
            * probability
            * sqrt_sum_visit_count
            / <double>(1 + child.visit_count)
        )
        if ucb > highest_ucb:
            highest_ucb = ucb
            best_child = child_index

    return best_child


cdef vector[size_t] _forward_pass(MCTSContext &context, mt19937 &rng) noexcept nogil:
    """Forward pass of an MCTS simulation."""

    cdef vector[size_t] search_path
    search_path.push_back(0)  # Root node

    cdef MCTSNode *node
    for i in range(context.leaf_depth):
        node = &context.search_tree[search_path.back()]
        if is_terminal_position(node.position) or node.children.empty():
            # It is a terminal position or the node has not been expanded yet.
            break
        search_path.push_back(
            _get_best_child(context, node.children, i == 0, rng)
        )

    # Use virtual losses to ensure each thread evaluates different nodes.
    cdef MCTSNode *child
    for i in range(1, search_path.size()):
        child = &context.search_tree[search_path[i]]
        child.visit_count += 1
        child.total_action_value -= 1.0
        child.mean_action_value = (
            child.total_action_value / <double>child.visit_count
        )

    return search_path


cdef double _get_terminal_position_action_value(Position &position) noexcept nogil:
    """Returns the action-value of a terminal position."""

    cdef int p1_num_discs = bitwise_count(position.p1_discs)
    cdef int p2_num_discs = bitwise_count(position.p2_discs)

    # With respect to P1: -1 for loss, 0 for draw, +1 for win.
    if p1_num_discs > p2_num_discs:
        return 1.0
    if p1_num_discs < p2_num_discs:
        return -1.0
    return 0.0


cdef void _mask_to_array(uint64_t mask, float[:, :] array) noexcept nogil:
    """Converts the given 8x8 mask into a 8x8 array."""

    cdef uint64_t square_mask = <uint64_t>1 << 63

    for row in range(8):
        for col in range(8):
            array[row, col] = <float>(mask & square_mask != 0)
            square_mask >>= 1


cpdef object create_net_input(Position &position):
    """Creates the input tensor for the neural network."""

    features = view.array(shape=(3, 8, 8), itemsize=sizeof(float), format="f")
    cdef float[:, :, :] features_view = features

    with nogil:
        _mask_to_array(position.p1_discs, features_view[0])
        _mask_to_array(position.p2_discs, features_view[1])
        features_view[2, :, :] = <float>(position.player == 2)

    array = np.asarray(features)
    return torch.from_numpy(array)


cdef void _simulate_once(
    MCTSContext &context,
    int thread_index,
    object net_input_queue,
    object net_output_queue,
    mt19937 &rng,
):
    """Performs a single simulation of the MCTS algorithm."""

    cdef vector[size_t] search_path
    cdef size_t leaf_index
    cdef MCTSNode *leaf
    cdef Position leaf_position
    cdef double action_value
    cdef vector[int] legal_actions
    cdef MCTSNode new_node
    cdef MCTSNode *child

    with nogil:
        context.search_tree_mutex.lock()

        search_path = _forward_pass(context, rng)
        leaf_index = search_path.back()
        leaf = &context.search_tree[leaf_index]

        # Expand and evaluate the leaf node.
        if is_terminal_position(leaf.position):
            # If the game is over, we do not need neural network evaluation.
            action_value = _get_terminal_position_action_value(leaf.position)
        else:
            leaf_position = leaf.position
            context.search_tree_mutex.unlock()
            legal_actions = get_legal_actions(leaf_position)

            with gil:
                net_input = create_net_input(leaf_position)
                net_input_queue.put((thread_index, net_input))
                policy, value = net_output_queue.get()

            # Holding the GIL while attempting to acquire another lock can result in a
            # deadlock.
            context.search_tree_mutex.lock()

            # The leaf pointer may be invalidated whenever the search tree grows.
            leaf = &context.search_tree[leaf_index]

            # There is a small chance that the leaf node has already been expanded by
            # another thread, in which case we should not overwrite the children.
            if leaf.children.empty():
                leaf.children.reserve(legal_actions.size())
                context.search_tree.reserve(
                    context.search_tree.size() + legal_actions.size()
                )

                for action in legal_actions:
                    new_node.position = apply_action(leaf_position, action)
                    new_node.previous_action = action
                    new_node.previous_action = action
                    new_node.visit_count = 0
                    new_node.total_action_value = 0.0
                    new_node.mean_action_value = 0.0
                    with gil:
                        new_node.prior_probability = policy[action].item()

                    leaf = &context.search_tree[leaf_index]
                    leaf.children.push_back(context.search_tree.size())
                    context.search_tree.push_back(new_node)

            with gil:
                action_value = value.item()

        # Backward pass to update the visit counts and action-values.
        for i in range(1, search_path.size()):
            child = &context.search_tree[search_path[i]]
            # The visit count has been incremented when setting the virtual loss.
            # +1 to cancel out -1 in the previous virtual loss.
            child.total_action_value += 1.0 + action_value * (
                1.0 if child.position.player == 1 else -1.0
            )
            child.mean_action_value = (
                child.total_action_value / <double>child.visit_count
            )

        context.search_tree_mutex.unlock()


cdef size_t _collect_subtree(
    vector[MCTSNode] &old_tree, vector[MCTSNode] &new_tree, size_t old_index
):
    """Collects the subtree rooted at the given node index from the old tree to the new
    tree."""

    cdef MCTSNode *old_node = &old_tree[old_index]

    cdef size_t new_index = new_tree.size()
    new_tree.emplace_back()
    cdef MCTSNode *new_node = &new_tree[new_index]

    new_node.position = old_node.position
    new_node.previous_action = old_node.previous_action
    new_node.visit_count = old_node.visit_count
    new_node.total_action_value = old_node.total_action_value
    new_node.mean_action_value = old_node.mean_action_value
    new_node.prior_probability = old_node.prior_probability

    new_node.children.reserve(old_node.children.size())
    cdef size_t new_child_index
    for old_child_index in old_node.children:
        new_child_index = _collect_subtree(old_tree, new_tree, old_child_index)
        new_node = &new_tree[new_index]
        new_node.children.push_back(new_child_index)

    return new_index


cdef vector[MCTSNode] _prune_search_tree(
    vector[MCTSNode] &tree, size_t new_root_index
):
    """Prunes the search tree to keep only the subtree rooted at the new root node."""

    cdef vector[MCTSNode] new_tree
    _collect_subtree(tree, new_tree, new_root_index)
    return new_tree


cdef class MCTSPlayer:
    """Monte Carlo Tree Search player."""

    cdef object _net
    cdef MCTSContext context

    def __init__(
        self,
        net: AlphaZeroResNet,
        num_simulations: int = 1600,
        batch_size: int = 24,
        num_threads: int = 24,
        leaf_depth: int = 64,
        c_puct: float = 1.0,
        dirichlet_epsilon: float = 0.0,
        dirichlet_alpha: float = 0.03,
    ):
        self._net = net

        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.leaf_depth = leaf_depth
        self.c_puct = c_puct
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha

        self.reset_search_tree(get_initial_position())

    def reset_search_tree(self, position: Position) -> None:
        """Resets the search tree to the given game position."""

        self.context.search_tree.clear()
        cdef MCTSNode root
        root.position = position
    
        # Statistics for the root node are never used, but we initialize them anyway.
        root.previous_action = 64
        root.visit_count = 0
        root.total_action_value = 0.0
        root.mean_action_value = 0.0
        root.prior_probability = 1.0
        self.context.search_tree.push_back(root)

    @property
    def net(self) -> AlphaZeroResNet:
        return self._net

    @net.setter
    def net(self, value: AlphaZeroResNet) -> None:
        self._net = value

    @property
    def num_simulations(self) -> int:
        return self.context.num_simulations

    @num_simulations.setter
    def num_simulations(self, value: int) -> None:
        if value < 1:
            raise ValueError(
                f"Number of simulations must be at least 1, but got {value}."
            )
        self.context.num_simulations = value

    @property
    def batch_size(self) -> int:
        return self.context.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"Batch size must be at least 1, but got {value}.")
        self.context.batch_size = value

    @property
    def num_threads(self) -> int:
        return self.context.num_threads

    @num_threads.setter
    def num_threads(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"Number of threads must be at least 1, but got {value}.")
        self.context.num_threads = value

    @property
    def leaf_depth(self) -> int:
        return self.context.leaf_depth

    @leaf_depth.setter
    def leaf_depth(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"Leaf depth must be at least 1, but got {value}.")
        self.context.leaf_depth = value

    @property
    def c_puct(self) -> float:
        return self.context.c_puct

    @c_puct.setter
    def c_puct(self, value: float) -> None:
        if not value > 0.0:
            raise ValueError(f"c_PUCT must be positive, but got {value}.")
        self.context.c_puct = value

    @property
    def dirichlet_epsilon(self) -> float:
        return self.context.dirichlet_epsilon

    @dirichlet_epsilon.setter
    def dirichlet_epsilon(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Dirichlet epsilon must be between 0 and 1, but got {value}."
            )
        self.context.dirichlet_epsilon = value

    @property
    def dirichlet_alpha(self) -> float:
        return self.context.dirichlet_alpha

    @dirichlet_alpha.setter
    def dirichlet_alpha(self, value: float) -> None:
        if not value > 0.0:
            raise ValueError(f"Dirichlet alpha must be positive, but got {value}.")
        self.context.dirichlet_alpha = value

    def get_position(self) -> Position:
        """Returns the current game position."""

        return self.context.search_tree[0].position

    def get_best_action(self) -> int:
        """Chooses the best move according to the search tree."""

        actions, visit_counts, _ = self.search()
        return actions[np.argmax(visit_counts)]

    def search(self) -> tuple[list[int], list[int], list[float]]:
        """Performs a Monte Carlo Tree Search."""

        cdef int num_simulations = self.context.num_simulations
        cdef int num_threads = self.context.num_threads
        cdef int remainder = num_simulations % num_threads
        if remainder != 0:
            raise ValueError(
                f"Number of simulations ({num_simulations}) must be a multiple of the "
                f"number of threads ({num_threads})."
            )

        return _search(self)

    def apply_action(self, action: int) -> None:
        """Applies the given action to the current game position and updates the search
        tree accordingly."""

        cdef MCTSNode root = self.context.search_tree[0]
        if is_terminal_position(root.position):
            raise ValueError("Cannot apply action to a terminal position.")

        if root.children.empty():
            # The root node has not been expanded yet. Create a new search tree from
            # scratch.
            root.position = apply_action(root.position, action)
            root.previous_action = action
            root.visit_count = 0
            root.total_action_value = 0.0
            root.mean_action_value = 0.0
            root.prior_probability = 1.0
            root.children.clear()
            self.context.search_tree.clear()
            self.context.search_tree.push_back(root)
            return

        # The root node has been expanded. Find the child node corresponding to the
        # given action.
        cdef size_t child_index = 0
        for child_index in root.children:
            if self.context.search_tree[child_index].previous_action == action:
                break
        else:
            raise ValueError(f"Action {action} is not a legal move.")

        # Prune the search tree.
        self.context.search_tree = _prune_search_tree(
            self.context.search_tree, child_index
        )


def _search_thread(
    player: MCTSPlayer,
    thread_index: int,
    net_input_queue: Queue,
    net_output_queue: Queue,
) -> None:
    """Entry point for a search thread."""

    cdef random_device rd
    cdef mt19937 rng
    rng.seed(rd())

    cdef MCTSContext *context = &player.context
    cdef int num_simulations = context.num_simulations // context.num_threads
    for _ in range(num_simulations):
        _simulate_once(
            deref(context), thread_index, net_input_queue, net_output_queue, rng
        )
    # Put a None to indicate that the thread has finished.
    net_input_queue.put((thread_index, None))


def _net_input_thread(
    player: MCTSPlayer,
    net_input_queue: Queue,
    batched_net_input_queue: Queue,
    device: torch.device,
) -> None:
    """Entry point for the neural network input thread. It is responsible for batching
    input tensors and sending them to the device of the neural network."""

    cdef int num_running_threads = player.context.num_threads
    cdef int batch_size = player.context.batch_size
    if num_running_threads < batch_size:
        batch_size = num_running_threads

    thread_indices = []
    net_inputs = []

    while True:
        thread_index, net_input = net_input_queue.get()
        if net_input is None:
            # A thread has finished.
            num_running_threads -= 1
            if num_running_threads == 0:
                # All threads have finished.
                break
            if num_running_threads < batch_size:
                batch_size = num_running_threads
        else:
            thread_indices.append(thread_index)
            net_inputs.append(net_input)

        if len(net_inputs) == batch_size:
            # We have a full batch.
            batched_net_input = torch.stack(net_inputs).to(device)
            batched_net_input_queue.put((thread_indices, batched_net_input))
            thread_indices = []
            net_inputs.clear()

    # Put an empty batch to indicate that all threads have finished.
    batched_net_input_queue.put(([], None))


def _net_output_thread(
    net_output_queues: list[Queue], batched_net_output_queue: Queue
) -> None:
    """Entry point for the neural network output thread. It is responsible for moving
    output tensors to the CPU and distributing them to the corresponding search
    threads."""

    while True:
        thread_indices, batched_policy, batched_value = batched_net_output_queue.get()
        if not thread_indices:
            # An empty batch indicates that all threads have finished.
            break
        batched_policy = batched_policy.cpu()
        batched_value = batched_value.cpu()
        for thread_index, policy, value in zip(
            thread_indices, batched_policy, batched_value
        ):
            net_output_queues[thread_index].put((policy, value))


def _search(player: MCTSPlayer) -> tuple[list[int], list[int], list[float]]:
    """Performs a Monte Carlo Tree Search."""

    player.net.eval()
    device = next(player.net.parameters()).device

    net_input_queue = Queue()
    batched_net_input_queue = Queue()

    net_output_queues = [Queue() for _ in range(player.context.num_threads)]
    batched_net_output_queue = Queue()

    search_threads = []
    for thread_index in range(player.context.num_threads):
        search_thread = Thread(
            target=_search_thread,
            args=(player, thread_index, net_input_queue, net_output_queues[thread_index]),
        )
        search_thread.start()
        search_threads.append(search_thread)

    net_input_thread = Thread(
        target=_net_input_thread,
        args=(player, net_input_queue, batched_net_input_queue, device),
    )
    net_input_thread.start()

    net_output_thread = Thread(
        target=_net_output_thread,
        args=(net_output_queues, batched_net_output_queue),
    )
    net_output_thread.start()

    with torch.no_grad():
        while True:
            thread_indices, batched_net_input = batched_net_input_queue.get()
            if not thread_indices:
                # An empty batch indicates that all threads have finished.
                # Propagate the signal to the output thread.
                batched_net_output_queue.put(([], None, None))
                break
            batched_policy, batched_value = player.net(batched_net_input)
            batched_net_output_queue.put(
                (thread_indices, batched_policy, batched_value)
            )

    net_input_thread.join()
    net_output_thread.join()
    for search_thread in search_threads:
        search_thread.join()

    cdef MCTSNode *root = &player.context.search_tree[0]
    cdef MCTSNode *child
    actions = []
    visit_counts = []
    mean_action_values = []
    for child_index in root.children:
        child = &player.context.search_tree[child_index]
        actions.append(child.previous_action)
        visit_counts.append(child.visit_count)
        mean_action_values.append(child.mean_action_value)

    return actions, visit_counts, mean_action_values
