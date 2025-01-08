# Training an Othello-Playing Agent with AlphaZero

This project replicates [the AlphaZero algorithm](https://arxiv.org/abs/1712.01815) to train an Othello-playing agent. A randomly initialized neural network generates self-play data using Monte Carlo Tree Search (MCTS). The network is then trained to predict optimal moves (policies) and game outcomes (values) from this data. Through iterative cycles of self-play and training, the agent refines its strategy and performance.

While many open-source AlphaZero implementations exist, they often simplify the algorithm or lack performance optimization. This project aims to deliver a high-performance AlphaZero implementation specifically for Othello. Key features include:

- Efficient bitwise board position representation and manipulation.
- Multi-threaded C++ MCTS implementation.
- PyTorch-based ResNet neural network architecture from the AlphaGo Zero paper.
- A training pipeline for self-play data generation and neural network training.
- A CLI for playing games between humans, the AlphaZero agent, and third-party agents (e.g., Egaroucid).
- A script for evaluating training progress using an Elo rating system variant.

Due to Othello's nature and resource limitations, this implementation differs from the original AlphaZero:

- Neural network size and training volume are significantly reduced.
- To compensate for the cost of generating self-play data, board symmetry is leveraged for 16x data augmentation (4 rotations × 2 flips × 2 color inversions).
- Since legal moves in Othello depend only on the current board state and not on previous positions, our implementation simplifies the neural network input features by excluding historical positions.
- There is no resignation mechanism. Consequently, all self-play games are played to completion and included in the training process.

## Results

With the optimized C++ MCTS implementation and a downscaled neural network, our setup achieves **27,200 MCTS simulations per second** on a 24-core CPU paired with an NVIDIA GeForce RTX 4090 GPU. Each action takes **30 milliseconds** on average for 816 simulations, enabling rapid self-play data generation.

Training and Evaluation: Results are pending and will be added upon completion.

## Installation

This project is organized into two packages: `othello-mcts` (located in the `cpp` subdirectory) containing the C++ MCTS implementation, and `othello-alphazero` (in the `python` subdirectory) housing the remaining Python-based components. To install these packages, clone the repository and run:

```bash
cd /path/to/Othello-AlphaZero
pip install ./cpp
pip install ./python
```

The `othello-mcts` package requires LibTorch as a build dependency, which is included with the `torch` package. If PyTorch is built with CUDA support, the necessary CUDA libraries must also be available.

The `othello-alphazero` package depends on:

- The locally built `othello-mcts`
- `numpy`
- `torch`
- `tqdm`

If you want to control the versions of these dependencies, install them beforehand.

## Usage

### Training

Use the `othello-train` command to train an AlphaZero agent. The following example demonstrates how to run the training process:

```bash
othello-train \
    --iterations 360 \
    --self-play-games-per-iteration 500 \
    --neural-net-feature-channels 128 \
    --neural-net-residual-blocks 9 \
    --neural-net-value-head-hidden-size 128 \
    --optimizer-lr 0.02 \
    --lr-scheduler-milestones 60 120 \
    --lr-scheduler-gamma 0.1 \
    --mcts-simulations 816 \
    --mcts-batch-size 24 \
    --mcts-threads 24 \
    --compile-neural-net \
    --training-batch-size 32
```

The script saves a checkpoint after each iteration in the directory specified by `--output-dir`. If training is interrupted, it can be resumed by providing the checkpoint directory using the `--from-checkpoint` option.

### Playing Games

Use the `othello-play` command to play games between humans, agents, or a combination of both. Below are examples demonstrating different scenarios:

#### Human vs. Human

Start with a simple game between two human players:

```bash
othello-play --player1 human --player2 human
```

#### Random and Greedy Players

For debugging purposes, two basic player types are included:

- `random`: Makes completely random moves.
- `greedy`: Chooses moves that maximize the immediate number of flips.

Example game:

```bash
othello-play --player1 random --player2 greedy
```

#### Human vs. AlphaZero Agent

To specify an AlphaZero agent as a player, use the `alphazero` type:

```bash
othello-play --player1 human --player2 alphazero --alphazero-checkpoint /path/to/checkpoint/dir --alphazero-compile-neural-net
```

If both players are AlphaZero agents and need different checkpoints, override them with:

- `--alphazero-checkpoint-player1`
- `--alphazero-checkpoint-player2`

The `--alphazero-compile-neural-net` flag enables torch.compile() for neural network optimization.

#### AlphaZero vs. Egaroucid

The Egaroucid engine is integrated as a player type by wrapping its CLI application in subprocesses. This allows direct benchmarking of AlphaZero agents against Egaroucid through automated gameplay.

To set up a game between an AlphaZero agent and Egaroucid, use the following command:

```bash
othello-play --player1 alphazero --player2 egaroucid --alphazero-checkpoint /path/to/checkpoint/dir --alphazero-compile-neural-net --egaroucid-exe /path/to/Egaroucid_for_Console.out --egaroucid-level 0
```

#### Additional Options

This section covers only a subset of the available options. For a full list, run:

```bash
othello-play --help
```

## Implementation Details

### Board Position Representation and Manipulation

The Othello board consists of an 8x8 grid, where each square can be empty, black, or white. This can be conveniently represented using a pair of 64-bit integers, one for each player. Finding legal moves and updating the position can both be done efficiently using bitwise operations, and the set of legal moves can also be represented as a bit mask. This representation saves memory and speeds up the search.

### Monte Carlo Tree Search (MCTS)

AlphaZero originally employs a multi-threaded MCTS implementation to maximize GPU utilization, using virtual loss to encourage exploration. However, Python's Global Interpreter Lock (GIL) limits multi-threading efficiency. To address this, we implement MCTS in C++ with integration via pybind11 and LibTorch.

In this implementation, the main thread handles neural network inference, while multiple search threads traverse the tree and send board position features to the main thread for evaluation. Additional threads manage data transfer between the CPU and GPU to mitigate slow I/O bottlenecks. The process is illustrated in the following diagram:

![Monte Carlo Tree Search](./images/mcts.svg)

A further optimization involves using a contiguous vector to store search tree nodes. While accessing nodes via indices introduces a level of indirection, this approach minimizes memory fragmentation and allows vector space to be reused across multiple self-play games.

### Neural Network

We use the same ResNet architecture as in [AlphaGo Zero](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf), implemented in PyTorch. However, given Othello's simpler rules and resource constraints, we downscale the network by reducing the number of feature channels from 256 to 128 and the residual blocks from 19 to 9, resulting in a total of 2.7 million parameters. Additionally, instead of using the last eight board positions, we simplify the input features to include only the current board state, reducing the input channels from 17 to 3.

During self-play, neural network inference batch size is limited by the number of search threads, leading to suboptimal GPU utilization. We found that `torch.compile()` significantly enhances GPU efficiency, nearly halving inference time. Therefore, we enable it whenever supported by the PyTorch version.

### Training Pipeline

We lack the resources for a comprehensive hyperparameter search as conducted in the original paper. Instead, we performed small-scale experiments and selected the following hyperparameters:

- 360 self-play and training iterations.
- 500 self-play games per iteration, yielding approximately 480,000 training examples after data augmentation.
- 816 MCTS simulations per action, distributed across 24 threads.
- 32 batch size for neural network training, with self-play data trained for only one epoch.
- L2 weight regularization with a coefficient of 1e-4.
- SGD optimizer with an initial learning rate of 0.02 and momentum of 0.9.
- The learning rate decreases by a factor of 0.1 after 60 and 120 iterations.

### Evaluation

We evaluate the relative strengths of agents by playing matches between randomly selected pairs from the following pool:

- AlphaZero agents at different training iterations, with either 816 or 3216 MCTS simulations.
- [Egaroucid](https://www.egaroucid.nyanyan.dev/en/), a strong open-source Othello engine, running in no-book mode at level 0.
- Random player, making entirely random moves.
- Greedy player, selecting moves that maximize the immediate number of flips.

To estimate the Elo rating $e(\cdot)$ of each agent, we assume that the probability of agent $a$ defeating agent $b$ follows:
$$
P(a \text{ defeats } b) = \mathrm{sigmoid}(c_\mathrm{elo} (e(a) - e(b))), \quad c_\mathrm{elo} = \frac{1}{400} \text{.}
$$

We then use gradient descent to minimize the binary cross-entropy loss between observed game outcomes and predicted probabilities, with a small L2 regularization term to prevent overfitting.
