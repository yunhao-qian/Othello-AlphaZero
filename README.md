# Training an Othello-Playing Agent with AlphaZero

This project replicates [the AlphaZero algorithm](https://arxiv.org/abs/1712.01815) to train an Othello-playing agent. A randomly initialized neural network generates self-play data using Monte Carlo Tree Search (MCTS). The network is then trained to predict optimal moves (policies) and game outcomes (values) from this data. Through iterative cycles of self-play and training, the agent refines its strategy and performance.

While many open-source AlphaZero implementations exist, they often simplify the algorithm or lack performance optimization. This project aims to deliver a high-performance AlphaZero implementation specifically for Othello. Key features include:

- Efficient bitwise board position representation and manipulation.
- Multi-threaded, batched MCTS implementation in C++.
- PyTorch-based ResNet neural network architecture from the [AlphaGo Zero]((https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)) paper.
- A training pipeline for self-play data generation and neural network training.
- A CLI for playing games between humans, the AlphaZero agent, and third-party agents (e.g., Egaroucid).
- A script for evaluating training progress using an Elo rating system variant.

Due to Othello's nature and resource limitations, this implementation differs from the original AlphaZero in several ways:

- To reduce the need for self-play data and accelerate training, we leverage the game's inherent symmetry, as outlined in the AlphaGo Zero paper. Self-play data is augmented eightfold using reflection and rotation, and random transformations are applied to board positions before neural network evaluation.
- There is no resignation mechanism. Consequently, all self-play games are played to completion and included in the training process.

## Results

### 128x10b ResNet

In this experiment, we use a downscaled version of the AlphaGo Zero ResNet with 2.7 million parameters, consisting of 1 convolutional block followed by 9 residual blocks. All convolutional layers, as well as the hidden layer in the value head, are configured with 128 channels. The neural network is trained over 180 iterations, starting with an initial learning rate of 0.02, which is reduced by a factor of 0.1 at the 60th and 120th iterations. Each self-play and training iteration generates 1,000 games, resulting in approximately 480,000 training examples after applying data augmentation.

Leveraging an optimized C++ MCTS implementation alongside the downscaled neural network, our setup achieves an impressive **28,000 MCTS simulations per second** on a 24-core CPU and an NVIDIA GeForce RTX 4090 GPU, and each action takes less than 30 milliseconds for 800 MCTS simulations. Each self-play and training iteration completes in about 30 minutes, and the entire training process spans roughly 4 days. While the model is not trained to its maximum potential, we intentionally stop training early to save time for other tasks.

The following two figures depict the loss over training iterations. They highlight that while the loss decreases rapidly at the start and after each reduction in the learning rate, it quickly plateaus thereafter. Notably, only the L2 regularization term continues to decline steadily. However, this plateau does not necessarily mean the model has stopped improving because:

- The loss signal is noisy and relatively weak due to techniques designed to encourage exploration, such as Dirichlet noise and a high PUCT exploration rate in MCTS.
- The distributions of policy and value predictions are moving targets that shift as the model improves, meaning the loss does not directly correlate with the model's overall strength.

![Loss curve for the 128x10b model (stacked)](./images/128x10b_loss_vs_iteration_stacked.svg)

![Loss curve for the 128x10b model (split)](./images/128x10b_loss_vs_iteration_split.svg)

In contrast, the Elo rating provides a clearer indication of progress. We evaluate the agent's performance against the Egaroucid engine at various levels. To estimate Elo ratings, we conduct pairwise matches between all possible combinations of players, record the outcomes, and calculate the rating for each player. The following figure demonstrates how the agent's Elo rating evolves over training iterations. Although the curve is noisy, it consistently trends upward with a significant slope, indicating that the agent is learning and improving, even as the loss curve has plateaued and the learning rate has reached very low levels.

![Relative Elo rating over training iterations](./images/128x10b_elo_vs_iteration.svg)

The AlphaZero agent is still much weaker than the higher levels of the Egaroucid engine. However, it consistently outperforms amateur players. From my experience playing a few games against Egaroucid, I estimate my skill level to be between level 1 and level 2. Remarkably, AlphaZero achieves a 400 Elo rating advantage over me while using just 0.1 seconds per action.

## Installation

This project is organized into two packages: `othello-mcts` (located in the `cpp` subdirectory) containing the C++ MCTS implementation, and `othello-alphazero` (in the `python` subdirectory) housing the remaining Python-based components. To install these packages, clone the repository and run:

```bash
cd /path/to/Othello-AlphaZero
pip install ./cpp
pip install ./python
```

The `othello-mcts` package requires LibTorch as a build dependency, which is included with the `torch` package. If PyTorch is built with CUDA support, the necessary CUDA libraries must also be available.

The `othello-alphazero` package depends on:

- The locally built `othello-mcts` package
- `numpy`
- `torch`
- `tqdm`

If you want to control the versions of these dependencies, install them beforehand.

## Usage

### Training

Use the `othello-train` command to train an AlphaZero agent. The following example demonstrates how to run the training process:

```bash
othello-train \
    --output-dir checkpoints \
    --device cuda \
    --pin-memory \
    --iterations 180 \
    --self-play-games-per-iteration 1000 \
    --history-size 8 \
    --neural-net-conv-channels 128 \
    --neural-net-residual-blocks 9 \
    --neural-net-value-head-hidden-channels 128 \
    --optimizer-lr 0.02 \
    --lr-scheduler-milestones 60 120 \
    --lr-scheduler-gamma 0.1 \
    --mcts-simulations 800 \
    --mcts-threads 2 \
    --mcts-batch-size 16 \
    --mcts-c-puct-base 20000.0 \
    --mcts-c-puct-init 2.5 \
    --compile-neural-net \
    --training-batch-size 256
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

### Evaluation

The `othello_alphazero.evaluation` module offers utility functions to evaluate player performance. A command-line interface is not included, as designing one to suit all scenarios is challenging.

The play_games() function simulates a series of games between randomly selected player pairs from a pool, recording the results (win, loss, or draw) in a JSON file. Since this process can be time-consuming, users can provide a callback function to report progress after each game pair.

The `estimate_elo()` function calculates rough Elo ratings for players based on the recorded game outcomes. This implementation is simplified, so we recommend using the [BayesElo]((https://github.com/ddugovic/BayesianElo)) program for more accurate Elo rating calculations. BayesElo expects PGN files as input, which can be generated using the `save_pgn()` function.

## Implementation Details

### Board Position Representation and Manipulation

The Othello board consists of an 8x8 grid, where each square can be empty, black, or white. This can be conveniently represented using a pair of 64-bit integers, one for each player. Finding legal moves and updating the position can both be done efficiently using bitwise operations, and the set of legal moves can also be represented as a bit mask. This representation saves memory and speeds up the search.

### Monte Carlo Tree Search (MCTS)

AlphaZero originally employs a multi-threaded MCTS implementation to maximize GPU utilization, using virtual loss to encourage exploration. However, Python's Global Interpreter Lock (GIL) limits multi-threading efficiency. To address this, we implement MCTS in C++ with integration via pybind11 and LibTorch.

In this implementation, the main thread handles neural network inference, while multiple search threads traverse the tree and send board position features to the main thread for evaluation. Additional threads manage data transfer between the CPU and GPU to mitigate slow I/O bottlenecks. The process is illustrated in the following diagram:

![Monte Carlo Tree Search](./images/mcts.svg)

A further optimization involves using a contiguous vector to store search tree nodes. While accessing nodes via indices introduces a level of indirection, this approach minimizes memory fragmentation and allows vector space to be reused across multiple self-play games.

### Neural Network

We use the same ResNet architecture as in AlphaGo Zero, implemented in PyTorch. However, given Othello's simpler rules and resource constraints, we downscale the network by reducing the number of feature channels and residual blocks.

During self-play, the neural network inference batch size must remain small because highly parallel MCTS searches can degrade result quality. However, this leads to suboptimal GPU utilization. To address this, we found that `torch.compile()` significantly enhances GPU efficiency, reducing inference time by nearly half. Therefore, we enable it whenever the PyTorch version supports this feature.

### Training Pipeline

We lack the resources for a comprehensive hyperparameter search as conducted in the original paper. Instead, we performed small-scale experiments and selected the following hyperparameters:

- 800 MCTS simulations per action, with 2 threads and a batch size of 16.
- Self-play data is trained for only one epoch.
- L2 weight regularization with a coefficient of 1e-4.
- SGD optimizer with an initial learning rate of 0.02 and momentum of 0.9.

### Evaluation

We evaluate the relative strengths of agents by playing matches between randomly selected pairs from the following pool:

- AlphaZero agents after different training iterations, with 3200 MCTS simulations per action.
- [Egaroucid](https://www.egaroucid.nyanyan.dev/en/), a strong open-source Othello engine, running in no-book mode at various levels.

To estimate the Elo rating $e(\cdot)$ of each agent, we assume that a game between two players $a$ (Black) and $b$ (White) has outcomes with the following probability distribution:

$$
\begin{cases}
    P(a \text{ wins}) &= f(e(a) - e(b) + e_\text{advantage} - e_\text{draw}) \\
    P(b \text{ wins}) &= f(e(b) - e(a) - e_\text{advantage} - e_\text{draw}) \\
    P(\text{draw}) &= 1 - P(a \text{ wins}) - P(b \text{ wins})
\end{cases} \text{,}
$$
$$
\text{where } f(\Delta) = \frac{1}{1 + 10^{-\Delta / 400}} \text{.}
$$

In the above equations, $e_\text{advantage}$ is a variable indicating the advantage of playing first (as Black), and $e_\text{draw}$ is a positive variable indicating how likely draws are. This estimation model is adapted from the [BayesElo website](https://www.remi-coulom.fr/Bayesian-Elo/).

We then apply gradient descent to perform Maximum Likelihood Estimation (MLE) of the Elo ratings by minimizing the negative log-likelihood of the observed outcomes.
