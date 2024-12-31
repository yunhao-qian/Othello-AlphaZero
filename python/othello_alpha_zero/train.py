"""Training script for Othello AlphaZero."""

import json
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm, trange

from .mcts_player import (
    MCTSPlayer,
    create_net_input,
    get_terminal_position_action_value,
)
from .othello import get_initial_position, is_terminal_position
from .resnet import AlphaZeroResNet


def main() -> None:
    """Entry point of the training script."""

    parser = ArgumentParser(description="Train an AlphaZero model to play Othello")

    parser.add_argument(
        "--output-dir",
        default=Path("./checkpoints"),
        type=Path,
        help="directory to save model checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "device to use for neural network training and inference "
            "(default: cuda if available, else cpu)"
        ),
    )
    parser.add_argument(
        "--iterations",
        default=100,
        type=int,
        help="number of iterations to train the model (default: 100)",
    )
    parser.add_argument(
        "--self-play-games-per-iteration",
        default=100,
        type=int,
        help="number of self-play games per iteration (default: 100)",
    )
    parser.add_argument(
        "--from-checkpoint",
        default=None,
        type=Path,
        help="start training from a checkpoint directory if specified",
    )
    parser.add_argument(
        "--net-feature-channels",
        default=256,
        type=int,
        help="number of feature channels in the neural network (default: 256)",
    )
    parser.add_argument(
        "--net-residual-blocks",
        default=19,
        type=int,
        help="number of residual blocks in the neural network (default: 19)",
    )
    parser.add_argument(
        "--net-value-head-hidden-size",
        default=256,
        type=int,
        help="hidden size of the value head in the neural network (default: 256)",
    )
    parser.add_argument(
        "--self-play-temperature",
        default=1.0,
        type=float,
        help="temperature for action selection in self-play (default: 1.0)",
    )
    parser.add_argument(
        "--training-batch-size",
        default=16,
        type=int,
        help="batch size for neural network training (default: 16)",
    )
    parser.add_argument(
        "--mcts-simulations",
        default=1600,
        type=int,
        help="number of simulations per action in MCTS (default: 1600)",
    )
    parser.add_argument(
        "--mcts-batch-size",
        default=16,
        type=int,
        help="batch size for neural network inference in MCTS (default: 16)",
    )
    parser.add_argument(
        "--mcts-threads",
        default=16,
        type=int,
        help="number of threads for parallel MCTS (default: 16)",
    )
    parser.add_argument(
        "--mcts-c-puct",
        default=1.0,
        type=float,
        help="exploration constant of the PUCT algorithm in MCTS (default: 1.0)",
    )
    parser.add_argument(
        "--mcts-dirichlet-epsilon",
        default=0.25,
        type=float,
        help=(
            "magnitude of the Dirichlet noises added to root nodes in MCTS "
            "(default: 0.25)"
        ),
    )
    parser.add_argument(
        "--mcts-dirichlet-alpha",
        default=0.03,
        type=float,
        help=(
            "weight of the Dirichlet noises added to root nodes in MCTS "
            "(default: 0.03)"
        ),
    )

    args = parser.parse_args()

    if args.from_checkpoint is None:
        net_config = {
            "in_channels": 3,
            "num_grids": 8 * 8,
            "num_actions": 8 * 8 + 1,
            "feature_channels": args.net_feature_channels,
            "num_residual_blocks": args.net_residual_blocks,
            "value_head_hidden_size": args.net_value_head_hidden_size,
        }
        net = AlphaZeroResNet(**net_config)
    else:
        with (args.from_checkpoint / "config.json").open() as config_file:
            net_config = json.load(config_file)
        net = AlphaZeroResNet(**net_config)

        checkpoint = torch.load(
            args.from_checkpoint / "model.pth", map_location="cpu", weights_only=True
        )
        net.load_state_dict(checkpoint["model"])

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(args.device)

    player = MCTSPlayer(
        net,
        num_simulations=args.mcts_simulations,
        batch_size=args.mcts_batch_size,
        num_threads=args.mcts_threads,
        leaf_depth=128,
        c_puct=args.mcts_c_puct,
        dirichlet_epsilon=args.mcts_dirichlet_epsilon,
        dirichlet_alpha=args.mcts_dirichlet_alpha,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    args.output_dir.mkdir(exist_ok=True)

    for iteration in range(1, args.iterations + 1):
        print(f"Iteration {iteration}/{args.iterations}")

        mean_losses = _run_one_iteration(player, optimizer, args)

        iteration_dir = args.output_dir / f"iteration-{iteration:03d}"
        iteration_dir.mkdir(exist_ok=True)

        with (iteration_dir / "config.json").open("w") as file:
            json.dump(net_config, file, indent=4)

        with (iteration_dir / "losses.json").open("w") as file:
            json.dump(mean_losses, file, indent=4)

        net.eval()
        torch.save(net.state_dict(), iteration_dir / "model.pth")


class _AlphaZeroDataset(torch.utils.data.Dataset):
    """In-memory dataset for AlphaZero training."""

    def __init__(self) -> None:
        self.net_inputs = []
        self.policies = []
        self.values = []

    def __len__(self) -> int:
        return len(self.net_inputs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, float]:
        return self.net_inputs[index], self.policies[index], self.values[index]


def _run_one_iteration(
    player: MCTSPlayer, optimizer: torch.optim.Optimizer, args: Namespace
) -> None:
    dataset = _AlphaZeroDataset()

    for _ in trange(args.self_play_games_per_iteration, desc="Self-play"):
        net_inputs, policies, values = _self_play(player, args.self_play_temperature)
        dataset.net_inputs += net_inputs
        dataset.policies += policies
        dataset.values += values

    net = player.net
    net.train()

    total_losses = []
    mse_losses = []
    ce_losses = []
    l2_losses = []
    mean_losses = {}

    progress_bar = tqdm(
        torch.utils.data.DataLoader(
            dataset, batch_size=args.training_batch_size, shuffle=True
        ),
        desc="Training",
        total=math.ceil(len(dataset) / args.training_batch_size),
    )
    for net_inputs, target_policies, target_values in progress_bar:
        optimizer.zero_grad()

        net_inputs = net_inputs.to(args.device, torch.float32)
        target_policies = target_policies.to(args.device, torch.float32)
        target_values = target_values.to(args.device, torch.float32)
        output_policies, output_values = net(net_inputs)

        mse_loss = torch.nn.functional.mse_loss(output_values, target_values)
        ce_loss = -(target_policies * output_policies.log()).sum(dim=1).mean()
        l2_loss = 1e-4 * sum(param.square().sum() for param in net.parameters())
        total_loss = mse_loss + ce_loss + l2_loss

        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        mse_losses.append(mse_loss.item())
        ce_losses.append(ce_loss.item())
        l2_losses.append(l2_loss.item())

        mean_losses = {
            "total_loss": np.mean(total_losses),
            "mse_loss": np.mean(mse_losses),
            "ce_loss": np.mean(ce_losses),
            "l2_loss": np.mean(l2_losses),
        }
        progress_bar.set_postfix(mean_losses)

    return mean_losses


def _self_play(
    player: MCTSPlayer, temperature: float
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Runs a single self-play game."""

    player.reset_position(get_initial_position())

    net_inputs = []
    policies = []

    while True:
        position = player.get_position()
        if is_terminal_position(position):
            break
        net_inputs.append(create_net_input(position))

        actions, visit_counts, _ = player.search()

        probabilities = np.array(visit_counts, dtype=np.float32)
        probabilities **= 1.0 / temperature
        probabilities /= probabilities.sum()
        action = np.random.choice(actions, p=probabilities)
        action = int(action)

        probabilities = np.array(visit_counts, dtype=np.float32)
        probabilities /= probabilities.sum()
        policy = np.zeros(8 * 8 + 1, dtype=np.float32)
        policy[actions] = probabilities
        policies.append(policy)

        player.apply_action(action)

    action_value = get_terminal_position_action_value(position)
    values = []
    for _ in range(len(net_inputs)):
        values.append(action_value)
        action_value = -action_value

    return net_inputs, policies, values
