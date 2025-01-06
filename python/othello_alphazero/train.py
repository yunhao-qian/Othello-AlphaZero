"""Training script for the AlphaZero model."""

import itertools
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from othello_mcts import MCTS, Position
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm, trange

from .neural_net import AlphaZeroNet


def main() -> None:
    """Entry point of the training script."""

    parser = ArgumentParser(description="Train an AlphaZero model to play Othello")

    parser.add_argument(
        "--output-dir",
        default=Path("checkpoints"),
        type=Path,
        help="directory to save model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "device to use for model training and inference (default: cuda if "
            "available, else cpu)"
        ),
    )
    parser.add_argument(
        "--iterations",
        default=100,
        type=int,
        help="number of self-play and training iterations (default: 100)",
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
        help="if specified, resume training from the checkpoint directory",
    )
    parser.add_argument(
        "--neural-net-feature-channels",
        default=256,
        type=int,
        help="number of feature channels in the neural net (default: 256)",
    )
    parser.add_argument(
        "--neural-net-residual-blocks",
        default=19,
        type=int,
        help="number of residual blocks in the neural net (default: 19)",
    )
    parser.add_argument(
        "--neural-net-value-head-hidden-size",
        default=256,
        type=int,
        help="hidden size of the value head in the neural net (default: 256)",
    )
    parser.add_argument(
        "--optimizer-lr",
        default=0.01,
        type=float,
        help="learning rate for the optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--optimizer-momentum",
        default=0.9,
        type=float,
        help="momentum for the optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--lr-scheduler-milestones",
        type=int,
        nargs="+",
        help="iterations after which to decay the learning rate",
    )
    parser.add_argument(
        "--lr-scheduler-gamma",
        default=0.1,
        type=float,
        help="multiplicative factor of learning rate decay (default: 0.1)",
    )
    parser.add_argument(
        "--self-play-temperature",
        default=1.0,
        type=float,
        help="temperature for action selection during self-play (default: 1.0)",
    )
    parser.add_argument(
        "--mcts-simulations",
        default=800,
        type=int,
        help="number of simulations per action in MCTS (default: 800)",
    )
    parser.add_argument(
        "--mcts-batch-size",
        default=16,
        type=int,
        help="batch size for neural net inference in MCTS (default: 16)",
    )
    parser.add_argument(
        "--mcts-threads",
        default=16,
        type=int,
        help="number of threads for tree search in MCTS (default: 16)",
    )
    parser.add_argument(
        "--mcts-exploration-weight",
        default=1.0,
        type=float,
        help="exploration weight of the PUCT algorithm in MCTS (default: 1.0)",
    )
    parser.add_argument(
        "--mcts-dirichlet-epsilon",
        default=0.25,
        type=float,
        help="epsilon for Dirichlet noises in MCTS (default: 0.25)",
    )
    parser.add_argument(
        "--mcts-dirichlet-alpha",
        default=0.3,
        type=float,
        help="alpha for Dirichlet noises in MCTS (default: 0.3)",
    )
    parser.add_argument(
        "--compile-neural-net",
        action="store_true",
        help="compile the neural net (default: False)",
    )
    parser.add_argument(
        "--compile-neural-net-mode",
        default="max-autotune",
        help="compilation mode for the neural net (default: max-autotune)",
    )
    parser.add_argument(
        "--training-batch-size",
        default=16,
        type=int,
        help="batch size for neural net training (default: 16)",
    )
    parser.add_argument(
        "--training-dataloader-workers",
        default=1,
        type=int,
        help="number of workers for the training dataloader (default: 1)",
    )
    parser.add_argument(
        "--l2-weight-regulation",
        default=1e-4,
        type=float,
        help="L2 weight regularization in the loss function (default: 1e-4)",
    )

    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Arguments: {args}")

    if args.from_checkpoint is not None:
        iteration_start, config, mcts, neural_net, optimizer, lr_scheduler = (
            _resume_from_checkpoint(args)
        )
    else:
        iteration_start = 1
        config = {
            "mcts": {
                "torch_device": args.device,
                "num_simulations": args.mcts_simulations,
                "batch_size": args.mcts_batch_size,
                "num_threads": args.mcts_threads,
                "exploration_weight": args.mcts_exploration_weight,
                "dirichlet_epsilon": args.mcts_dirichlet_epsilon,
                "dirichlet_alpha": args.mcts_dirichlet_alpha,
            },
            "neural_net": {
                "in_channels": 3,
                "num_squares": 64,
                "num_actions": 65,
                "feature_channels": args.neural_net_feature_channels,
                "num_residual_blocks": args.neural_net_residual_blocks,
                "value_head_hidden_size": args.neural_net_value_head_hidden_size,
            },
            "optimizer": {"lr": args.optimizer_lr, "momentum": args.optimizer_momentum},
            "lr_scheduler": {
                "milestones": args.lr_scheduler_milestones,
                "gamma": args.lr_scheduler_gamma,
            },
        }
        mcts = MCTS(**config["mcts"])
        neural_net = AlphaZeroNet(**config["neural_net"]).to(args.device)
        if args.compile_neural_net:
            neural_net = torch.compile(
                neural_net, fullgraph=True, mode=args.compile_neural_net_mode
            )
        # Adam does not work well with this kind of tasks:
        # https://github.com/leela-zero/leela-zero/issues/78#issuecomment-353651540
        optimizer = SGD(neural_net.parameters(), **config["optimizer"])
        lr_scheduler = MultiStepLR(optimizer, **config["lr_scheduler"])

    print(f"Configuration:\n{json.dumps(config, indent=4)}")

    iteration_stop = iteration_start + args.iterations
    for iteration in range(iteration_start, iteration_stop):
        lr = lr_scheduler.get_last_lr()[0]
        print(f"Iteration {iteration}/{iteration_stop - 1} (lr={lr})")
        mean_losses = _run_iteration(mcts, neural_net, optimizer, args)
        lr_scheduler.step()

        iteration_dir = args.output_dir / f"{iteration:03d}"
        print(f"Saving checkpoint to '{iteration_dir}'")
        iteration_dir.mkdir(exist_ok=True, parents=True)

        (iteration_dir / "iteration.txt").write_text(str(iteration))

        with (iteration_dir / "config.json").open("w") as config_file:
            json.dump(config, config_file, indent=4)

        with (iteration_dir / "stats.json").open("w") as stats_file:
            json.dump({"lr": lr, **mean_losses}, stats_file, indent=4)

        if args.compile_neural_net:
            neural_net_state_dict = neural_net._orig_mod.state_dict()
        else:
            neural_net_state_dict = neural_net.state_dict()
        torch.save(neural_net_state_dict, iteration_dir / "neural_net.pth")
        torch.save(optimizer.state_dict(), iteration_dir / "optimizer.pth")
        torch.save(lr_scheduler.state_dict(), iteration_dir / "lr_scheduler.pth")


def _resume_from_checkpoint(
    args: Namespace,
) -> tuple[int, dict[str, Any], MCTS, AlphaZeroNet, SGD, MultiStepLR]:
    """Loads components from a checkpoint directory."""

    with (args.from_checkpoint / "config.json").open() as config_file:
        config = json.load(config_file)

    iteration = int((args.from_checkpoint / "iteration.txt").read_text().strip())
    iteration_start = iteration + 1

    config["mcts"]["torch_device"] = args.device
    mcts = MCTS(**config["mcts"])

    neural_net = AlphaZeroNet(**config["neural_net"]).to(args.device)
    neural_net.load_state_dict(
        torch.load(
            args.from_checkpoint / "neural_net.pth",
            map_location=args.device,
            weights_only=True,
        )
    )
    if args.compile_neural_net:
        neural_net = torch.compile(
            neural_net, fullgraph=True, mode=args.compile_neural_net_mode
        )

    optimizer = SGD(neural_net.parameters(), **config["optimizer"])
    optimizer.load_state_dict(
        torch.load(
            args.from_checkpoint / "optimizer.pth",
            map_location=args.device,
            weights_only=True,
        )
    )

    lr_scheduler = MultiStepLR(optimizer, **config["lr_scheduler"])
    lr_scheduler.load_state_dict(
        torch.load(
            args.from_checkpoint / "lr_scheduler.pth",
            map_location=args.device,
            weights_only=True,
        )
    )

    return iteration_start, config, mcts, neural_net, optimizer, lr_scheduler


class _AlphaZeroDataset(torch.utils.data.Dataset):
    """In-memory dataset for AlphaZero training."""

    def __init__(self) -> None:
        self.features: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []
        self.values: list[float] = []

    def __len__(self) -> int:
        return len(self.features) * 16

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, float]:
        actual_index = index // 16
        features = self.features[actual_index]  # (3, 8, 8)
        policy = self.policies[actual_index]  # (65,)
        value = self.values[actual_index]  # float

        augmentation = index % 16
        horizontal_flip = augmentation % 2 == 1
        augmentation //= 2
        rotation = augmentation % 4
        augmentation //= 4
        swap_players = augmentation % 2 == 1

        policy_indices = np.arange(64).reshape((8, 8))
        if horizontal_flip:
            features = np.flip(features, axis=2)
            policy_indices = np.flip(policy_indices, axis=1)
        for _ in range(rotation):
            features = np.rot90(features, axes=(1, 2))
            policy_indices = np.rot90(policy_indices)
        if swap_players:
            features = np.stack((features[1], features[0], 1.0 - features[2]))
        policy_indices = np.append(policy_indices.flatten(), 64)
        policy = policy[policy_indices]

        return features.copy(), policy, value


def _run_iteration(
    mcts: MCTS, neural_net: AlphaZeroNet, optimizer: SGD, args: Namespace
) -> dict[str, float]:
    """Runs a single iteration of self-play and training."""

    neural_net.eval()

    dataset = _AlphaZeroDataset()
    for _ in trange(args.self_play_games_per_iteration, desc="Self-play"):
        features, policies, values = _self_play(mcts, neural_net, args)
        dataset.features += features
        dataset.policies += policies
        dataset.values += values

    mean_losses = _train(neural_net, optimizer, dataset, args)
    return mean_losses


@torch.no_grad()
def _self_play(
    mcts: MCTS, neural_net: AlphaZeroNet, args: Namespace
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Runs a single self-play game."""

    mcts.reset_position(Position.initial_position())

    features = []
    policies = []

    for time_step in itertools.count():
        position = mcts.root_position()
        if position.is_terminal():
            break
        features.append(
            np.array(position.to_features(), dtype=np.float32).reshape((3, 8, 8))
        )
        search_result = mcts.search(neural_net)

        visit_counts = np.array(search_result["visit_counts"], dtype=np.float32)

        if time_step < 12:
            action_probabilities = visit_counts ** (1.0 / args.self_play_temperature)
            action_probabilities /= action_probabilities.sum()
            action = np.random.choice(search_result["actions"], p=action_probabilities)
            action = int(action)
        else:
            action = search_result["actions"][visit_counts.argmax()]

        policy_probabilities = visit_counts / visit_counts.sum()
        policy = np.zeros(65, dtype=np.float32)
        policy[search_result["actions"]] = policy_probabilities
        policies.append(policy)

        mcts.apply_action(action)

    action_value = position.action_value()
    values = []
    for _ in range(len(features)):
        values.append(action_value)
        action_value = -action_value

    return features, policies, values


def _train(
    neural_net: AlphaZeroNet,
    optimizer: SGD,
    dataset: _AlphaZeroDataset,
    args: Namespace,
) -> dict[str, float]:
    neural_net.train()

    total_losses = []
    mse_losses = []
    ce_losses = []
    l2_losses = []
    mean_losses = {}

    progress_bar = tqdm(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=args.training_batch_size,
            shuffle=True,
            num_workers=args.training_dataloader_workers,
            drop_last=True,
        ),
        desc="Training",
        total=len(dataset) // args.training_batch_size,
    )
    for features, target_policies, target_values in progress_bar:
        features = features.to(args.device, torch.float32)
        target_policies = target_policies.to(args.device, torch.float32)
        target_values = target_values.to(args.device, torch.float32)

        optimizer.zero_grad()
        output_policies, output_values = neural_net(features)

        mse_loss = nn.functional.mse_loss(output_values, target_values)
        ce_loss = -(target_policies * output_policies.log()).sum(dim=1).mean()
        l2_loss = args.l2_weight_regulation * sum(
            parameter.square().sum() for parameter in neural_net.parameters()
        )
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
