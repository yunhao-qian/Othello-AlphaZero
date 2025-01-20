"""Training script for the AlphaZero model."""

import itertools
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from othello_mcts import MCTS
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
        "--pin-memory",
        action="store_true",
        help="use pinned memory for MCTS and data loading (default: False)",
    )
    parser.add_argument(
        "--torch-float32-matmul-precision",
        default="highest",
        choices=["highest", "high", "medium"],
        help=(
            "precision for float32 matrix multiplication in PyTorch (default: highest)"
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
        "--history-size",
        default=4,
        type=int,
        help=(
            "number of history positions to include in the neural net input features "
            "(default: 4)"
        ),
    )
    parser.add_argument(
        "--neural-net-conv-channels",
        default=128,
        type=int,
        help=(
            "number of channels in the convolutional and residual blocks (default: 128)"
        ),
    )
    parser.add_argument(
        "--neural-net-residual-blocks",
        default=9,
        type=int,
        help="number of residual blocks in the neural net (default: 9)",
    )
    parser.add_argument(
        "--neural-net-value-head-hidden-channels",
        default=128,
        type=int,
        help=(
            "number of hidden channels of the value head in the neural net "
            "(default: 128)"
        ),
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
        "--mcts-threads",
        default=2,
        type=int,
        help="number of threads for tree search in MCTS (default: 2)",
    )
    parser.add_argument(
        "--mcts-batch-size",
        default=16,
        type=int,
        help="batch size for neural net inference in MCTS (default: 16)",
    )
    parser.add_argument(
        "--mcts-c-puct-base",
        default=20000.0,
        type=float,
        help="c_puct_base of the PUCT algorithm in MCTS (default: 20000.0)",
    )
    parser.add_argument(
        "--mcts-c-puct-init",
        default=2.5,
        type=float,
        help="c_puct_init of the PUCT algorithm in MCTS (default: 2.5)",
    )
    parser.add_argument(
        "--mcts-dirichlet-epsilon",
        default=0.25,
        type=float,
        help="epsilon for Dirichlet noises in MCTS (default: 0.25)",
    )
    parser.add_argument(
        "--mcts-dirichlet-alpha",
        default=0.5,
        type=float,
        help="alpha for Dirichlet noises in MCTS (default: 0.5)",
    )
    parser.add_argument(
        "--compile-neural-net",
        action="store_true",
        help="compile the neural net (default: False)",
    )
    parser.add_argument(
        "--compile-neural-net-backend",
        default="inductor",
        help="compilation backend for the neural net (default: inductor)",
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

    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)

    if args.from_checkpoint is not None:
        iteration_start, config, mcts, neural_net, optimizer, lr_scheduler = (
            _resume_from_checkpoint(args)
        )
    else:
        iteration_start = 1
        config = {
            "mcts": {
                "history_size": args.history_size,
                "torch_device": args.device,
                "torch_pin_memory": args.pin_memory,
                "num_simulations": args.mcts_simulations,
                "num_threads": args.mcts_threads,
                "batch_size": args.mcts_batch_size,
                "c_puct_base": args.mcts_c_puct_base,
                "c_puct_init": args.mcts_c_puct_init,
                "dirichlet_epsilon": args.mcts_dirichlet_epsilon,
                "dirichlet_alpha": args.mcts_dirichlet_alpha,
            },
            "neural_net": {
                "in_channels": 1 + args.history_size * 2,
                "num_squares": 64,
                "num_actions": 65,
                "conv_channels": args.neural_net_conv_channels,
                "num_residual_blocks": args.neural_net_residual_blocks,
                "value_head_hidden_channels": (
                    args.neural_net_value_head_hidden_channels
                ),
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
                neural_net,
                fullgraph=True,
                backend=args.compile_neural_net_backend,
                mode=args.compile_neural_net_mode,
            )
        # Adam does not work well with this kind of tasks:
        # https://github.com/leela-zero/leela-zero/issues/78#issuecomment-353651540
        optimizer = SGD(neural_net.parameters(), **config["optimizer"])
        lr_scheduler = MultiStepLR(optimizer, **config["lr_scheduler"])

    print(f"Configuration:\n{json.dumps(config, indent=4)}")

    if args.compile_neural_net:
        print("Compiling the neural net")
        neural_net.eval()
        with torch.no_grad():
            neural_net(
                torch.zeros(
                    (args.mcts_batch_size, config["neural_net"]["in_channels"], 8, 8),
                    device=args.device,
                )
            )
        neural_net.train()
        dummy_output = neural_net(
            torch.zeros(
                (args.training_batch_size, config["neural_net"]["in_channels"], 8, 8),
                device=args.device,
            )
        )
        # To avoid UserWarning: Unable to hit fast path of CUDAGraphs because of
        # pending, uninvoked backwards.
        dummy_output["value"][0].backward()
        optimizer.zero_grad()

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

        with (iteration_dir / "config.json").open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=4)

        with (iteration_dir / "stats.json").open("w", encoding="utf-8") as stats_file:
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

    with (args.from_checkpoint / "config.json").open(encoding="utf-8") as config_file:
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
            neural_net,
            fullgraph=True,
            backend=args.compile_neural_net_backend,
            mode=args.compile_neural_net_mode,
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
        self.features: list[torch.Tensor] = []
        self.policies: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "policy": self.policies[index],
            "value": self.values[index],
        }


def _run_iteration(
    mcts: MCTS, neural_net: AlphaZeroNet, optimizer: SGD, args: Namespace
) -> dict[str, float]:
    """Runs a single iteration of self-play and training."""

    neural_net.eval()

    dataset = _AlphaZeroDataset()
    for _ in trange(args.self_play_games_per_iteration, desc="Self-play"):
        data = _self_play(mcts, neural_net, args)
        dataset.features += data["features"]
        dataset.policies += data["policies"]
        dataset.values += data["values"]

    mean_losses = _train(neural_net, optimizer, dataset, args)
    return mean_losses


@torch.no_grad()
def _self_play(
    mcts: MCTS, neural_net: AlphaZeroNet, args: Namespace
) -> dict[str, list[torch.Tensor]]:
    """Runs a single self-play game."""

    mcts.reset_position()

    features = []
    policies = []

    for time_step in itertools.count():
        position = mcts.position()
        if position.is_terminal():
            break
        mcts.search(neural_net)

        visit_counts = np.array(mcts.visit_counts(), dtype=np.float32)
        if time_step < 12:
            action_probabilities = visit_counts ** (1.0 / args.self_play_temperature)
            action_probabilities /= action_probabilities.sum()
            action = np.random.choice(position.legal_actions(), p=action_probabilities)
            action = int(action)
        else:
            best_action_indices = np.where(visit_counts == visit_counts.max())[0]
            best_action_index = np.random.choice(best_action_indices)
            action = position.legal_actions()[best_action_index]

        data = mcts.self_play_data()
        features += data["features"]
        policies += data["policy"]

        mcts.apply_action(action)

    num_player1_discs = np.bitwise_count(np.uint64(position.player1_discs()))
    num_player2_discs = np.bitwise_count(np.uint64(position.player2_discs()))
    if num_player1_discs > num_player2_discs:
        action_value = 1.0
    elif num_player1_discs < num_player2_discs:
        action_value = -1.0
    else:
        action_value = 0.0

    values = []
    while len(values) < len(features):
        values += [torch.tensor(action_value, dtype=torch.float32)] * 8
        action_value = -action_value

    return {"features": features, "policies": policies, "values": values}


def _train(
    neural_net: AlphaZeroNet,
    optimizer: SGD,
    dataset: _AlphaZeroDataset,
    args: Namespace,
) -> dict[str, float]:
    neural_net.train()

    total_losses = []
    policy_losses = []
    value_losses = []
    l2_losses = []
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    l2_loss_sum = 0.0
    mean_losses = {}

    progress_bar = tqdm(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=args.training_batch_size,
            shuffle=True,
            num_workers=args.training_dataloader_workers,
            drop_last=True,
            pin_memory=args.pin_memory,
            pin_memory_device=args.device if args.pin_memory else "",
        ),
        desc="Training",
        total=len(dataset) // args.training_batch_size,
    )
    for batch in progress_bar:
        features = batch["features"].to(args.device, torch.float32)
        target_policies = batch["policy"].to(args.device, torch.float32)
        target_values = batch["value"].to(args.device, torch.float32)

        optimizer.zero_grad()
        output = neural_net(features)

        policy_loss = -(target_policies * output["policy"].log()).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(output["value"], target_values)
        l2_loss = args.l2_weight_regulation * sum(
            parameter.square().sum() for parameter in neural_net.parameters()
        )
        total_loss = policy_loss + value_loss + l2_loss

        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        l2_losses.append(l2_loss.item())

        total_loss_sum += total_losses[-1]
        policy_loss_sum += policy_losses[-1]
        value_loss_sum += value_losses[-1]
        l2_loss_sum += l2_losses[-1]
        mean_losses = {
            "total_loss": total_loss_sum / len(total_losses),
            "policy_loss": policy_loss_sum / len(policy_losses),
            "value_loss": value_loss_sum / len(value_losses),
            "l2_loss": l2_loss_sum / len(l2_losses),
        }
        progress_bar.set_postfix(mean_losses)

    return mean_losses
