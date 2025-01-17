"""Script for playing a game of Othello between humans and agents."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from .player import (
    AlphaZeroPlayer,
    EgaroucidPlayer,
    GreedyPlayer,
    HumanPlayer,
    Player,
    RandomPlayer,
    play_game,
)


def main() -> None:
    """Entry point of the script."""

    parser = ArgumentParser(description="Play a game of Othello")

    parser.add_argument(
        "--player1",
        default="human",
        choices=["human", "random", "greedy", "alphazero", "egaroucid"],
        help="kind of player for the Black player (default: human)",
    )
    parser.add_argument(
        "--player2",
        default="human",
        choices=["human", "random", "greedy", "alphazero", "egaroucid"],
        help="kind of player for the White player (default: human)",
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
        "--alphazero-device",
        default=None,
        help="device for the AlphaZero player (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--alphazero-pin-memory",
        action="store_true",
        help="use pinned memory for the AlphaZero player (default: False)",
    )
    parser.add_argument(
        "--alphazero-simulations",
        type=int,
        default=800,
        help="number of MCTS simulations for the AlphaZero player (default: 800)",
    )
    parser.add_argument(
        "--alphazero-simulations-player1",
        type=int,
        default=None,
        help="override alphazero-simulations for player 1",
    )
    parser.add_argument(
        "--alphazero-simulations-player2",
        type=int,
        default=None,
        help="override alphazero-simulations for player 2",
    )
    parser.add_argument(
        "--alphazero-threads",
        type=int,
        default=2,
        help="number of threads for the AlphaZero player (default: 2)",
    )
    parser.add_argument(
        "--alphazero-batch-size",
        type=int,
        default=16,
        help="batch size for the AlphaZero player (default: 16)",
    )
    parser.add_argument(
        "--alphazero-checkpoint",
        type=Path,
        default=None,
        help="checkpoint directory for the AlphaZero player",
    )
    parser.add_argument(
        "--alphazero-checkpoint-player1",
        type=Path,
        default=None,
        help="override alphazero-checkpoint for player 1",
    )
    parser.add_argument(
        "--alphazero-checkpoint-player2",
        type=Path,
        default=None,
        help="override alphazero-checkpoint for player 2",
    )
    parser.add_argument(
        "--alphazero-compile-neural-net",
        action="store_true",
        help="compile the neural net for the AlphaZero player",
    )
    parser.add_argument(
        "--alphazero-compile-neural-net-mode",
        default="max-autotune",
        help="compilation mode for the AlphaZero player (default: max-autotune)",
    )
    parser.add_argument(
        "--egaroucid-exe",
        type=Path,
        default=None,
        help="path to the Egaroucid executable",
    )
    parser.add_argument(
        "--egaroucid-level",
        type=int,
        default=21,
        help="level for the Egaroucid player (default: 21)",
    )
    parser.add_argument(
        "--egaroucid-level-player1",
        type=int,
        default=None,
        help="override egaroucid-level for player 1",
    )
    parser.add_argument(
        "--egaroucid-level-player2",
        type=int,
        default=None,
        help="override egaroucid-level for player 2",
    )
    parser.add_argument(
        "--egaroucid-threads",
        type=int,
        default=24,
        help="number of threads for the Egaroucid player (default: 24)",
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.torch_float32_matmul_precision)

    player1 = _create_player(args, 1)
    player2 = _create_player(args, 2)
    play_game(player1, player2, quiet=False)


def _create_player(args: Namespace, player: int) -> Player:
    """Creates a player object based on the command-line arguments."""

    player_kind = args.player1 if player == 1 else args.player2
    if player_kind == "human":
        return HumanPlayer()
    if player_kind == "random":
        return RandomPlayer()
    if player_kind == "greedy":
        return GreedyPlayer()
    if player_kind == "alphazero":
        device = args.alphazero_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        num_simulations = (
            args.alphazero_simulations_player1
            if player == 1
            else args.alphazero_simulations_player2
        )
        if num_simulations is None:
            num_simulations = args.alphazero_simulations

        checkpoint_dir = (
            args.alphazero_checkpoint_player1
            if player == 1
            else args.alphazero_checkpoint_player2
        )
        if checkpoint_dir is None:
            checkpoint_dir = args.alphazero_checkpoint
        if checkpoint_dir is None:
            raise ValueError("AlphaZero checkpoint directory not specified")

        return AlphaZeroPlayer(
            device=device,
            pin_memory=args.alphazero_pin_memory,
            num_simulations=num_simulations,
            num_threads=args.alphazero_threads,
            batch_size=args.alphazero_batch_size,
            checkpoint_dir=checkpoint_dir,
            compile_neural_net=args.alphazero_compile_neural_net,
            compile_neural_net_mode=args.alphazero_compile_neural_net_mode,
            quiet=False,
        )
    if player_kind == "egaroucid":
        if args.egaroucid_exe is None:
            raise ValueError("Egaroucid executable not specified")

        level = (
            args.egaroucid_level_player1
            if player == 1
            else args.egaroucid_level_player2
        )
        if level is None:
            level = args.egaroucid_level

        return EgaroucidPlayer(args.egaroucid_exe, level, args.egaroucid_threads)

    raise ValueError(f"Invalid player kind: {player_kind}")
