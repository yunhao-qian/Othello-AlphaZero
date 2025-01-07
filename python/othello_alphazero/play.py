"""Script for playing a game of Othello between humans and agents."""

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from othello_mcts import MCTS, Position

from .neural_net import AlphaZeroNet


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
        help="precision for float32 matrix multiplication in PyTorch (default: highest)",
    )
    parser.add_argument(
        "--alphazero-device",
        default=None,
        help="device for the AlphaZero player (default: cuda if available, else cpu)",
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
        "--alphazero-batch-size",
        type=int,
        default=16,
        help="batch size for the AlphaZero player (default: 16)",
    )
    parser.add_argument(
        "--alphazero-threads",
        type=int,
        default=24,
        help="number of threads for the AlphaZero player (default: 24)",
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

    position = Position.initial_position()
    while True:
        print(position)
        num_black_discs = position.num_p1_discs()
        num_white_discs = position.num_p2_discs()
        print(f"Black: {num_black_discs}, White: {num_white_discs}")
        if position.is_terminal():
            break

        if position.player() == 1:
            print("Black's turn")
            player = player1
        else:
            print("White's turn")
            player = player2

        legal_actions = position.legal_actions()
        print(
            "Legal actions:",
            ", ".join(_ACTION_NAMES[action] for action in legal_actions),
        )
        action = player.get_action()
        print("Player action:", _ACTION_NAMES[action])

        position = position.apply_action(action)
        player1.apply_action(action)
        player2.apply_action(action)

        print()

    print("Game over")
    if num_black_discs > num_white_discs:
        print("Black wins")
    elif num_white_discs > num_black_discs:
        print("White wins")
    else:
        print("Draw")


class Player(ABC):
    """Abstract base class for a player in a game."""

    @abstractmethod
    def get_action(self) -> int:
        """Returns the player's action for the current position."""

    @abstractmethod
    def apply_action(self, action: int) -> None:
        """Updates the current position with the player or opponent's action."""


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
            num_simulations=num_simulations,
            batch_size=args.alphazero_batch_size,
            num_threads=args.alphazero_threads,
            checkpoint_dir=checkpoint_dir,
            compile_neural_net=args.alphazero_compile_neural_net,
            compile_neural_net_mode=args.alphazero_compile_neural_net_mode,
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


class HumanPlayer(Player):
    """Player that prompts the user to select actions."""

    def __init__(self) -> None:
        self.position = Position.initial_position()

    def get_action(self) -> int:
        legal_actions = self.position.legal_actions()
        legal_action_names = [_ACTION_NAMES[action] for action in legal_actions]
        while True:
            print("Enter your action: ", end="")
            action_name = input().strip()
            if action_name in legal_action_names:
                return _ACTION_NAMES.index(action_name)
            print("Invalid action")

    def apply_action(self, action: int) -> None:
        self.position = self.position.apply_action(action)


class RandomPlayer(Player):
    """Player that selects actions uniformly at random."""

    def __init__(self) -> None:
        self.position = Position.initial_position()

    def get_action(self) -> int:
        return int(np.random.choice(self.position.legal_actions()))

    def apply_action(self, action: int) -> None:
        self.position = self.position.apply_action(action)


class GreedyPlayer(Player):
    """Player that selects the action with the most flipped discs."""

    def __init__(self) -> None:
        self.position = Position.initial_position()

    def get_action(self) -> int:
        legal_actions = self.position.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        all_num_flips = [self.position.num_flips(action) for action in legal_actions]
        max_num_flips = max(all_num_flips)
        best_actions = [
            action
            for action, num_flips in zip(legal_actions, all_num_flips)
            if num_flips == max_num_flips
        ]
        # Break ties randomly.
        return int(np.random.choice(best_actions))

    def apply_action(self, action: int) -> None:
        self.position = self.position.apply_action(action)


class AlphaZeroPlayer(Player):
    """Player that selects actions using the AlphaZero algorithm."""

    def __init__(
        self,
        device: str,
        num_simulations: int,
        batch_size: int,
        num_threads: int,
        checkpoint_dir: str | os.PathLike,
        compile_neural_net: bool,
        compile_neural_net_mode: str,
    ) -> None:
        self.mcts = MCTS(
            torch_device=device,
            num_simulations=num_simulations,
            batch_size=batch_size,
            num_threads=num_threads,
            exploration_weight=0.0,
            dirichlet_epsilon=0.0,
        )

        checkpoint_dir = Path(checkpoint_dir)
        with (checkpoint_dir / "config.json").open() as config_file:
            config = json.load(config_file)

        self.neural_net = AlphaZeroNet(**config["neural_net"]).to(device)
        self.neural_net.load_state_dict(
            torch.load(
                checkpoint_dir / "neural_net.pth",
                map_location=device,
                weights_only=True,
            )
        )
        self.neural_net.eval()
        self.neural_net.requires_grad_(False)

        if compile_neural_net:
            self.neural_net = torch.compile(
                self.neural_net,
                fullgraph=True,
                dynamic=False,
                mode=compile_neural_net_mode,
            )
            dummy_input = torch.zeros(
                (min(batch_size, num_threads), 3, 8, 8), device=device
            )
            self.neural_net(dummy_input)

    def get_action(self) -> int:
        search_result = self.mcts.search(self.neural_net)
        action_index = np.argmax(search_result["visit_counts"])
        action_value = search_result["mean_action_values"][action_index]
        print(f"Action value: {action_value:.3f}")
        return search_result["actions"][action_index]

    def apply_action(self, action: int) -> None:
        self.mcts.apply_action(action)


class EgaroucidPlayer(Player):
    """Player that calls the Egaroucid application to select actions."""

    def __init__(
        self, egaroucid_exe: str | os.PathLike, level: int, num_threads: int
    ) -> None:
        self.egaroucid_exe = str(egaroucid_exe)
        self.level = level
        self.num_threads = num_threads

        self.position = Position.initial_position()

    def get_action(self) -> int:
        legal_actions = self.position.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        with tempfile.NamedTemporaryFile("w+") as problem_file:
            for row in range(8):
                for col in range(8):
                    status = self.position(row, col)
                    if status == 1:
                        problem_file.write("B")
                    elif status == 2:
                        problem_file.write("W")
                    else:
                        problem_file.write(".")
            if self.position.player() == 1:
                problem_file.write("B")
            else:
                problem_file.write("W")
            problem_file.write("\n")
            problem_file.flush()

            output = subprocess.run(
                [
                    self.egaroucid_exe,
                    "-level",
                    str(self.level),
                    "-nobook",
                    "-threads",
                    str(self.num_threads),
                    "-solve",
                    problem_file.name,
                ],
                capture_output=True,
                check=True,
                text=True,
            ).stdout

        # Line format: | <level> | <depth> | <move> | <score> | ...
        line = output.splitlines()[1]
        action_name = line.split("|")[3].strip()
        return _ACTION_NAMES.index(action_name)

    def apply_action(self, action: int) -> None:
        self.position = self.position.apply_action(action)


def _get_action_names() -> list[str]:
    row_names = "12345678"
    col_names = "abcdefgh"
    action_names = []
    for row_name in row_names:
        for col_name in col_names:
            action_names.append(col_name + row_name)
    action_names.append("pass")
    return action_names


_ACTION_NAMES = _get_action_names()
