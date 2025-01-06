"""Script for playing a game of Othello between humans and agents."""

import json
import os
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
        choices=["human", "random", "greedy", "alphazero"],
        help="kind of player for the Black player (default: human)",
    )
    parser.add_argument(
        "--player2",
        default="human",
        choices=["human", "random", "greedy", "alphazero"],
        help="kind of player for the White player (default: human)",
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

    args = parser.parse_args()
    player1 = _create_player(args, 1)
    player2 = _create_player(args, 2)

    position = Position.initial_position()
    while True:
        print(position)
        black_discs, white_discs = _count_discs(position)
        print(f"Black: {black_discs}, White: {white_discs}")
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
    if black_discs > white_discs:
        print("Black wins")
    elif white_discs > black_discs:
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

    # AlphaZero player

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

    return AlphaZeroPlayer(
        device=device,
        num_simulations=num_simulations,
        batch_size=args.alphazero_batch_size,
        num_threads=args.alphazero_threads,
        checkpoint_dir=checkpoint_dir,
        compile_neural_net=args.alphazero_compile_neural_net,
        compile_neural_net_mode=args.alphazero_compile_neural_net_mode,
    )


def _count_discs(position: Position) -> tuple[int, int]:
    black_discs = 0
    white_discs = 0
    for row in range(8):
        for col in range(8):
            status = position(row, col)
            if status == 1:
                black_discs += 1
            elif status == 2:
                white_discs += 1
    return black_discs, white_discs


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

        best_action = None
        best_num_flips = 0
        for action in legal_actions:
            old_black_discs, old_white_discs = _count_discs(self.position)
            new_position = self.position.apply_action(action)
            new_black_discs, new_white_discs = _count_discs(new_position)
            if self.position.player() == 1:
                num_flips = old_white_discs - new_white_discs
            else:
                num_flips = old_black_discs - new_black_discs
            if num_flips > best_num_flips:
                best_action = action
                best_num_flips = num_flips
        return best_action

    def apply_action(self, action: int) -> None:
        self.position = self.position.apply_action(action)


class AlphaZeroPlayer(Player):
    """Player that uses Monte Carlo Tree Search to select actions."""

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
