"""Player classes and function for playing games between two players."""

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from othello_mcts import MCTS, Position

from .neural_net import AlphaZeroNet


class Player(ABC):
    """Abstract base class for a player in a game."""

    @abstractmethod
    def reset_position(self) -> None:
        """Resets the current position to the initial position."""

    @abstractmethod
    def get_action(self) -> int:
        """Returns the player's action for the current position."""

    @abstractmethod
    def apply_action(self, action: int) -> None:
        """Updates the current position with the player or opponent's action."""


def play_game(player1: Player, player2: Player, quiet: bool) -> int:
    """Plays a game between two players.

    :param player1: Black player.
    :param player2: White player.
    :param quiet: Whether to suppress output.
    :return: Winner of the game (1 for Black, 2 for White, 0 for a draw).
    """

    player1.reset_position()
    player2.reset_position()

    position = Position.initial_position()
    while True:
        if not quiet:
            print(position)
            num_black_discs = position.num_p1_discs()
            num_white_discs = position.num_p2_discs()
            print(f"Black: {num_black_discs}, White: {num_white_discs}")
        if position.is_terminal():
            break

        if position.player() == 1:
            if not quiet:
                print("Black's turn")
            player = player1
        else:
            if not quiet:
                print("White's turn")
            player = player2

        if not quiet:
            legal_actions = position.legal_actions()
            print(
                "Legal actions:",
                ", ".join(_ACTION_NAMES[action] for action in legal_actions),
            )
        action = player.get_action()
        if not quiet:
            print("Player action:", _ACTION_NAMES[action])

        position = position.apply_action(action)
        player1.apply_action(action)
        player2.apply_action(action)

        if not quiet:
            print()

    if not quiet:
        print("Game over")
    num_black_discs = position.num_p1_discs()
    num_white_discs = position.num_p2_discs()
    if num_black_discs > num_white_discs:
        if not quiet:
            print("Black wins")
        return 1
    if num_white_discs > num_black_discs:
        if not quiet:
            print("White wins")
        return 2
    if not quiet:
        print("Draw")
    return 0


class HumanPlayer(Player):
    """Player that prompts the user to select actions."""

    def __init__(self) -> None:
        self.position = Position.initial_position()

    def reset_position(self) -> None:
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

    def reset_position(self) -> None:
        self.position = Position.initial_position()

    def get_action(self) -> int:
        return int(np.random.choice(self.position.legal_actions()))

    def apply_action(self, action: int) -> None:
        self.position = self.position.apply_action(action)


class GreedyPlayer(Player):
    """Player that selects the action with the most flipped discs."""

    def __init__(self) -> None:
        self.position = Position.initial_position()

    def reset_position(self) -> None:
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
        quiet: bool,
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

        self.quiet = quiet

    def reset_position(self) -> None:
        self.mcts.reset_position(Position.initial_position())

    def get_action(self) -> int:
        search_result = self.mcts.search(self.neural_net)
        action_index = np.argmax(search_result["visit_counts"])
        action_value = search_result["mean_action_values"][action_index]
        if not self.quiet:
            print(f"Action-value: {action_value:.3f}")
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

    def reset_position(self) -> None:
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


def _create_action_names() -> tuple[str, ...]:
    row_names = "12345678"
    col_names = "abcdefgh"
    action_names = []
    for row_name in row_names:
        for col_name in col_names:
            action_names.append(col_name + row_name)
    action_names.append("pass")
    return tuple(action_names)


_ACTION_NAMES = _create_action_names()
