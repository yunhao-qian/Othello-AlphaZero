"""Utility functions for evaluating the performance of players."""

import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from .play import play_game
from .player import Player


def play_games(
    game_results_file: str | os.PathLike,
    player_ids: Sequence[str],
    create_player_fn: Callable[[str], Player],
    max_games_per_pair: int = 2,
    callback: Callable[[list[dict[str, str | int]]], None] | None = None,
) -> None:
    """Plays games between pairs of players and record the results.

    :param game_results_file: Path to the file where the game results are stored.
    :param player_ids: List of player IDs.
    :param create_player_fn: Function that creates a player given an ID.
    :param max_games_per_pair: Maximum number of games to play between each pair of
        players.
    :param callback: Optional callback function that is called after each pair of games.
    """

    game_results_file = Path(game_results_file)
    if game_results_file.exists():
        with game_results_file.open(encoding="utf-8") as file:
            game_results = json.load(file)
    else:
        game_results = []

    sample_failures = 0

    def should_play_game(player1_id: str, player2_id: str) -> bool:
        player_id_pair = tuple(sorted([player1_id, player2_id]))
        count = 0

        for result in game_results:
            if tuple(sorted([result["player1"], result["player2"]])) == player_id_pair:
                count += 1
                if count >= max_games_per_pair:
                    return False

        return True

    while True:
        player1_id, player2_id = sorted(random.sample(player_ids, 2))
        if not should_play_game(player1_id, player2_id):
            sample_failures += 1
            if sample_failures > 10000:
                break
            continue
        sample_failures = 0

        print(f"Playing games between '{player1_id}' and '{player2_id}'")
        player1 = create_player_fn(player1_id)
        player2 = create_player_fn(player2_id)

        result = play_game(player1, player2, quiet=True)
        print(("Draw", f"'{player1_id}' wins", f"'{player2_id}' wins")[result])
        game_results.append(
            {
                "player1": player1_id,
                "player2": player2_id,
                "result": result,
            }
        )

        result = play_game(player2, player1, quiet=True)
        print(("Draw", f"'{player2_id}' wins", f"'{player1_id}' wins")[result])
        game_results.append(
            {
                "player1": player2_id,
                "player2": player1_id,
                "result": result,
            }
        )

        with game_results_file.open("w", encoding="utf-8") as file:
            json.dump(game_results, file, indent=4)

        if callback is not None:
            callback(game_results)


def estimate_elo(
    game_results: Sequence[Mapping[str, str | int]],
    optimizer_lr: float = 0.01,
    optimization_steps: int = 4000,
) -> dict[str, float]:
    """Estimates Elo ratings from game results.

    :param game_results: List of game results, where each result is a dictionary with
        keys "player1", "player2", and "result".
    :param optimizer_lr: Learning rate of the optimizer.
    :param optimization_steps: Number of optimization steps.
    :return: Dictionary mapping player IDs to Elo ratings.
    """

    player_ids = set()
    for result in game_results:
        player_ids.add(result["player1"])
        player_ids.add(result["player2"])
    player_ids = sorted(player_ids)

    player1_indices = []
    player2_indices = []
    results = []
    for result in game_results:
        player1_indices.append(player_ids.index(result["player1"]))
        player2_indices.append(player_ids.index(result["player2"]))
        results.append(result["result"])
    player1_indices = torch.tensor(player1_indices)
    player2_indices = torch.tensor(player2_indices)
    results = torch.tensor(results)

    ratings = torch.randn(len(player_ids), requires_grad=True)
    elo_advantage = torch.randn((), requires_grad=True)
    elo_draw = torch.randn((), requires_grad=True)

    optimizer = torch.optim.Adam([ratings, elo_advantage, elo_draw], lr=optimizer_lr)

    for i in range(optimization_steps):
        player1_ratings = ratings[player1_indices]
        player2_ratings = ratings[player2_indices]
        player1_probabilities = 1 / (
            1 + 10 ** (player2_ratings - player1_ratings - elo_advantage + elo_draw)
        )
        player2_probabilities = 1 / (
            1 + 10 ** (player1_ratings - player2_ratings + elo_advantage + elo_draw)
        )
        draw_probabilities = 1 - player1_probabilities - player2_probabilities
        probabilities = torch.where(
            results == 1,
            player1_probabilities,
            torch.where(results == 2, player2_probabilities, draw_probabilities),
        )
        loss = -probabilities.log().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % (optimization_steps // 10) == 0:
            print(f"Loss: {loss.item()}")

    ratings = ratings.detach().numpy() * 400
    return {player_id: float(rating) for player_id, rating in zip(player_ids, ratings)}
