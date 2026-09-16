"""Agents that do not learn or search, because their game is already solved.

These exist to be yardsticks. A learner's score only means something next to
the best score available, and on Nim and Blackjack that best score is known
exactly rather than approximately.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..core.game import Game, Move, State
from .base import Agent


class PerfectNimAgent(Agent):
    """Plays Nim by the nim-sum rule. Unbeatable from a won position.

    Worth having separately from a search agent because it is instant and
    exact, which makes it a clean reference for whether a learner has really
    solved the first rung.
    """

    name = "perfect-nim"

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        optimal = getattr(game, "optimal_moves", None)
        if optimal is None:
            raise ValueError(f"{type(self).__name__} only plays Nim, not {game.name}")
        winning = optimal(state)
        if winning:
            return rng.choice(winning)
        # Already lost against correct play: take one object and hope.
        return min(legal_moves, key=lambda m: m[1])


class PerfectBlackjackAgent(Agent):
    """Plays the exactly correct hit/stick, solved by dynamic programming.

    It still loses money -- around 4.7 cents per hand -- because that is the
    house edge in a game with no doubling, splitting or naturals. That number
    is the ceiling, not a failure, and it is the only thing a Blackjack
    learner should be measured against.
    """

    name = "perfect-blackjack"

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        optimal = getattr(game, "optimal_move", None)
        if optimal is None:
            raise ValueError(f"{type(self).__name__} only plays Blackjack, not {game.name}")
        return optimal(state)
