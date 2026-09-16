"""The floor: an agent with no idea what it is doing.

Every learning curve in this project is measured against this, because a
win-rate only means something relative to a known-bad opponent.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..core.game import Game, Move, State
from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        return rng.choice(list(legal_moves))


class FirstMoveAgent(Agent):
    """Always takes the first legal move. Deterministic, and useful in tests."""

    name = "first"

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        return legal_moves[0]
