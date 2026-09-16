"""A keyboard-driven 'agent', so you can sit down opposite one of the learners."""

from __future__ import annotations

import random
from typing import Callable, Sequence

from ..core.game import Game, Move, State
from .base import Agent


class HumanAgent(Agent):
    name = "human"

    def __init__(self, name: str | None = None, *, prompt: Callable[[str], str] = input) -> None:
        super().__init__(name)
        self.prompt = prompt

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        print(game.render(state))
        options = ", ".join(game.move_label(m) for m in legal_moves)
        while True:
            try:
                text = self.prompt(f"  your move [{options}]: ")
            except EOFError:
                print("\n  no input; playing a random legal move")
                return rng.choice(list(legal_moves))
            try:
                return game.parse_move(text, state)
            except ValueError as exc:
                print(f"  {exc}")
