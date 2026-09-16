"""Nim: the first rung of the ladder.

Players alternate removing any number of objects from a single heap. Whoever
takes the last object wins (normal play). The state space is tiny and the game
is solved -- the winning strategy is to leave a zero nim-sum -- which makes it
the ideal place to check that a learner actually learns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import reduce
from operator import xor
from typing import Hashable

from ..core.game import Game, Rewards

#: A move is "take ``count`` objects from some heap of size ``heap_size``".
#: Describing moves by heap *size* rather than heap *index* folds away the
#: symmetry between identical heaps for free, which shrinks what a table-based
#: agent has to learn.
Move = tuple[int, int]


@dataclass(frozen=True, slots=True)
class NimState:
    heaps: tuple[int, ...]
    player: int

    def __str__(self) -> str:
        return " | ".join(f"{'*' * n or '-'}({n})" for n in self.heaps)


class Nim(Game):
    name = "nim"
    title = "Nim"
    num_players = 2
    deterministic = True
    difficulty = 1

    def __init__(self, heaps: tuple[int, ...] = (1, 3, 5, 7)) -> None:
        if not heaps or any(h < 0 for h in heaps):
            raise ValueError("heaps must be non-empty and non-negative")
        self.heaps = tuple(heaps)

    def initial_state(self, rng: random.Random) -> NimState:
        return NimState(heaps=self.heaps, player=0)

    def current_player(self, state: NimState) -> int:
        return state.player

    def legal_moves(self, state: NimState) -> list[Move]:
        sizes = sorted({h for h in state.heaps if h > 0})
        return [(size, count) for size in sizes for count in range(1, size + 1)]

    def step(self, state: NimState, move: Move, rng: random.Random) -> tuple[NimState, Rewards]:
        heap_size, count = move
        heaps = list(state.heaps)
        try:
            index = heaps.index(heap_size)
        except ValueError:
            raise ValueError(f"no heap of size {heap_size} in {state.heaps}") from None
        if not 1 <= count <= heap_size:
            raise ValueError(f"cannot take {count} from a heap of {heap_size}")
        heaps[index] -= count

        nxt = NimState(heaps=tuple(heaps), player=1 - state.player)
        if any(heaps):
            return nxt, (0.0, 0.0)
        # The move emptied the last heap, so the mover wins.
        rewards = [0.0, 0.0]
        rewards[state.player] = 1.0
        rewards[1 - state.player] = -1.0
        return nxt, tuple(rewards)

    def is_terminal(self, state: NimState) -> bool:
        return not any(state.heaps)

    def key(self, state: NimState) -> Hashable:
        return (tuple(sorted(state.heaps)), state.player)

    def outcome(self, state: NimState) -> Rewards:
        # At a terminal state the player to move is the one who just lost.
        return (-1.0, 1.0) if state.player == 0 else (1.0, -1.0)

    def render(self, state: NimState) -> str:
        rows = [f"  heap {i}: {'*' * n:<8} ({n})" for i, n in enumerate(state.heaps)]
        return "\n".join(rows) + f"\n  to move: player {state.player}"

    def move_label(self, move: Move) -> str:
        heap_size, count = move
        return f"take{count}from{heap_size}"

    def parse_move(self, text: str, state: NimState) -> Move:
        """Accept both ``take2from5`` and the friendlier ``5 2``."""
        cleaned = text.strip().lower().replace("take", " ").replace("from", " ")
        parts = cleaned.replace(",", " ").split()
        if len(parts) == 2:
            try:
                count, heap_size = int(parts[0]), int(parts[1])
            except ValueError:
                raise ValueError(f"cannot read {text!r} as a Nim move") from None
            for candidate in ((heap_size, count), (count, heap_size)):
                if candidate in self.legal_moves(state):
                    return candidate
        raise ValueError(f"{text!r} is not a legal move")

    # -- solved-game helper, used by the perfect-play baseline -------------

    @staticmethod
    def nim_sum(heaps: tuple[int, ...]) -> int:
        return reduce(xor, heaps, 0)

    def optimal_moves(self, state: NimState) -> list[Move]:
        """Every move that leaves the opponent a losing position.

        Empty when the position is already lost with correct play.
        """
        winning = []
        for move in self.legal_moves(state):
            heap_size, count = move
            heaps = list(state.heaps)
            heaps[heaps.index(heap_size)] -= count
            if self.nim_sum(tuple(heaps)) == 0:
                winning.append(move)
        return winning
