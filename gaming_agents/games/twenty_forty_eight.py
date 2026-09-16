"""2048: the top of the solo track.

Slide and merge tiles; a random tile appears after every move. The state space
is hopeless to tabulate and the reward arrives in a long, noisy stream, so this
is the rung that separates agents that memorise from agents that plan.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Hashable

from ..core.game import Game, Rewards

UP, DOWN, LEFT, RIGHT = "up", "down", "left", "right"
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)


@dataclass(frozen=True, slots=True)
class Game2048State:
    #: Row-major tile exponents are avoided on purpose -- these are face values.
    cells: tuple[int, ...]
    score: int = 0
    moves: int = 0


class TwentyFortyEight(Game):
    name = "2048"
    title = "2048"
    num_players = 1
    deterministic = False
    difficulty = 5

    def __init__(self, size: int = 4, target: int = 2048) -> None:
        self.size = size
        self.target = target

    def initial_state(self, rng: random.Random) -> Game2048State:
        cells = (0,) * (self.size * self.size)
        cells = _spawn(cells, rng)
        cells = _spawn(cells, rng)
        return Game2048State(cells=cells)

    def current_player(self, state: Game2048State) -> int:
        return 0

    def legal_moves(self, state: Game2048State) -> list[str]:
        return [d for d in DIRECTIONS if _slide(state.cells, self.size, d)[0] != state.cells]

    def step(self, state: Game2048State, move: str, rng: random.Random) -> tuple[Game2048State, Rewards]:
        if move not in DIRECTIONS:
            raise ValueError(f"unknown direction {move!r}")
        cells, gained = _slide(state.cells, self.size, move)
        if cells == state.cells:
            raise ValueError(f"moving {move} changes nothing")
        cells = _spawn(cells, rng)
        nxt = Game2048State(cells=cells, score=state.score + gained, moves=state.moves + 1)
        # Scores span four orders of magnitude, so hand the learner the log of
        # the merge value instead. Raw scores would make one lucky 512-merge
        # outweigh every decision that led to it.
        reward = 0.0 if gained == 0 else gained.bit_length() / 10.0
        return nxt, (reward,)

    def is_terminal(self, state: Game2048State) -> bool:
        return not self.legal_moves(state)

    def key(self, state: Game2048State) -> Hashable:
        return state.cells

    def outcome(self, state: Game2048State) -> Rewards:
        return (float(state.score),)

    def render(self, state: Game2048State) -> str:
        lines = []
        for r in range(self.size):
            row = state.cells[r * self.size : (r + 1) * self.size]
            lines.append(" " + " ".join(f"{v or '.':>5}" for v in row))
        best = max(state.cells)
        reached = "  *** reached the target ***" if best >= self.target else ""
        return "\n".join(lines) + f"\n  score: {state.score}   best tile: {best}{reached}"

    def move_label(self, move: str) -> str:
        return move

    def parse_move(self, text: str, state: Game2048State) -> str:
        aliases = {"u": UP, "d": DOWN, "l": LEFT, "r": RIGHT, "w": UP, "s": DOWN, "a": LEFT, "e": RIGHT}
        choice = text.strip().lower()
        choice = aliases.get(choice, choice)
        if choice not in self.legal_moves(state):
            raise ValueError(f"{text!r} is not a legal direction here")
        return choice


def _rows(cells: tuple[int, ...], size: int) -> list[list[int]]:
    return [list(cells[r * size : (r + 1) * size]) for r in range(size)]


def _flatten(rows: list[list[int]]) -> tuple[int, ...]:
    return tuple(v for row in rows for v in row)


def _collapse(row: list[int]) -> tuple[list[int], int]:
    """Squash one row to the left, merging each pair at most once."""
    tiles = [v for v in row if v]
    out: list[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            out.append(merged)
            gained += merged
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (len(row) - len(out)))
    return out, gained


@lru_cache(maxsize=1 << 17)
def _slide(cells: tuple[int, ...], size: int, direction: str) -> tuple[tuple[int, ...], int]:
    """Return the board after sliding ``direction``, plus the score gained.

    Cached: search re-slides the same board thousands of times, and ``_slide``
    is pure, so memoising it is most of the difference between a playable
    search and an unusable one.
    """
    rows = _rows(cells, size)
    if direction in (UP, DOWN):
        rows = [list(col) for col in zip(*rows)]
    reverse = direction in (RIGHT, DOWN)

    gained = 0
    out = []
    for row in rows:
        if reverse:
            row = row[::-1]
        collapsed, points = _collapse(row)
        gained += points
        out.append(collapsed[::-1] if reverse else collapsed)

    if direction in (UP, DOWN):
        out = [list(col) for col in zip(*out)]
    return _flatten(out), gained


def _spawn(cells: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    """Drop a 2 (90% of the time) or a 4 into a random empty cell."""
    empty = [i for i, v in enumerate(cells) if v == 0]
    if not empty:
        return cells
    index = rng.choice(empty)
    value = 4 if rng.random() < 0.1 else 2
    return cells[:index] + (value,) + cells[index + 1 :]
