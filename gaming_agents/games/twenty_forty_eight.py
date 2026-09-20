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

    def features(self, state: Game2048State) -> dict[str, float]:
        """The universal vocabulary, read onto a grid, plus what is spatial here.

        2048 is the one game in this project where memorising cannot work at
        all -- a 4x4 board essentially never comes round twice -- so it is the
        cleanest test of whether describing a position is enough on its own.

        The universal entries, applied as literally as this game allows:

        * *progress* -- how far the largest tile has come toward the target.
        * *my_strength* -- empty cells. Room to manoeuvre is the single best
          thing about a 2048 position; a full board is a lost one.
        * *their_strength* -- what is working against the player is clutter:
          neighbouring tiles of different values that cannot be merged and so
          sit in the way.
        * *win_available* -- the target tile is made. Almost always zero, which
          is honest: this game has no winning move to spot, only a long climb.
        * *must_block* -- two empty cells or fewer, where one unlucky spawn
          jams the board.

        The rest is the spatial structure a person would talk about: whether
        values run in order, and whether the biggest tile is anchored in a
        corner instead of stranded in the middle.
        """
        size = self.size
        cells = state.cells
        empty = sum(1 for value in cells if value == 0)
        largest = max(cells)

        mergeable = clutter = 0
        for row in range(size):
            for col in range(size):
                value = cells[row * size + col]
                if not value:
                    continue
                for down, across in ((1, 0), (0, 1)):
                    r, c = row + down, col + across
                    if r >= size or c >= size:
                        continue
                    neighbour = cells[r * size + c]
                    if not neighbour:
                        continue
                    if neighbour == value:
                        mergeable += 1
                    else:
                        clutter += 1
        adjacent = max(1, mergeable + clutter)

        ordered = 0
        for index in range(size):
            row = [cells[index * size + c] for c in range(size)]
            col = [cells[r * size + index] for r in range(size)]
            for line in (row, col):
                kept = [value for value in line if value]
                if len(kept) < 2 or kept == sorted(kept) or kept == sorted(kept, reverse=True):
                    ordered += 1

        corners = (0, size - 1, size * (size - 1), size * size - 1)
        cornered = 1.0 if largest and any(cells[i] == largest for i in corners) else 0.0

        return {
            "bias": 1.0,
            "progress": (largest.bit_length() - 1) / 11.0 if largest else 0.0,
            "my_strength": empty / (size * size),
            "their_strength": clutter / adjacent,
            "win_available": 1.0 if largest >= self.target else 0.0,
            "must_block": 1.0 if empty <= 2 else 0.0,
            # The spatial part: a board whose values run in order can be
            # collapsed, and a biggest tile parked in a corner stays out of
            # the way of everything else.
            "2048:ordered": ordered / (2.0 * size),
            "2048:cornered": cornered,
            "2048:mergeable": mergeable / adjacent,
        }

    #: The tile ladder. Each rung needs the one below it, so these unlock in
    #: dependency order without the game having to enforce anything.
    TILE_LADDER = (32, 64, 128, 256, 512, 1024, 2048)

    def all_achievements(self) -> tuple[str, ...]:
        return tuple(f"tile_{rung}" for rung in self.TILE_LADDER) + (
            "lasted_300_moves", "anchored_a_big_tile", "stayed_roomy_while_big",
        )

    def achievements(self, state: Game2048State) -> frozenset[str]:
        """Milestones this board has reached.

        Mostly the tile ladder, which is naturally ordered -- a 512 cannot
        happen without a 256 first. The three that are not about tiles are
        about *how* the board was played, and are the ones a score alone would
        never show: staying alive a long time, keeping the biggest tile out of
        the way, and reaching something substantial while still having room to
        move.
        """
        earned = {f"tile_{rung}" for rung in self.TILE_LADDER if max(state.cells) >= rung}

        if state.moves >= 300:
            earned.add("lasted_300_moves")

        largest = max(state.cells)
        corners = (0, self.size - 1, self.size * (self.size - 1), self.size * self.size - 1)
        if largest >= 256 and any(state.cells[i] == largest for i in corners):
            earned.add("anchored_a_big_tile")

        empty = sum(1 for value in state.cells if value == 0)
        if largest >= 256 and empty >= self.size * self.size // 2:
            earned.add("stayed_roomy_while_big")

        return frozenset(earned)

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
