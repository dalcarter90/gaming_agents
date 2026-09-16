"""Connect Four, with the board size left configurable.

The standard 6x7 board is far too big to tabulate, so this is the rung where a
purely table-based learner stops being enough and search starts to matter. The
smaller ``connect4-mini`` variant sits in between: still unlearnable cell by
cell, but small enough that a few thousand self-play games show real progress.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Hashable

from ..core.game import Game, Rewards

MARKS = (".", "X", "O")


@dataclass(frozen=True, slots=True)
class ConnectFourState:
    #: Column-major: ``columns[c]`` is that column's discs, bottom first.
    columns: tuple[tuple[int, ...], ...]
    player: int
    winner: int | None = None
    filled: int = 0


class ConnectFour(Game):
    name = "connect4"
    title = "Connect Four"
    num_players = 2
    deterministic = True
    difficulty = 4

    def __init__(self, rows: int = 6, cols: int = 7, connect: int = 4, name: str | None = None) -> None:
        if connect > max(rows, cols):
            raise ValueError("connect length does not fit on the board")
        self.rows = rows
        self.cols = cols
        self.connect = connect
        if name:
            self.name = name
            self.title = f"Connect Four {rows}x{cols}"

    def initial_state(self, rng: random.Random) -> ConnectFourState:
        return ConnectFourState(columns=tuple(() for _ in range(self.cols)), player=0)

    def current_player(self, state: ConnectFourState) -> int:
        return state.player

    def legal_moves(self, state: ConnectFourState) -> list[int]:
        if state.winner is not None:
            return []
        return [c for c in range(self.cols) if len(state.columns[c]) < self.rows]

    def step(self, state: ConnectFourState, move: int, rng: random.Random) -> tuple[ConnectFourState, Rewards]:
        if not 0 <= move < self.cols:
            raise ValueError(f"column {move} is off the board")
        column = state.columns[move]
        if len(column) >= self.rows:
            raise ValueError(f"column {move} is full")

        mark = state.player + 1
        columns = state.columns[:move] + (column + (mark,),) + state.columns[move + 1 :]
        row = len(column)
        winner = state.player if self._wins(columns, move, row, mark) else None

        nxt = ConnectFourState(
            columns=columns,
            player=1 - state.player,
            winner=winner,
            filled=state.filled + 1,
        )
        if winner is None:
            return nxt, (0.0, 0.0)
        rewards = [0.0, 0.0]
        rewards[winner] = 1.0
        rewards[1 - winner] = -1.0
        return nxt, tuple(rewards)

    def is_terminal(self, state: ConnectFourState) -> bool:
        return state.winner is not None or state.filled >= self.rows * self.cols

    def key(self, state: ConnectFourState) -> Hashable:
        # A board and its left-right mirror are the same position, so store
        # whichever ordering sorts first and let both share what is learned.
        mirrored = state.columns[::-1]
        canonical = min(state.columns, mirrored)
        return (canonical, state.player)

    def outcome(self, state: ConnectFourState) -> Rewards:
        if state.winner is None:
            return (0.0, 0.0)
        return (1.0, -1.0) if state.winner == 0 else (-1.0, 1.0)

    def heuristic(self, state: ConnectFourState, seat: int) -> float:
        """Count how many ``connect``-length windows each side could still fill.

        Deliberately cheap: depth-limited search calls this at every leaf, and
        a rough ordering of positions is worth far more than an exact one.
        """
        if state.winner is not None:
            return 1.0 if state.winner == seat else -1.0
        mark = seat + 1
        total = 0.0
        windows = 0
        for col in range(self.cols):
            for row in range(self.rows):
                for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    end_c, end_r = col + dc * (self.connect - 1), row + dr * (self.connect - 1)
                    if not (0 <= end_c < self.cols and 0 <= end_r < self.rows):
                        continue
                    window = tuple(
                        self._at(state.columns, col + dc * i, row + dr * i) for i in range(self.connect)
                    )
                    total += _score_window(window, mark)
                    windows += 1
        if not windows:
            return 0.0
        # Squeeze into (-1, 1) so it can never outrank a real win or loss.
        return max(-0.99, min(0.99, total / windows * 4.0))

    def render(self, state: ConnectFourState) -> str:
        lines = []
        for r in range(self.rows - 1, -1, -1):
            cells = []
            for c in range(self.cols):
                column = state.columns[c]
                cells.append(MARKS[column[r]] if r < len(column) else ".")
            lines.append(" " + " ".join(cells))
        lines.append(" " + " ".join(str(c) for c in range(self.cols)))
        return "\n".join(lines) + f"\n  to move: {MARKS[state.player + 1]} (player {state.player})"

    def parse_move(self, text: str, state: ConnectFourState) -> int:
        try:
            move = int(text.strip())
        except ValueError:
            raise ValueError(f"cannot read {text!r} as a column number") from None
        if move not in self.legal_moves(state):
            raise ValueError(f"column {move} is not playable")
        return move

    # -- win detection ----------------------------------------------------

    def _at(self, columns: tuple[tuple[int, ...], ...], col: int, row: int) -> int:
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return -1
        column = columns[col]
        return column[row] if row < len(column) else 0

    def _wins(self, columns, col: int, row: int, mark: int) -> bool:
        """Whether the disc just dropped at ``(col, row)`` completes a line."""
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            run = 1
            for direction in (1, -1):
                c, r = col + dc * direction, row + dr * direction
                while self._at(columns, c, r) == mark:
                    run += 1
                    if run >= self.connect:
                        return True
                    c += dc * direction
                    r += dr * direction
        return False


class ConnectFourMini(ConnectFour):
    """A 4x5 board: the gentler step up from Tic-Tac-Toe."""

    name = "connect4-mini"
    title = "Connect Four (mini 4x5)"
    difficulty = 3

    def __init__(self) -> None:
        super().__init__(rows=4, cols=5, connect=4)
        self.name = "connect4-mini"
        self.title = "Connect Four (mini 4x5)"


def _score_window(window: tuple[int, ...], mark: int) -> float:
    """Value a group of ``connect`` consecutive cells for the player ``mark``."""
    mine = window.count(mark)
    theirs = window.count(3 - mark)
    if mine and theirs:
        return 0.0  # blocked in both directions, worth nothing to either side
    if mine:
        return (mine / len(window)) ** 2
    if theirs:
        return -((theirs / len(window)) ** 2)
    return 0.0
