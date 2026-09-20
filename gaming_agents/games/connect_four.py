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
        self._window_cells = self._build_windows()
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

    def action_space(self) -> tuple[int, ...]:
        return tuple(range(self.cols))

    def encode(self, state: ConnectFourState) -> tuple[float, ...]:
        """Every cell twice over, mine then theirs, from the mover's side."""
        me = self.current_player(state)
        mine, theirs = me + 1, 2 - me
        board = self._flatten(state.columns)
        return tuple(1.0 if cell == mine else 0.0 for cell in board) + tuple(
            1.0 if cell == theirs else 0.0 for cell in board
        )

    def features(self, state: ConnectFourState) -> dict[str, float]:
        """Describe the position the way a person would, not by its identity.

        Nine numbers: how many lines each side is one move from completing, how
        many they are two moves from, whether either can win right now, who
        holds the centre, and how far into the game we are. Two boards that
        share these are strategically the same board, which is the whole point
        -- and it comes with a bonus, since a board and its mirror image
        describe identically and so can no longer be learned twice.

        One pass over the board does all of it. A near-win whose empty cell is
        the next to fill in its column *is* an immediate threat, so the winning
        and blocking features fall out of the same scan rather than costing a
        separate search.
        """
        me = self.current_player(state)
        mine, theirs = me + 1, 2 - me

        board = self._flatten(state.columns)
        heights = [len(column) for column in state.columns]

        near = {mine: 0, theirs: 0}
        now = {mine: 0, theirs: 0}
        building = {mine: 0, theirs: 0}
        almost = self.connect - 1

        for window in self._window_cells:
            held_me = held_them = 0
            gap = -1
            for cell in window:
                mark = board[cell]
                if mark == mine:
                    held_me += 1
                elif mark == theirs:
                    held_them += 1
                else:
                    gap = cell
            if held_me and held_them:
                continue  # contested, so worth nothing to either side
            if held_me:
                owner, held = mine, held_me
            elif held_them:
                owner, held = theirs, held_them
            else:
                continue
            if held == almost:
                near[owner] += 1
                # The one empty cell is reachable only if its column has
                # filled to exactly that row -- otherwise the line is a
                # promise, not a threat.
                if heights[gap // self.rows] == gap % self.rows:
                    now[owner] += 1
            elif held == almost - 1:
                building[owner] += 1

        centre = state.columns[self.cols // 2]
        cap = lambda value, scale: value / scale if value < scale else 1.0  # noqa: E731 - local shorthand

        return {
            # The universal vocabulary. "Strength" here is how close each side
            # is to completing a line, which is what a position being good
            # means on this board.
            "bias": 1.0,
            "progress": state.filled / (self.rows * self.cols),
            "my_strength": cap(near[mine], 3),
            "their_strength": cap(near[theirs], 3),
            "win_available": cap(now[mine], 1),
            "must_block": cap(now[theirs], 1),
            # Ideas every board game has, but no card game does.
            "board:building_mine": cap(building[mine], 6),
            "board:building_theirs": cap(building[theirs], 6),
            "board:centre_mine": cap(centre.count(mine), self.rows),
            "board:centre_theirs": cap(centre.count(theirs), self.rows),
        }

    def _flatten(self, columns) -> list[int]:
        """The board as one flat list, indexed ``column * rows + row``."""
        board = [0] * (self.cols * self.rows)
        for col, column in enumerate(columns):
            base = col * self.rows
            for row, mark in enumerate(column):
                board[base + row] = mark
        return board

    def _build_windows(self) -> tuple[tuple[int, ...], ...]:
        """Every run of ``connect`` cells that fits on the board, as flat indices.

        Computed once per game rather than per position: the board's geometry
        never changes, and this scan runs several times for every move played.
        """
        windows = []
        for col in range(self.cols):
            for row in range(self.rows):
                for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    end_c, end_r = col + dc * (self.connect - 1), row + dr * (self.connect - 1)
                    if 0 <= end_c < self.cols and 0 <= end_r < self.rows:
                        windows.append(
                            tuple((col + dc * i) * self.rows + (row + dr * i) for i in range(self.connect))
                        )
        return tuple(windows)

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
