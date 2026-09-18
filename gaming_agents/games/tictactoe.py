"""Tic-Tac-Toe: small enough to solve, big enough to need real credit assignment.

Rung two of the ladder. A correct learner ends up drawing every game against
perfect play, which is the mastery bar the curriculum checks for.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Hashable

from ..core.game import Game, Rewards

MARKS = (".", "X", "O")

LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)


@dataclass(frozen=True, slots=True)
class TicTacToeState:
    #: 9 cells, row-major; 0 empty, 1 for seat 0 (X), 2 for seat 1 (O).
    cells: tuple[int, ...]
    player: int
    winner: int | None = None
    """Seat that has three in a row, or None while nobody has."""


class TicTacToe(Game):
    name = "tictactoe"
    title = "Tic-Tac-Toe"
    num_players = 2
    deterministic = True
    difficulty = 2

    def initial_state(self, rng: random.Random) -> TicTacToeState:
        return TicTacToeState(cells=(0,) * 9, player=0)

    def current_player(self, state: TicTacToeState) -> int:
        return state.player

    def legal_moves(self, state: TicTacToeState) -> list[int]:
        if state.winner is not None:
            return []
        return [i for i, c in enumerate(state.cells) if c == 0]

    def step(self, state: TicTacToeState, move: int, rng: random.Random) -> tuple[TicTacToeState, Rewards]:
        if not 0 <= move < 9 or state.cells[move] != 0:
            raise ValueError(f"cell {move} is not empty")
        mark = state.player + 1
        cells = state.cells[:move] + (mark,) + state.cells[move + 1 :]
        winner = _winner(cells)

        nxt = TicTacToeState(cells=cells, player=1 - state.player, winner=winner)
        if winner is None:
            return nxt, (0.0, 0.0)
        rewards = [0.0, 0.0]
        rewards[winner] = 1.0
        rewards[1 - winner] = -1.0
        return nxt, tuple(rewards)

    def is_terminal(self, state: TicTacToeState) -> bool:
        return state.winner is not None or 0 not in state.cells

    def key(self, state: TicTacToeState) -> Hashable:
        return (state.cells, state.player)

    def outcome(self, state: TicTacToeState) -> Rewards:
        if state.winner is None:
            return (0.0, 0.0)
        return (1.0, -1.0) if state.winner == 0 else (-1.0, 1.0)

    def features(self, state: TicTacToeState) -> dict[str, float]:
        """The same idea as Connect Four's, on eight lines instead of hundreds.

        Deliberately the same *words* too. "One move from winning" means the
        same thing on both boards, so naming it the same way is what lets an
        agent carry an opinion about it from one game to the other.

        Because a description counts lines rather than naming squares, it is
        identical for a board and all seven of its rotations and reflections --
        so the 86% of the lookup table that was duplicated corners and edges
        simply stops existing.
        """
        me = state.player
        mine, theirs = me + 1, 2 - me

        one_away = {mine: 0, theirs: 0}
        started = {mine: 0, theirs: 0}
        for a, b, c in LINES:
            line = (state.cells[a], state.cells[b], state.cells[c])
            held_by_me, held_by_them = line.count(mine), line.count(theirs)
            if held_by_me and held_by_them:
                continue
            owner, held = (mine, held_by_me) if held_by_me else (theirs, held_by_them)
            if held == 2:
                one_away[owner] += 1
            elif held == 1:
                started[owner] += 1

        corners = (0, 2, 6, 8)
        squeeze = lambda value, scale: min(1.0, value / scale)  # noqa: E731 - local shorthand

        return {
            "bias": 1.0,
            "progress": sum(1 for cell in state.cells if cell) / 9.0,
            "my_strength": squeeze(one_away[mine], 2),
            "their_strength": squeeze(one_away[theirs], 2),
            "win_available": squeeze(one_away[mine], 1),
            "must_block": squeeze(one_away[theirs], 1),
            "board:building_mine": squeeze(started[mine], 4),
            "board:building_theirs": squeeze(started[theirs], 4),
            "board:centre_mine": 1.0 if state.cells[4] == mine else 0.0,
            "board:centre_theirs": 1.0 if state.cells[4] == theirs else 0.0,
            # Particular to this board.
            "tictactoe:corners_mine": squeeze(sum(state.cells[i] == mine for i in corners), 4),
            "tictactoe:corners_theirs": squeeze(sum(state.cells[i] == theirs for i in corners), 4),
        }

    def render(self, state: TicTacToeState) -> str:
        rows = []
        for r in range(3):
            cells = []
            for c in range(3):
                i = r * 3 + c
                cells.append(MARKS[state.cells[i]] if state.cells[i] else str(i))
            rows.append(" " + " | ".join(cells))
        board = "\n---+---+---\n".join(rows)
        return f"{board}\n  to move: {MARKS[state.player + 1]} (player {state.player})"

    def move_label(self, move: int) -> str:
        return str(move)

    def parse_move(self, text: str, state: TicTacToeState) -> int:
        try:
            move = int(text.strip())
        except ValueError:
            raise ValueError(f"cannot read {text!r} as a cell index 0-8") from None
        if move not in self.legal_moves(state):
            raise ValueError(f"cell {move} is not available")
        return move


def _winner(cells: tuple[int, ...]) -> int | None:
    for a, b, c in LINES:
        mark = cells[a]
        if mark and mark == cells[b] == cells[c]:
            return mark - 1
    return None
