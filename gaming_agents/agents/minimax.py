"""Alpha-beta search: the yardstick the curriculum measures mastery against.

On Nim and Tic-Tac-Toe the search is exhaustive, so this agent plays perfectly
and "can the learner avoid losing to it" is a well-defined bar. On bigger
boards it is depth-limited and leans on the game's heuristic, which makes it a
strong-but-beatable sparring partner rather than an oracle.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..core.game import Game, Move, State
from .base import Agent

INFINITY = float("inf")


class MinimaxAgent(Agent):
    name = "minimax"

    def __init__(self, name: str | None = None, *, depth: int = 12, randomize: bool = True) -> None:
        super().__init__(name)
        self.depth = depth
        self.randomize = randomize
        """Choose randomly among equally good moves. Keeps training opponents
        from playing the identical game every time."""
        self._cache: dict[tuple, float] = {}

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        if game.num_players != 2:
            raise ValueError(f"{type(self).__name__} only handles two-player games, not {game.name}")
        if len(legal_moves) == 1:
            return legal_moves[0]

        seat = game.current_player(state)
        self._cache.clear()
        best_value = -INFINITY
        best: list[Move] = []
        for move in legal_moves:
            child, _ = game.step(state, move, rng)
            value = -self._search(game, child, self.depth - 1, -INFINITY, INFINITY, 1 - seat, rng)
            if value > best_value:
                best_value, best = value, [move]
            elif value == best_value:
                best.append(move)
        return rng.choice(best) if self.randomize else best[0]

    def _search(
        self,
        game: Game,
        state: State,
        depth: int,
        alpha: float,
        beta: float,
        seat: int,
        rng: random.Random,
    ) -> float:
        """Negamax value of ``state`` from ``seat``'s point of view."""
        if game.is_terminal(state):
            # Prefer a win now over a win later, and a loss later over a loss
            # now, so the agent finishes won games instead of shuffling.
            return game.outcome(state)[seat] * (1.0 + depth * 1e-4)
        if depth <= 0:
            return game.heuristic(state, seat)

        cache_key = (game.key(state), depth, seat)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        original_alpha = alpha
        best = -INFINITY
        for move in game.legal_moves(state):
            child, _ = game.step(state, move, rng)
            value = -self._search(game, child, depth - 1, -beta, -alpha, 1 - seat, rng)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # the opponent would never allow this line

        # Only cache values the window did not clip. A score that triggered a
        # cutoff, or that never beat alpha, is a bound rather than the true
        # value, and storing it as exact quietly corrupts every later lookup.
        if original_alpha < best < beta:
            self._cache[cache_key] = best
        return best

    def stats(self) -> dict[str, int]:
        return {"depth": self.depth}


class PerfectNimAgent(Agent):
    """Plays Nim by the nim-sum rule. Unbeatable from a won position.

    Worth having separately from :class:`MinimaxAgent` because it is instant
    and exact, which makes it a clean reference when checking whether a
    learner has really solved the first rung.
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
