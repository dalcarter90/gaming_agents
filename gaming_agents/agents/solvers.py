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


class NashKuhnAgent(Agent):
    """The analytically known equilibrium for Kuhn poker.

    Kuhn solved this game in 1950. The second player's equilibrium strategy is
    unique; the first player's is a one-parameter family in ``alpha``, every
    member of which is equally unexploitable. The value of the game to the
    first player is -1/18: position is worth something, and no amount of skill
    takes that back.

    Unlike every other reference agent here it plays a *mixed* strategy, and it
    has to. A deterministic poker player can be read, and anything readable can
    be exploited -- which is the lesson of this rung.
    """

    name = "nash-kuhn"

    def __init__(self, name: str | None = None, *, alpha: float = 1 / 3) -> None:
        super().__init__(name)
        if not 0.0 <= alpha <= 1 / 3:
            raise ValueError(f"alpha must lie in [0, 1/3]; {alpha} is outside the equilibrium family")
        self.alpha = alpha
        """How often the first player bluffs a Jack. Every value in range is an
        equilibrium; they differ only in how they punish a non-equilibrium
        opponent."""

    def _mix(self, card: int, history: tuple[str, ...]) -> dict[str, float]:
        from ..games.kuhn_poker import BET, CALL, CHECK, FOLD, JACK, KING, QUEEN

        a = self.alpha
        if history == ():  # first to act
            bet = {JACK: a, QUEEN: 0.0, KING: 3 * a}[card]
            return {BET: bet, CHECK: 1.0 - bet}
        if history == (CHECK,):  # second to act, nobody has bet
            bet = {JACK: 1 / 3, QUEEN: 0.0, KING: 1.0}[card]
            return {BET: bet, CHECK: 1.0 - bet}
        if history == (BET,):  # second to act, facing a bet
            call = {JACK: 0.0, QUEEN: 1 / 3, KING: 1.0}[card]
            return {CALL: call, FOLD: 1.0 - call}
        if history == (CHECK, BET):  # first player, check-raised
            call = {JACK: 0.0, QUEEN: a + 1 / 3, KING: 1.0}[card]
            return {CALL: call, FOLD: 1.0 - call}
        raise ValueError(f"no equilibrium entry for history {history}")

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        if getattr(game, "name", None) != "kuhn":
            raise ValueError(f"{type(self).__name__} only plays Kuhn poker, not {game.name}")
        card, history = game.key(state)
        mix = self._mix(card, history)
        roll = rng.random()
        cumulative = 0.0
        for move in legal_moves:
            cumulative += mix.get(move, 0.0)
            if roll < cumulative:
                return move
        return legal_moves[-1]  # only reachable through floating-point slack

    def action_probabilities(self, game: Game, state: State, legal_moves: Sequence[Move]):
        card, history = game.key(state)
        mix = self._mix(card, history)
        return {game.move_label(move): mix.get(move, 0.0) for move in legal_moves}

    def stats(self) -> dict[str, float]:
        return {"alpha": self.alpha}
