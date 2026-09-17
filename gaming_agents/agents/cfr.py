"""Counterfactual regret minimisation: the algorithm poker actually needs.

Every learner up to this point improves by playing: take an action, see what it
returned, prefer it more next time. That works when there is a best move to
find. In poker there often is not -- any strategy a good opponent can read is a
strategy they can beat, so the target is a *mixture*, and a method that keeps
sharpening toward one action can only chase its own tail.

CFR converges to that mixture. At each spot it tracks how much it regrets not
having played each action, accumulated over the whole game tree and weighted by
how often the rest of the table would let that spot happen at all. It then
plays each action in proportion to its positive regret. The running average of
those strategies -- not the latest one -- provably converges to a Nash
equilibrium in two-player zero-sum games.

It learns by walking the tree, not by playing hands, so it needs a game that
can enumerate its openings. That restricts it to small games, which is exactly
where this project uses it.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Hashable, Sequence

from ..core.game import Game, Move, State
from .base import Agent

#: A fixed source for the deterministic ``step`` calls a tree walk makes.
_STILL = random.Random(0)


class CFRAgent(Agent):
    name = "cfr"

    def __init__(self, name: str | None = None, *, iterations_per_episode: int = 1) -> None:
        super().__init__(name)
        #: information set -> move label -> regret accumulated for not playing it
        self.regrets: dict[Hashable, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        #: information set -> move label -> weight of the average strategy
        self.strategy_sum: dict[Hashable, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.iterations = 0
        self.iterations_per_episode = iterations_per_episode
        """CFR has no use for a played hand, but the trainer counts in
        episodes, so an episode is used purely as a clock: each one triggers
        this many tree passes. It keeps CFR usable from the same trainer,
        curriculum and CLI as everything else."""

    # -- the strategy -----------------------------------------------------

    def _current_strategy(self, key: Hashable, labels: Sequence[str]) -> dict[str, float]:
        """Regret matching: play each action in proportion to its positive regret."""
        regrets = self.regrets[key]
        positive = {label: max(regrets.get(label, 0.0), 0.0) for label in labels}
        total = sum(positive.values())
        if total <= 0:
            # Nothing regretted yet, so nothing to prefer.
            return {label: 1.0 / len(labels) for label in labels}
        return {label: value / total for label, value in positive.items()}

    def average_strategy(self, key: Hashable, labels: Sequence[str]) -> dict[str, float]:
        """The running average, which is the thing that converges -- not the
        current strategy, which keeps moving forever."""
        weights = self.strategy_sum.get(key)
        total = sum(weights.get(label, 0.0) for label in labels) if weights else 0.0
        if not weights or total <= 0:
            return {label: 1.0 / len(labels) for label in labels}
        return {label: weights.get(label, 0.0) / total for label in labels}

    # -- solving ----------------------------------------------------------

    def solve(self, game: Game, iterations: int) -> None:
        """Run ``iterations`` full passes over the game tree."""
        openings = game.initial_outcomes()
        for _ in range(iterations):
            for opening, chance in openings:
                self._walk(game, opening, [1.0] * game.num_players, chance)
            self.iterations += 1

    def _walk(self, game: Game, state: State, reach: list[float], chance: float) -> list[float]:
        """Return each player's expected value, and bank regrets on the way out."""
        if game.is_terminal(state):
            return list(game.outcome(state))

        player = game.current_player(state)
        moves = game.legal_moves(state)
        labels = [game.move_label(move) for move in moves]
        key = game.key(state)
        strategy = self._current_strategy(key, labels)

        here = [0.0] * game.num_players
        values: dict[str, list[float]] = {}
        for move, label in zip(moves, labels):
            share = strategy[label]
            deeper = list(reach)
            deeper[player] *= share
            child, _ = game.step(state, move, _STILL)
            values[label] = self._walk(game, child, deeper, chance)
            for seat in range(game.num_players):
                here[seat] += share * values[label][seat]

        # Weight regret by how likely everyone *else* was to allow this spot.
        # That is what makes the update counterfactual: a spot the opponent
        # rarely permits should not dominate, however wrong the play there was.
        counterfactual = chance
        for seat, probability in enumerate(reach):
            if seat != player:
                counterfactual *= probability

        regrets = self.regrets[key]
        weights = self.strategy_sum[key]
        for label in labels:
            regrets[label] += counterfactual * (values[label][player] - here[player])
            # The average is weighted by this player's own reach, so spots it
            # seldom reaches count for little in the strategy it settles on.
            weights[label] += reach[player] * strategy[label]
        return here

    # -- playing ----------------------------------------------------------

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        labels = [game.move_label(move) for move in legal_moves]
        mix = self.average_strategy(game.key(state), labels)
        roll = rng.random()
        cumulative = 0.0
        for move, label in zip(legal_moves, labels):
            cumulative += mix[label]
            if roll < cumulative:
                return move
        return legal_moves[-1]

    def action_probabilities(self, game: Game, state: State, legal_moves: Sequence[Move]):
        labels = [game.move_label(move) for move in legal_moves]
        return self.average_strategy(game.key(state), labels)

    # -- trainer integration ----------------------------------------------

    def end_episode(self, game: Game, final_state: State, outcome: tuple[float, ...], seat: int) -> None:
        if not self.close_seat(seat) or not self.training:
            return
        self.solve(game, self.iterations_per_episode)

    # -- persistence ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "cfr",
            "name": self.name,
            "iterations": self.iterations,
            "regrets": {repr(key): dict(row) for key, row in self.regrets.items()},
            "strategy_sum": {repr(key): dict(row) for key, row in self.strategy_sum.items()},
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        raise NotImplementedError(
            "a CFR brain is keyed by information set, which does not survive a round trip "
            "through repr(); re-solve instead -- it takes seconds on the games this handles"
        )

    def stats(self) -> dict[str, Any]:
        return {"iterations": self.iterations, "info_sets": len(self.strategy_sum)}
