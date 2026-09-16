"""Tabular Q-learning: the agent that actually learns by playing.

It keeps one table per game, so a single agent object can be walked up the
whole curriculum and carry everything it learned on the earlier rungs. The
tables are keyed by strings (``repr`` of the game's state key, and the game's
own move label) so a brain saves and reloads as plain JSON.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence

from ..core.game import Game, Move, State
from .base import Agent, Transition


class TabularQAgent(Agent):
    name = "qlearner"

    def __init__(
        self,
        name: str | None = None,
        *,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
        optimistic: float = 0.0,
    ) -> None:
        super().__init__(name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimistic = optimistic
        """Initial value for unseen state-action pairs. A positive value makes
        the agent curious about moves it has never tried."""

        #: game name -> state key -> move label -> value
        self.q: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        self.episodes = 0
        self.updates = 0

    # -- acting -----------------------------------------------------------

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        if self.training and rng.random() < self.epsilon:
            return rng.choice(list(legal_moves))

        row = self.q[game.name].get(repr(game.key(state)), {})
        best_value = None
        best: list[Move] = []
        for move in legal_moves:
            value = row.get(game.move_label(move), self.optimistic)
            if best_value is None or value > best_value:
                best_value, best = value, [move]
            elif value == best_value:
                best.append(move)
        # Ties are broken at random rather than by move order, otherwise the
        # agent locks onto whichever move the game happens to list first and
        # never discovers the others.
        return rng.choice(best)

    # -- learning ---------------------------------------------------------

    def learn(self, game: Game, transition: Transition) -> None:
        if not self.training:
            return
        table = self.q[game.name]
        row = table[repr(transition.key)]
        move_label = game.move_label(transition.move)
        current = row.get(move_label, self.optimistic)

        if transition.done or transition.next_state is None:
            target = transition.reward
        else:
            next_row = table.get(repr(transition.next_key), {})
            future = max(
                (next_row.get(game.move_label(m), self.optimistic) for m in transition.next_legal),
                default=0.0,
            )
            target = transition.reward + self.gamma * future

        row[move_label] = current + self.alpha * (target - current)
        self.updates += 1

    def end_episode(self, game: Game, final_state: State, outcome: tuple[float, ...], seat: int) -> None:
        if not self.close_seat(seat):
            return  # still holding the other seat of a self-play game
        if not self.training:
            return  # evaluation games must not age the exploration schedule
        self.episodes += 1
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -- persistence ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "tabular_q",
            "name": self.name,
            "hyper": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "optimistic": self.optimistic,
            },
            "episodes": self.episodes,
            "updates": self.updates,
            "q": {game: {key: dict(row) for key, row in table.items()} for game, table in self.q.items()},
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        hyper = data.get("hyper", {})
        for field, value in hyper.items():
            if hasattr(self, field):
                setattr(self, field, value)
        self.name = data.get("name", self.name)
        self.episodes = data.get("episodes", 0)
        self.updates = data.get("updates", 0)
        self.q = defaultdict(lambda: defaultdict(dict))
        for game, table in data.get("q", {}).items():
            for key, row in table.items():
                self.q[game][key] = dict(row)

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "updates": self.updates,
            "epsilon": round(self.epsilon, 4),
            "states_known": sum(len(table) for table in self.q.values()),
        }

    def known_states(self, game_name: str) -> int:
        return len(self.q.get(game_name, {}))
