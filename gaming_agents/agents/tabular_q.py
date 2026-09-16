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
        alpha_mode: str = "constant",
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
        optimistic: float = 0.0,
    ) -> None:
        super().__init__(name)
        if alpha_mode not in ("constant", "visits"):
            raise ValueError(f"alpha_mode must be 'constant' or 'visits', not {alpha_mode!r}")
        self.alpha = alpha
        self.alpha_mode = alpha_mode
        """How the step size is chosen, and it matters more than it looks.

        ``"constant"`` keeps a fixed step, which never settles but keeps
        adapting -- right when the thing being learned about is itself moving,
        as an opponent does while it improves alongside you.

        ``"visits"`` uses 1/n for the nth visit to a state-action pair, the
        textbook stochastic-approximation schedule. It converges to the true
        value, but only if there *is* a fixed true value -- so use it against
        a fixed environment, like the dealer in Blackjack. Against a moving
        opponent it would go deaf to the change.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimistic = optimistic
        """Initial value for unseen state-action pairs. A positive value makes
        the agent curious about moves it has never tried."""

        #: game name -> state key -> move label -> value
        self.q: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        #: Same shape, counting updates. Only filled in "visits" mode -- on a
        #: game with hundreds of thousands of states it would otherwise double
        #: the memory for nothing.
        self.visits: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
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

        row[move_label] = current + self._step_size(game.name, transition.key, move_label) * (target - current)
        self.updates += 1

    def _step_size(self, game_name: str, key, move_label: str) -> float:
        if self.alpha_mode == "constant":
            return self.alpha
        seen = self.visits[game_name][repr(key)]
        count = seen.get(move_label, 0) + 1
        seen[move_label] = count
        return 1.0 / count

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
                "alpha_mode": self.alpha_mode,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "optimistic": self.optimistic,
            },
            "episodes": self.episodes,
            "updates": self.updates,
            "q": {game: {key: dict(row) for key, row in table.items()} for game, table in self.q.items()},
            "visits": {game: {key: dict(row) for key, row in table.items()} for game, table in self.visits.items()},
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
        # Visit counts have to come back too, or a reloaded agent would restart
        # its 1/n schedule at 1 and throw away everything it had settled on.
        self.visits = defaultdict(lambda: defaultdict(dict))
        for game, table in data.get("visits", {}).items():
            for key, row in table.items():
                self.visits[game][key] = dict(row)

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "updates": self.updates,
            "epsilon": round(self.epsilon, 4),
            "alpha_mode": self.alpha_mode,
            "states_known": sum(len(table) for table in self.q.values()),
        }

    def known_states(self, game_name: str) -> int:
        return len(self.q.get(game_name, {}))
