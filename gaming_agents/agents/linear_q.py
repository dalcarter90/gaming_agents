"""A learner that describes positions instead of memorising them.

The tabular agent keeps one number per exact position. On Connect Four that
means 95% of what it writes down it never reads again: the position never
recurs, so the experience is spent the moment it is recorded.

This agent keeps one number per *feature* instead. A position is described --
threats I have, threats they have, who holds the centre -- and the value of the
position is the weighted sum of that description. Every game it plays therefore
adjusts its opinion of every position sharing any of those features, including
the ones it will never see.

With the default feature map, which is a single indicator per exact position,
this is arithmetically identical to the tabular agent. That is deliberate: it
makes "what does describing a position actually buy" a measurable question
rather than an argument.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Sequence

from ..core.game import Game, Move, State
from .base import Agent, Transition


class LinearAgent(Agent):
    name = "linear"

    def __init__(
        self,
        name: str | None = None,
        *,
        alpha: float = 0.02,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
        shared: bool = False,
    ) -> None:
        super().__init__(name)
        self.shared = shared
        """Whether one set of opinions covers every game.

        Off by default, which keeps each game's weights separate and matches
        how the tabular agent behaves. Turn it on and anything the agent learns
        about a concept in the shared vocabulary -- being one move from losing,
        holding the centre -- applies the moment it meets that concept in
        another game. Descriptions that name no shared concept are prefixed
        with their game, so they stay local either way.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        #: game name -> feature -> weight. Unseen features start at zero, which
        #: is the honest prior: no opinion until something happens.
        self.weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.episodes = 0
        self.updates = 0

    #: The bucket a game's weights live in. With sharing on every game uses the
    #: same one, so concepts meet across games instead of being relearned.
    SHARED = "*"

    def _row(self, game: Game) -> dict[str, float]:
        return self.weights[self.SHARED if self.shared else game.name]

    # -- the value function -----------------------------------------------

    def value(self, game: Game, state: State) -> float:
        """What this position is worth to the player about to move in it."""
        if game.is_terminal(state):
            return 0.0
        row = self._row(game)
        return sum(row.get(feature, 0.0) * amount for feature, amount in game.features(state).items())

    def _score(self, game: Game, state: State, move: Move, seat: int, rng: random.Random) -> float:
        """What playing ``move`` is worth to ``seat``.

        This looks one move ahead using the rules, which is a real difference
        from the tabular agent -- see ``linear-onehot`` in the results for how
        much of the improvement is the lookahead and how much is the features.
        """
        nxt, rewards = game.step(state, move, rng)
        score = rewards[seat]
        if game.is_terminal(nxt):
            return score
        ahead = self.value(game, nxt)
        # ``value`` is always from the mover's point of view, so in a two-player
        # game the opponent's gain is this seat's loss.
        return score + self.gamma * (ahead if game.current_player(nxt) == seat else -ahead)

    # -- acting -----------------------------------------------------------

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        if self.training and rng.random() < self.epsilon:
            return rng.choice(list(legal_moves))

        seat = game.current_player(state)
        best_score = None
        best: list[Move] = []
        for move in legal_moves:
            score = self._score(game, state, move, seat, rng)
            if best_score is None or score > best_score:
                best_score, best = score, [move]
            elif score == best_score:
                best.append(move)
        return rng.choice(best)

    # -- learning ---------------------------------------------------------

    def learn(self, game: Game, transition: Transition) -> None:
        """One step of TD(0) on the value function.

        The update is spread across every feature of the position in proportion
        to how strongly that feature was present, which is exactly what lets a
        single game move the agent's opinion of positions it has never seen.
        """
        if not self.training:
            return
        target = transition.reward
        if not transition.done and transition.next_state is not None:
            target += self.gamma * self.value(game, transition.next_state)

        error = target - self.value(game, transition.state)
        row = self._row(game)
        for feature, amount in game.features(transition.state).items():
            row[feature] += self.alpha * error * amount
        self.updates += 1

    def end_episode(self, game: Game, final_state: State, outcome: tuple[float, ...], seat: int) -> None:
        if not self.close_seat(seat) or not self.training:
            return
        self.episodes += 1
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -- persistence ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "name": self.name,
            "hyper": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "shared": self.shared,
            },
            "episodes": self.episodes,
            "updates": self.updates,
            "weights": {game: dict(row) for game, row in self.weights.items()},
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        for field, value in data.get("hyper", {}).items():
            if hasattr(self, field):
                setattr(self, field, value)
        self.name = data.get("name", self.name)
        self.episodes = data.get("episodes", 0)
        self.updates = data.get("updates", 0)
        self.weights = defaultdict(lambda: defaultdict(float))
        for game, row in data.get("weights", {}).items():
            self.weights[game] = defaultdict(float, row)

    def stats(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "updates": self.updates,
            "epsilon": round(self.epsilon, 4),
            "weights_learned": sum(len(row) for row in self.weights.values()),
        }

    def explain(self, game_name: str, top: int = 12) -> list[tuple[str, float]]:
        """The features it leans on hardest, for reading what it actually learned."""
        row = self.weights.get(game_name, {})
        return sorted(row.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]
