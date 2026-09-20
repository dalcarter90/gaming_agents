"""A learner that works out its own concepts.

Every learner before this was handed its vocabulary. The tabular agent got one
slot per position; the linear agent got weights over threats, centre control
and progress -- all of which a person sat down and wrote. This one reads the
position as bare numbers and has to find whatever structure it needs.

``read`` decides which, and is the whole experiment:

* ``"position"`` -- the raw board or hand from :meth:`Game.encode`, with no
  concepts supplied at all.
* ``"concepts"`` -- the same hand-written features the linear agent used, so
  that any difference is the bend in the function rather than the input.

Everything else is deliberately identical to :class:`LinearAgent`: the same
one-move lookahead, the same TD(0) update, the same honest sampling in games
that hide something. Only the shape of the value function changes.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

from ..core.game import Game, Move, State
from .base import Agent, Transition
from .network import ValueNetwork


class NeuralAgent(Agent):
    name = "neural"

    def __init__(
        self,
        name: str | None = None,
        *,
        hidden: int = 24,
        alpha: float = 0.01,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
        read: str = "position",
        plan: bool = True,
        lookahead_samples: int = 8,
        seed: int = 0,
    ) -> None:
        super().__init__(name)
        if read not in ("position", "concepts"):
            raise ValueError(f"read must be 'position' or 'concepts', not {read!r}")
        self.read = read
        self.plan = plan
        """Whether to value a move by playing it out one step.

        Planning is exact in a deterministic game and *noisy* in a game of
        chance: valuing "stick" at Blackjack means sampling the dealer's whole
        hand, and eight samples of that carry more error than the gap between
        hit and stick. Turning planning off scores every action directly
        instead, one network output each, which is what a lookup table does
        without anyone calling it planning.
        """
        self.hidden = hidden
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lookahead_samples = lookahead_samples
        self.seed = seed
        #: One network per game. Their inputs are different widths, so unlike
        #: the linear agent's weights these cannot yet be pooled across games.
        self.nets: dict[str, ValueNetwork] = {}
        self.episodes = 0
        self.updates = 0

    # -- reading a position -----------------------------------------------

    def _inputs(self, game: Game, state: State) -> tuple[float, ...]:
        if self.read == "position":
            return tuple(game.encode(state))
        described = game.features(state)
        return tuple(described[name] for name in sorted(described))

    def _net(self, game: Game, width: int) -> ValueNetwork:
        net = self.nets.get(game.name)
        if net is None:
            heads = 1 if self.plan else len(self._actions(game))
            net = self.nets[game.name] = ValueNetwork(
                width, self.hidden, heads, random.Random(self.seed + len(self.nets))
            )
        return net

    def _actions(self, game: Game) -> tuple[Move, ...]:
        space = game.action_space()
        if not space:
            raise ValueError(
                f"{game.name} cannot enumerate its moves, so it can only be played "
                "by looking ahead -- construct this agent with plan=True"
            )
        return space

    def scores(self, game: Game, state: State, legal_moves: Sequence[Move]) -> dict[Move, float]:
        """What each legal move is worth, scored straight from the position."""
        actions = self._actions(game)
        x = self._inputs(game, state)
        outs, _ = self._net(game, len(x)).forward(x)
        return {move: outs[actions.index(move)] for move in legal_moves}

    def value(self, game: Game, state: State) -> float:
        if game.is_terminal(state):
            return 0.0
        x = self._inputs(game, state)
        net = self._net(game, len(x))
        if self.plan:
            return net.value(x)
        legal = game.legal_moves(state)
        outs, _ = net.forward(x)
        actions = self._actions(game)
        return max(outs[actions.index(move)] for move in legal) if legal else 0.0

    # -- acting -----------------------------------------------------------

    def _score(self, game: Game, state: State, move: Move, seat: int, rng: random.Random) -> float:
        looks = 1 if game.deterministic else max(1, self.lookahead_samples)
        total = 0.0
        for _ in range(looks):
            start = game.redeal(state, seat, rng) if game.imperfect_information else state
            nxt, rewards = game.step(start, move, rng)
            score = rewards[seat]
            if not game.is_terminal(nxt):
                ahead = self.value(game, nxt)
                score += self.gamma * (ahead if game.current_player(nxt) == seat else -ahead)
            total += score
        return total / looks

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        if self.training and rng.random() < self.epsilon:
            return rng.choice(list(legal_moves))
        seat = game.current_player(state)
        direct = None if self.plan else self.scores(game, state, legal_moves)
        best_score, best = None, []
        for move in legal_moves:
            score = self._score(game, state, move, seat, rng) if direct is None else direct[move]
            if best_score is None or score > best_score:
                best_score, best = score, [move]
            elif score == best_score:
                best.append(move)
        return rng.choice(best)

    # -- learning ---------------------------------------------------------

    def learn(self, game: Game, transition: Transition) -> None:
        if not self.training:
            return
        target = transition.reward
        if not transition.done and transition.next_state is not None:
            if self.plan:
                target += self.gamma * self.value(game, transition.next_state)
            else:
                ahead = self.scores(game, transition.next_state, transition.next_legal)
                if ahead:
                    target += self.gamma * max(ahead.values())

        x = self._inputs(game, transition.state)
        net = self._net(game, len(x))
        predicted, hidden = net.forward(x)
        if self.plan:
            net.learn(x, hidden, target - predicted[0], self.alpha)
        else:
            # Only the action actually taken has a target; the other heads are
            # left alone rather than told something nobody observed.
            slot = self._actions(game).index(transition.move)
            net.learn(x, hidden, {slot: target - predicted[slot]}, self.alpha)
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
            "kind": "neural",
            "name": self.name,
            "hyper": {"hidden": self.hidden, "alpha": self.alpha, "gamma": self.gamma,
                      "epsilon": self.epsilon, "epsilon_min": self.epsilon_min,
                      "epsilon_decay": self.epsilon_decay, "read": self.read,
                      "plan": self.plan, "lookahead_samples": self.lookahead_samples},
            "episodes": self.episodes,
            "updates": self.updates,
            "nets": {game: net.state_dict() for game, net in self.nets.items()},
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        for field, value in data.get("hyper", {}).items():
            if hasattr(self, field):
                setattr(self, field, value)
        self.name = data.get("name", self.name)
        self.episodes = data.get("episodes", 0)
        self.updates = data.get("updates", 0)
        self.nets = {game: ValueNetwork.from_state_dict(blob)
                     for game, blob in data.get("nets", {}).items()}

    def stats(self) -> dict[str, Any]:
        return {"episodes": self.episodes, "updates": self.updates, "reads": self.read,
                "plans": self.plan,
                "hidden": self.hidden,
                "weights": sum(n.inputs * n.hidden + 2 * n.hidden + 1 for n in self.nets.values())}
