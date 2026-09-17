"""The agent interface: something that picks moves, and may learn from them."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..core.game import Game, Move, State
from ..core.transition import Transition


class Agent(ABC):
    """Something that picks moves, and may learn from how they turned out.

    The lifecycle per episode is ``start_episode`` → many ``select_move`` /
    ``learn`` pairs → ``end_episode``. Agents that do not learn simply inherit
    the no-op hooks.
    """

    #: Identifier used in reports, saved brains and the CLI.
    name: str = "agent"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name
        #: When False the agent should exploit rather than explore. The
        #: trainer flips this off for evaluation matches.
        self.training = True
        #: Seats this agent currently occupies. In self-play it holds both, so
        #: this is what stops one game being counted twice.
        self._open_seats: set[int] = set()

    @abstractmethod
    def select_move(
        self,
        game: Game,
        state: State,
        legal_moves: Sequence[Move],
        rng: random.Random,
    ) -> Move:
        """Choose one of ``legal_moves``."""

    def action_probabilities(
        self,
        game: Game,
        state: State,
        legal_moves: Sequence[Move],
    ) -> dict[str, float] | None:
        """This agent's move distribution here, if it can state one exactly.

        Most agents cannot: they return a move when asked and their strategy
        has to be estimated by asking repeatedly. An agent that holds an
        explicit mixed strategy should report it here instead, because
        sampling would put a noise floor under any measurement of it -- and
        for a poker agent the measurement that matters, exploitability, is
        supposed to reach zero.
        """
        return None

    # -- learning hooks (no-ops by default) -------------------------------

    def start_episode(self, game: Game, seat: int) -> None:
        """Called before the first move of an episode, once per seat held."""
        self._open_seats.add(seat)

    def learn(self, game: Game, transition: Transition) -> None:
        """Called once per closed transition for this agent's seat."""

    def end_episode(self, game: Game, final_state: State, outcome: tuple[float, ...], seat: int) -> None:
        """Called after the episode ends, once per seat held."""
        self.close_seat(seat)

    def close_seat(self, seat: int) -> bool:
        """Release ``seat`` and report whether the whole episode is now over.

        Subclasses that count episodes call this instead of testing the seat
        number: an agent playing itself holds both seats and would otherwise
        count one game twice, while an agent alternating seats against a
        sparring partner would only ever count half of them.
        """
        self._open_seats.discard(seat)
        return not self._open_seats

    # -- persistence ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of everything learned. Empty if stateless."""
        return {}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        """Restore from a :meth:`state_dict` snapshot."""

    def stats(self) -> dict[str, Any]:
        """Free-form numbers to show in training reports."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name!r}>"
