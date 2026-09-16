"""The unit of experience a learning agent is fed.

It lives in ``core`` rather than next to the agents so that the episode runner
never has to import an agent, which keeps the dependency arrow pointing one
way: agents depend on the game layer, and the game layer depends on nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .game import Move, State


@dataclass(frozen=True)
class Transition:
    """One decision by one seat, closed once that seat is due to act again.

    In a two-player game the opponent moves in between, so ``next_state`` is
    the position the seat faces on *its* next turn rather than the position
    immediately after its own move. ``reward`` sums everything that seat
    collected in the meantime, which is what makes ordinary TD updates work
    in a turn-based adversarial game.
    """

    seat: int
    state: State
    key: Hashable
    move: Move
    reward: float
    next_state: State | None
    next_key: Hashable | None
    next_legal: tuple[Move, ...]
    done: bool
