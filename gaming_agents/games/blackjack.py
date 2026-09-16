"""Blackjack: the first game where the world itself is random.

Solitaire against a fixed dealer policy, dealt from an infinite deck so the
state stays the classic ``(player total, dealer upcard, usable ace)`` triple.
Nothing here can be won reliably -- the point of this rung is that the agent
has to separate a bad outcome from a bad decision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Hashable

from ..core.game import Game, Rewards

HIT = "hit"
STICK = "stick"

#: Face cards all count ten, so the deck is 1-9 plus four ways to draw a ten.
_CARDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)

DEALER_STICKS_AT = 17


@dataclass(frozen=True, slots=True)
class BlackjackState:
    player_total: int
    player_ace: bool
    """True when the player holds an ace still counted as 11."""
    dealer_upcard: int
    dealer_total: int
    dealer_ace: bool
    done: bool = False
    result: float = 0.0
    """+1 win, 0 draw, -1 loss. Only meaningful once ``done``."""


class Blackjack(Game):
    name = "blackjack"
    title = "Blackjack"
    num_players = 1
    deterministic = False
    difficulty = 3

    def initial_state(self, rng: random.Random) -> BlackjackState:
        player_total, player_ace = 0, False
        for _ in range(2):
            player_total, player_ace = _add(player_total, player_ace, _draw(rng))
        dealer_up = _draw(rng)
        dealer_total, dealer_ace = _add(0, False, dealer_up)
        dealer_total, dealer_ace = _add(dealer_total, dealer_ace, _draw(rng))
        return BlackjackState(
            player_total=player_total,
            player_ace=player_ace,
            dealer_upcard=dealer_up,
            dealer_total=dealer_total,
            dealer_ace=dealer_ace,
        )

    def current_player(self, state: BlackjackState) -> int:
        return 0

    def legal_moves(self, state: BlackjackState) -> list[str]:
        return [] if state.done else [HIT, STICK]

    def step(self, state: BlackjackState, move: str, rng: random.Random) -> tuple[BlackjackState, Rewards]:
        if state.done:
            raise ValueError("the hand is already over")

        if move == HIT:
            total, ace = _add(state.player_total, state.player_ace, _draw(rng))
            if total > 21:
                return _finish(state, total, ace, -1.0), (-1.0,)
            return (
                BlackjackState(
                    player_total=total,
                    player_ace=ace,
                    dealer_upcard=state.dealer_upcard,
                    dealer_total=state.dealer_total,
                    dealer_ace=state.dealer_ace,
                ),
                (0.0,),
            )

        if move != STICK:
            raise ValueError(f"unknown move {move!r}")

        dealer_total, dealer_ace = state.dealer_total, state.dealer_ace
        while dealer_total < DEALER_STICKS_AT:
            dealer_total, dealer_ace = _add(dealer_total, dealer_ace, _draw(rng))
        if dealer_total > 21:
            result = 1.0
        elif dealer_total > state.player_total:
            result = -1.0
        elif dealer_total < state.player_total:
            result = 1.0
        else:
            result = 0.0
        final = BlackjackState(
            player_total=state.player_total,
            player_ace=state.player_ace,
            dealer_upcard=state.dealer_upcard,
            dealer_total=dealer_total,
            dealer_ace=dealer_ace,
            done=True,
            result=result,
        )
        return final, (result,)

    def is_terminal(self, state: BlackjackState) -> bool:
        return state.done

    def key(self, state: BlackjackState) -> Hashable:
        # The dealer's hole card is hidden, so it must not enter the key --
        # only the upcard, which is what the player can actually see.
        return (state.player_total, state.dealer_upcard, state.player_ace)

    def outcome(self, state: BlackjackState) -> Rewards:
        return (state.result,)

    def render(self, state: BlackjackState) -> str:
        ace = " (usable ace)" if state.player_ace else ""
        dealer = state.dealer_total if state.done else f"{state.dealer_upcard} + ?"
        return f"  you: {state.player_total}{ace}\n  dealer: {dealer}"

    def move_label(self, move: str) -> str:
        return move

    def parse_move(self, text: str, state: BlackjackState) -> str:
        choice = text.strip().lower()
        aliases = {"h": HIT, "hit": HIT, "s": STICK, "stick": STICK, "stand": STICK}
        if choice not in aliases:
            raise ValueError("type 'hit' or 'stick'")
        return aliases[choice]


def _draw(rng: random.Random) -> int:
    return rng.choice(_CARDS)


def _add(total: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    """Add ``card`` to a hand, demoting an ace from 11 to 1 if that saves it."""
    if card == 1 and total + 11 <= 21:
        return total + 11, True
    total += card
    if total > 21 and usable_ace:
        return total - 10, False
    return total, usable_ace


def _finish(state: BlackjackState, total: int, ace: bool, result: float) -> BlackjackState:
    return BlackjackState(
        player_total=total,
        player_ace=ace,
        dealer_upcard=state.dealer_upcard,
        dealer_total=state.dealer_total,
        dealer_ace=state.dealer_ace,
        done=True,
        result=result,
    )
