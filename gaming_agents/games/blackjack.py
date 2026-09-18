"""Blackjack: the first game where the world itself is random.

Solitaire against a fixed dealer policy, dealt from an infinite deck so the
state stays the classic ``(player total, dealer upcard, usable ace)`` triple.
Nothing here can be won reliably -- the point of this rung is that the agent
has to separate a bad outcome from a bad decision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
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
    #: The hole card is dealt but not shown. Nothing had searched this game
    #: before, so nothing had ever had the chance to peek at it -- but an agent
    #: that steps the game forward to evaluate "stick" is playing the dealer
    #: out from a hand it is not allowed to see.
    imperfect_information = True
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

    def redeal(self, state: BlackjackState, seat: int, rng: random.Random) -> BlackjackState:
        """Keep everything on show, deal the dealer a fresh hole card.

        The upcard stays, the player's hand stays, and the card underneath is
        drawn again -- so a searcher reasons about a table it could actually be
        sitting at rather than the one it secretly knows it is.
        """
        total, ace = _add(0, False, state.dealer_upcard)
        total, ace = _add(total, ace, _draw(rng))
        return BlackjackState(
            player_total=state.player_total,
            player_ace=state.player_ace,
            dealer_upcard=state.dealer_upcard,
            dealer_total=total,
            dealer_ace=ace,
            done=state.done,
            result=state.result,
        )

    def features(self, state: BlackjackState) -> dict[str, float]:
        """The universal vocabulary, read onto a hand of cards.

        Every entry is the definition in :attr:`Game.UNIVERSAL_FEATURES`
        applied as literally as this game allows, and no more:

        * *progress* -- how close the hand is to being over, which here is how
          close the total is to the 21 that ends it.
        * *my_strength* -- standing on 11 or less never wins and standing on 21
          always does, so strength ramps between them.
        * *their_strength* -- all that is visible of the dealer is the upcard,
          and a higher one is a stronger dealer. That is an approximation: it
          misses that a dealer showing 2 is less dangerous than one showing 6.
        * *win_available* -- holding 21, which cannot be beaten.
        * *must_block* -- standing on 11 or less loses for certain, so the hand
          is lost unless the player acts.
        """
        total = state.player_total
        upcard = 11 if state.dealer_upcard == 1 else state.dealer_upcard
        ramp = lambda value, low, high: min(1.0, max(0.0, (value - low) / (high - low)))  # noqa: E731

        return {
            "bias": 1.0,
            "progress": ramp(total, 4, 21),
            "my_strength": ramp(total, 11, 21),
            "their_strength": ramp(upcard, 2, 11),
            "win_available": 1.0 if total == 21 else 0.0,
            "must_block": 1.0 if total <= 11 else 0.0,
            # Particular to this game: an ace counted as eleven can be demoted,
            # so a hand holding one cannot bust on the next card.
            "blackjack:usable_ace": 1.0 if state.player_ace else 0.0,
        }

    def optimal_move(self, state: BlackjackState) -> str:
        """The exactly correct play in ``state``, solved rather than learned.

        Ties go to sticking, which matters only for the handful of positions
        where the two options are worth the same.
        """
        stick = stick_value(state.player_total, state.dealer_upcard)
        hit = hit_value(state.player_total, state.player_ace, state.dealer_upcard)
        return STICK if stick >= hit else HIT

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


# -- exact solution -------------------------------------------------------
#
# The deck is infinite and the dealer's rule is fixed, so this variant can be
# solved outright rather than approximated. That gives the curriculum a real
# ceiling to measure the learner against, the same way Nim's nim-sum does.

#: Each rank with its probability. Faces all count ten, so a ten is four times
#: as likely as any other draw.
_DECK: tuple[tuple[int, float], ...] = tuple(
    (card, (4 if card == 10 else 1) / 13) for card in range(1, 11)
)


@lru_cache(maxsize=None)
def _dealer_finals(total: int, usable_ace: bool) -> tuple[tuple[int, float], ...]:
    """Distribution of the dealer's final total, using 22 to mean 'bust'."""
    if total >= DEALER_STICKS_AT:
        return ((min(total, 22), 1.0),)
    outcomes: dict[int, float] = {}
    for card, p in _DECK:
        nxt, ace = _add(total, usable_ace, card)
        for final, q in _dealer_finals(nxt, ace):
            outcomes[final] = outcomes.get(final, 0.0) + p * q
    return tuple(sorted(outcomes.items()))


@lru_cache(maxsize=None)
def _dealer_finals_from_upcard(upcard: int) -> tuple[tuple[int, float], ...]:
    """Same, but from the player's side of the table: the hole card is unknown."""
    total, ace = _add(0, False, upcard)
    outcomes: dict[int, float] = {}
    for card, p in _DECK:
        nxt, nxt_ace = _add(total, ace, card)
        for final, q in _dealer_finals(nxt, nxt_ace):
            outcomes[final] = outcomes.get(final, 0.0) + p * q
    return tuple(sorted(outcomes.items()))


@lru_cache(maxsize=None)
def stick_value(player_total: int, dealer_upcard: int) -> float:
    """Expected payoff from sticking on ``player_total``."""
    value = 0.0
    for final, p in _dealer_finals_from_upcard(dealer_upcard):
        if final > 21 or player_total > final:
            value += p
        elif player_total < final:
            value -= p
    return value


@lru_cache(maxsize=None)
def hit_value(player_total: int, usable_ace: bool, dealer_upcard: int) -> float:
    """Expected payoff from hitting, assuming optimal play from then on."""
    value = 0.0
    for card, p in _DECK:
        total, ace = _add(player_total, usable_ace, card)
        value += p * (-1.0 if total > 21 else _best_value(total, ace, dealer_upcard))
    return value


@lru_cache(maxsize=None)
def _best_value(player_total: int, usable_ace: bool, dealer_upcard: int) -> float:
    return max(
        stick_value(player_total, dealer_upcard),
        hit_value(player_total, usable_ace, dealer_upcard),
    )


@lru_cache(maxsize=None)
def optimal_value() -> float:
    """Expected payoff per hand under perfect play. Around -0.047 -- the house
    edge that is left once the player has no doubling, splitting or naturals."""
    value = 0.0
    for first, p1 in _DECK:
        for second, p2 in _DECK:
            total, ace = _add(*_add(0, False, first), second)
            for upcard, pu in _DECK:
                value += p1 * p2 * pu * _best_value(total, ace, upcard)
    return value
