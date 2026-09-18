"""Kuhn poker: three cards, one betting round, and hidden information.

The smallest game that is genuinely poker. Each player antes one chip and is
dealt one card from a three-card deck; there is a single round of betting; the
higher card wins at showdown. That is the whole game, and it still contains
bluffing, and its solution is known in closed form -- which is exactly why it
belongs here rather than a real poker variant.

It is also the first game in this project where a state holds something the
player to move must not see. Everything that makes the earlier rungs work --
searching from the current state, keying a table on it -- has to be re-examined
against that, which is the point of the rung.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import permutations
from typing import Hashable

from ..core.game import Game, Rewards

#: The deck, weakest first. Only the order matters.
JACK, QUEEN, KING = 0, 1, 2
CARD_NAMES = ("J", "Q", "K")

CHECK = "check"
BET = "bet"
CALL = "call"
FOLD = "fold"

#: Everyone puts this in before the cards come out.
ANTE = 1
#: The only bet size in the game.
BET_SIZE = 1


@dataclass(frozen=True, slots=True)
class KuhnState:
    #: ``cards[seat]`` is that seat's hole card. Seats never see each other's.
    cards: tuple[int, int]
    #: Actions so far, in order, as a tuple of move names.
    history: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{'/'.join(CARD_NAMES[c] for c in self.cards)} {'-'.join(self.history) or '(no action)'}"


#: Betting sequences that end the hand, and how they end it. ``showdown`` means
#: compare cards; otherwise the named seat won because the other folded.
_TERMINAL: dict[tuple[str, ...], str] = {
    (CHECK, CHECK): "showdown",
    (BET, FOLD): "seat0",
    (BET, CALL): "showdown",
    (CHECK, BET, FOLD): "seat1",
    (CHECK, BET, CALL): "showdown",
}


class KuhnPoker(Game):
    name = "kuhn"
    title = "Kuhn Poker"
    num_players = 2
    deterministic = False
    imperfect_information = True
    difficulty = 6

    def initial_state(self, rng: random.Random) -> KuhnState:
        deck = [JACK, QUEEN, KING]
        rng.shuffle(deck)
        return KuhnState(cards=(deck[0], deck[1]))

    def current_player(self, state: KuhnState) -> int:
        return len(state.history) % 2

    def legal_moves(self, state: KuhnState) -> list[str]:
        if self.is_terminal(state):
            return []
        # Facing a bet you may fold or call; otherwise you may check or bet.
        if state.history and state.history[-1] == BET:
            return [FOLD, CALL]
        return [CHECK, BET]

    def step(self, state: KuhnState, move: str, rng: random.Random) -> tuple[KuhnState, Rewards]:
        if move not in self.legal_moves(state):
            raise ValueError(f"{move!r} is not available after {state.history}")
        nxt = KuhnState(cards=state.cards, history=state.history + (move,))
        if not self.is_terminal(nxt):
            return nxt, (0.0, 0.0)
        return nxt, self.outcome(nxt)

    def is_terminal(self, state: KuhnState) -> bool:
        return state.history in _TERMINAL

    def key(self, state: KuhnState) -> Hashable:
        """The acting player's own card and the betting so far -- nothing else.

        This is the whole game's hidden-information boundary. Putting both
        cards in here would hand every table-based agent a look at the
        opponent's hand and make every number measured afterwards a fiction.
        """
        return (state.cards[self.current_player(state)], state.history)

    def outcome(self, state: KuhnState) -> Rewards:
        ending = _TERMINAL.get(state.history)
        if ending is None:
            return (0.0, 0.0)
        if ending == "seat0":
            return (float(ANTE), -float(ANTE))
        if ending == "seat1":
            return (-float(ANTE), float(ANTE))
        # Showdown: the pot holds the antes plus any called bet.
        stake = float(ANTE + (BET_SIZE if CALL in state.history else 0))
        winner = 0 if state.cards[0] > state.cards[1] else 1
        return (stake, -stake) if winner == 0 else (-stake, stake)

    def features(self, state: KuhnState) -> dict[str, float]:
        """The universal vocabulary, read onto a poker hand.

        As with Blackjack, each entry is the shared definition applied as
        literally as the game allows:

        * *progress* -- how far through the betting the hand has got.
        * *my_strength* -- the rank of the card held, which is the whole of
          this game's hand strength.
        * *their_strength* -- the opponent's card is hidden, but holding a King
          means they hold a Jack or a Queen. The average of whatever is left in
          the deck is honest information and gives no look at their hand.
        * *win_available* -- holding the King, which cannot lose a showdown.
        * *must_block* -- facing a bet, where the pot goes to them unless the
          player answers it.
        """
        seat = self.current_player(state)
        mine = state.cards[seat]
        others = [card for card in (JACK, QUEEN, KING) if card != mine]

        return {
            "bias": 1.0,
            "progress": len(state.history) / 3.0,
            "my_strength": mine / 2.0,
            "their_strength": sum(others) / 2.0 / len(others),
            "win_available": 1.0 if mine == KING else 0.0,
            "must_block": 1.0 if state.history and state.history[-1] == BET else 0.0,
            # Particular to this game: how much is already in the middle.
            "kuhn:pot": self.pot(state) / 4.0,
        }

    def redeal(self, state: KuhnState, seat: int, rng: random.Random) -> KuhnState:
        """Keep ``seat``'s card, deal the opponent a new one from the rest."""
        mine = state.cards[seat]
        theirs = rng.choice([c for c in (JACK, QUEEN, KING) if c != mine])
        cards = (mine, theirs) if seat == 0 else (theirs, mine)
        return KuhnState(cards=cards, history=state.history)

    def initial_outcomes(self) -> list[tuple[KuhnState, float]]:
        """All six deals, each equally likely."""
        deals = list(permutations((JACK, QUEEN, KING), 2))
        share = 1.0 / len(deals)
        return [(KuhnState(cards=deal), share) for deal in deals]

    def render(self, state: KuhnState) -> str:
        seat = self.current_player(state)
        lines = [f"  your card: {CARD_NAMES[state.cards[seat]]}"]
        lines.append(f"  betting:   {' -> '.join(state.history) if state.history else '(nothing yet)'}")
        lines.append(f"  pot:       {self.pot(state)}")
        if self.is_terminal(state):
            lines.append(f"  showdown:  {'/'.join(CARD_NAMES[c] for c in state.cards)}")
        else:
            lines.append(f"  to act:    player {seat}")
        return "\n".join(lines)

    def pot(self, state: KuhnState) -> int:
        chips = 2 * ANTE
        chips += BET_SIZE * sum(1 for move in state.history if move in (BET, CALL))
        return chips

    def move_label(self, move: str) -> str:
        return move

    def parse_move(self, text: str, state: KuhnState) -> str:
        aliases = {"c": CHECK, "k": CHECK, "b": BET, "r": BET, "f": FOLD, "call": CALL}
        choice = text.strip().lower()
        choice = aliases.get(choice, choice)
        legal = self.legal_moves(state)
        # "c" is check when there is nothing to call and call when there is.
        if choice == CHECK and CHECK not in legal and CALL in legal:
            choice = CALL
        if choice not in legal:
            raise ValueError(f"{text!r} is not available; try {' or '.join(legal)}")
        return choice
