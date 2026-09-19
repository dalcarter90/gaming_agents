"""Three tasks that are not games, for a broader childhood.

The agents have only ever practised games. These are episodic decision tasks
shaped like other kinds of reasoning: hitting a numeric target, narrowing a
hypothesis under a budget, and learning a hidden syntax by being cut off
whenever you break it.

They are emphatically not mathematics, science or language. They are the
smallest tasks that demand those *shapes* of reasoning while still fitting an
interface built for games, which is what makes the question askable: does
practice at something that is not a game transfer to games at all?

Each describes itself in the universal vocabulary, so anything it teaches is
stored in the same weights a game would use.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable

from ..core.game import Game, Rewards


# -- arithmetic ------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReckoningState:
    value: int
    target: int
    steps: int


class Reckoning(Game):
    """Reach a target number from zero, in a budget of moves.

    The operations are deliberately awkward -- doubling overshoots, and the
    only subtraction is larger than the only small addition -- so arriving
    exactly takes planning rather than counting.
    """

    name = "reckoning"
    title = "Reckoning (hit the number)"
    num_players = 1
    deterministic = False
    difficulty = 2

    #: (label, how it changes the running value)
    MOVES = (("+1", 1), ("+4", 4), ("x2", None), ("-3", -3))
    BUDGET = 8
    FLOOR, CEILING = -20, 200

    def __init__(self, low: int = 10, high: int = 99) -> None:
        self.low, self.high = low, high

    def initial_state(self, rng: random.Random) -> ReckoningState:
        return ReckoningState(value=0, target=rng.randint(self.low, self.high), steps=0)

    def current_player(self, state: ReckoningState) -> int:
        return 0

    def _apply(self, value: int, label: str) -> int:
        for name, delta in self.MOVES:
            if name == label:
                return value * 2 if delta is None else value + delta
        raise ValueError(f"unknown operation {label!r}")

    def legal_moves(self, state: ReckoningState) -> list[str]:
        if self.is_terminal(state):
            return []
        return [name for name, _ in self.MOVES
                if self.FLOOR <= self._apply(state.value, name) <= self.CEILING]

    def step(self, state: ReckoningState, move: str, rng: random.Random) -> tuple[ReckoningState, Rewards]:
        if move not in self.legal_moves(state):
            raise ValueError(f"{move!r} is not available from {state.value}")
        nxt = ReckoningState(value=self._apply(state.value, move),
                             target=state.target, steps=state.steps + 1)
        if nxt.value == nxt.target:
            return nxt, (1.0,)
        if nxt.steps >= self.BUDGET:
            return nxt, (-1.0,)
        return nxt, (0.0,)

    def is_terminal(self, state: ReckoningState) -> bool:
        return state.value == state.target or state.steps >= self.BUDGET

    def key(self, state: ReckoningState) -> Hashable:
        return (state.value, state.target, state.steps)

    def outcome(self, state: ReckoningState) -> Rewards:
        return (1.0,) if state.value == state.target else (-1.0,)

    def features(self, state: ReckoningState) -> dict[str, float]:
        gap = abs(state.target - state.value)
        span = max(1, self.high)
        moves = self.legal_moves(state)
        worse = sum(1 for m in moves if abs(state.target - self._apply(state.value, m)) > gap)
        left = self.BUDGET - state.steps
        return {
            "bias": 1.0,
            "progress": state.steps / self.BUDGET,
            # Being near the target is the whole of a good position here.
            "my_strength": max(0.0, 1.0 - gap / span),
            # What is against you is a move set that mostly drives you away.
            "their_strength": (worse / len(moves)) if moves else 1.0,
            "win_available": 1.0 if any(self._apply(state.value, m) == state.target for m in moves) else 0.0,
            "must_block": 1.0 if left <= 1 and not any(
                self._apply(state.value, m) == state.target for m in moves) else 0.0,
            "reckoning:reachable_parity": 1.0 if (state.target - state.value) % 2 == 0 else 0.0,
        }

    def render(self, state: ReckoningState) -> str:
        return (f"  at {state.value}, aiming for {state.target}\n"
                f"  moves used: {state.steps}/{self.BUDGET}")

    def move_label(self, move: str) -> str:
        return move

    def parse_move(self, text: str, state: ReckoningState) -> str:
        choice = text.strip().lower().replace("*", "x")
        if choice not in self.legal_moves(state):
            raise ValueError(f"try one of {self.legal_moves(state)}")
        return choice


# -- hypothesis narrowing --------------------------------------------------

@dataclass(frozen=True, slots=True)
class InquiryState:
    low: int
    high: int
    answer: int
    tests: int
    guessed: bool = False
    right: bool = False


class Inquiry(Game):
    """Find a hidden number by testing, with a budget of tests.

    Each test splits what is left; guessing early is a gamble and guessing late
    wastes the budget. The lesson available here is that an experiment is worth
    what it rules out, which is not a lesson any of the games teach.
    """

    name = "inquiry"
    title = "Inquiry (narrow it down)"
    num_players = 1
    deterministic = False
    difficulty = 2

    RANGE = 64
    BUDGET = 8
    MOVES = ("probe_low", "probe_middle", "probe_high", "guess")

    def initial_state(self, rng: random.Random) -> InquiryState:
        return InquiryState(low=1, high=self.RANGE, answer=rng.randint(1, self.RANGE), tests=0)

    def current_player(self, state: InquiryState) -> int:
        return 0

    def legal_moves(self, state: InquiryState) -> list[str]:
        if self.is_terminal(state):
            return []
        # With one candidate left there is nothing a further test could rule
        # out, so the only thing to do is say it. Leaving probes available here
        # let an agent test forever without ever answering.
        return ["guess"] if state.low == state.high else list(self.MOVES)

    def _threshold(self, state: InquiryState, move: str) -> int:
        span = state.high - state.low
        share = {"probe_low": 0.25, "probe_middle": 0.5, "probe_high": 0.75}[move]
        return state.low + max(0, min(span - 1, int(span * share)))

    def step(self, state: InquiryState, move: str, rng: random.Random) -> tuple[InquiryState, Rewards]:
        if move not in self.legal_moves(state):
            raise ValueError(f"{move!r} is not available")
        if move == "guess":
            picked = (state.low + state.high) // 2
            right = picked == state.answer
            nxt = InquiryState(low=state.low, high=state.high, answer=state.answer,
                               tests=state.tests, guessed=True, right=right)
            # Guessing right sooner is worth more, which is what makes a test
            # that rules out a lot better than one that rules out a little.
            return nxt, (1.0 - 0.5 * state.tests / self.BUDGET if right else -1.0,)

        cut = self._threshold(state, move)
        if state.answer > cut:
            low, high = cut + 1, state.high
        else:
            low, high = state.low, cut
        nxt = InquiryState(low=low, high=high, answer=state.answer, tests=state.tests + 1)
        if nxt.tests >= self.BUDGET and low != high:
            return nxt, (-1.0,)
        return nxt, (0.0,)

    def is_terminal(self, state: InquiryState) -> bool:
        # Out of tests ends the hand whether or not the answer was found, which
        # it did not used to: a state narrowed to one candidate with the budget
        # spent had no exit at all.
        return state.guessed or state.tests >= self.BUDGET

    def key(self, state: InquiryState) -> Hashable:
        # The answer is hidden; only what the tests revealed may be used.
        return (state.high - state.low, state.tests, state.guessed)

    def outcome(self, state: InquiryState) -> Rewards:
        if state.guessed:
            return (1.0 if state.right else -1.0,)
        return (1.0,) if state.low == state.high else (-1.0,)

    def features(self, state: InquiryState) -> dict[str, float]:
        left = state.high - state.low + 1
        bits = math.log2(max(1, left))
        tests_left = max(0, self.BUDGET - state.tests)
        return {
            "bias": 1.0,
            "progress": state.tests / self.BUDGET,
            # A good position is one where little is still unknown.
            "my_strength": 1.0 - bits / math.log2(self.RANGE),
            # What is against you is uncertainty you no longer have time for.
            "their_strength": min(1.0, bits / max(1.0, tests_left)),
            "win_available": 1.0 if left == 1 else 0.0,
            "must_block": 1.0 if bits > tests_left else 0.0,
            "inquiry:span": left / self.RANGE,
        }

    def render(self, state: InquiryState) -> str:
        return (f"  somewhere in {state.low}..{state.high}\n"
                f"  tests used: {state.tests}/{self.BUDGET}")

    def move_label(self, move: str) -> str:
        return move

    def parse_move(self, text: str, state: InquiryState) -> str:
        choice = text.strip().lower()
        aliases = {"l": "probe_low", "m": "probe_middle", "h": "probe_high", "g": "guess"}
        choice = aliases.get(choice, choice)
        if choice not in self.legal_moves(state):
            raise ValueError(f"try one of {self.MOVES}")
        return choice


# -- hidden syntax ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GrammarState:
    written: tuple[str, ...]
    broken: bool = False


class Grammar(Game):
    """Build a string as long as you can without breaking a rule you were never told.

    The grammar is fixed, so it can be learned; nothing announces it, so it has
    to be learned by being cut off. Which letter may follow which is the only
    structure there is, and it is a different structure from a board.
    """

    name = "grammar"
    title = "Grammar (learn the rule)"
    num_players = 1
    deterministic = True
    difficulty = 2

    ALPHABET = ("a", "b", "c", "d")
    #: What may follow what. Nothing may repeat, and 'd' is a near-dead end.
    ALLOWED = {
        "": ("a", "b"),
        "a": ("b", "c"),
        "b": ("c", "d"),
        "c": ("a", "d"),
        "d": ("a",),
    }
    MAX_LENGTH = 12

    def initial_state(self, rng: random.Random) -> GrammarState:
        return GrammarState(written=())

    def current_player(self, state: GrammarState) -> int:
        return 0

    def legal_moves(self, state: GrammarState) -> list[str]:
        # Every letter may be *attempted*; that is the point. Only the rule
        # decides whether the attempt ends the episode.
        return [] if self.is_terminal(state) else list(self.ALPHABET)

    def _valid(self, state: GrammarState) -> tuple[str, ...]:
        return self.ALLOWED[state.written[-1] if state.written else ""]

    def step(self, state: GrammarState, move: str, rng: random.Random) -> tuple[GrammarState, Rewards]:
        if move not in self.ALPHABET:
            raise ValueError(f"{move!r} is not a letter")
        if move not in self._valid(state):
            return GrammarState(written=state.written + (move,), broken=True), (-1.0,)
        nxt = GrammarState(written=state.written + (move,))
        # A small payment per letter, so length is worth something on its own.
        return nxt, (0.25 if len(nxt.written) >= self.MAX_LENGTH else 0.1,)

    def is_terminal(self, state: GrammarState) -> bool:
        return state.broken or len(state.written) >= self.MAX_LENGTH

    def key(self, state: GrammarState) -> Hashable:
        # Only the last letter and the length matter to what may come next.
        return (state.written[-1] if state.written else "", len(state.written), state.broken)

    def outcome(self, state: GrammarState) -> Rewards:
        return (float(len(state.written)) / self.MAX_LENGTH * (0.0 if state.broken else 1.0),)

    def features(self, state: GrammarState) -> dict[str, float]:
        valid = self._valid(state)
        traps = sum(1 for letter in valid if len(self.ALLOWED[letter]) <= 1)
        return {
            "bias": 1.0,
            "progress": len(state.written) / self.MAX_LENGTH,
            # A good position is one with room to carry on.
            "my_strength": len(valid) / len(self.ALPHABET),
            # What is against you is that some of those ways out are traps.
            "their_strength": (traps / len(valid)) if valid else 1.0,
            "win_available": 1.0 if len(state.written) >= self.MAX_LENGTH - 1 and valid else 0.0,
            "must_block": 1.0 if not valid else 0.0,
            "grammar:at_dead_end": 1.0 if len(valid) <= 1 else 0.0,
        }

    def render(self, state: GrammarState) -> str:
        written = "".join(state.written) or "(nothing yet)"
        return f"  written: {written}\n  {'broke the rule' if state.broken else 'still going'}"

    def move_label(self, move: str) -> str:
        return move

    def parse_move(self, text: str, state: GrammarState) -> str:
        choice = text.strip().lower()
        if choice not in self.ALPHABET:
            raise ValueError(f"letters are {self.ALPHABET}")
        return choice
