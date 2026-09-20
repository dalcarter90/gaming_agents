"""The game interface every environment in this project implements.

One interface covers both solitaire-style games (``num_players == 1``) and
turn-based two-player games, and both deterministic and stochastic dynamics.
Agents are written against this interface only, which is what lets the same
learner walk the whole curriculum from Nim up to Connect Four.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Hashable, Sequence

State = Any
Move = Any
Rewards = tuple[float, ...]


class Game(ABC):
    """A turn-based game.

    States are immutable: ``step`` returns a new state rather than mutating the
    one it was given. Agents rely on that when they search (MCTS replays states
    freely) and when they store them in tables.

    Most games here are perfect-information, where a state is simply what
    everyone can see. Poker is not: a state carries cards its holder's opponent
    must not know about. :attr:`imperfect_information` marks those games, and
    :meth:`redeal` is how a searching agent gets a state it is allowed to think
    about. :meth:`key` is the boundary in either case -- see its docstring.
    """

    #: Short identifier used by the registry, the CLI and saved brains.
    name: str = "game"
    #: Human-readable name for reports.
    title: str = "Game"
    #: 1 for solitaire games, 2 for adversarial ones.
    num_players: int = 2
    #: False when ``step`` or ``initial_state`` consume randomness.
    deterministic: bool = True
    #: Rough ordering hint used to build curricula (higher == harder).
    difficulty: int = 0
    #: True when a state holds something the player to move cannot see. Agents
    #: that search must consult :meth:`redeal` rather than the raw state, and
    #: agents that cannot handle hidden information should refuse outright.
    imperfect_information: bool = False

    @abstractmethod
    def initial_state(self, rng: random.Random) -> State:
        """Return a fresh starting state."""

    @abstractmethod
    def current_player(self, state: State) -> int:
        """Index of the player to move in ``state``."""

    @abstractmethod
    def legal_moves(self, state: State) -> list[Move]:
        """Moves available in ``state``; empty iff the state is terminal."""

    @abstractmethod
    def step(self, state: State, move: Move, rng: random.Random) -> tuple[State, Rewards]:
        """Apply ``move`` and return ``(next_state, per_player_rewards)``.

        Rewards are *immediate* — the sum over an episode is what an agent
        maximises. Zero-sum games hand out their payoff only on the final step.
        """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Whether the episode has ended."""

    @abstractmethod
    def key(self, state: State) -> Hashable:
        """What the player to move knows: its information set.

        This is the one place a game says what is visible. A key must contain
        everything the player to move is entitled to use and nothing else --
        in Blackjack that means the dealer's upcard but not the hole card, and
        in poker the player's own hand but not the opponent's.

        Two states sharing a key must be interchangeable for the player to
        move, so a key may canonicalise (folding a mirrored board onto itself,
        say) but must never merge states that need different play.
        """

    @abstractmethod
    def outcome(self, state: State) -> Rewards:
        """Final per-player result of a terminal state, used for scoring.

        Two-player games report ``+1 / -1`` for a win and ``0 / 0`` for a draw
        regardless of how rewards were shaped during the episode. Solo games
        report their natural score.
        """

    #: The vocabulary every game is expected to speak, defined once here so
    #: that a weight learned for one of these means the same thing in the next
    #: game. Each is stated in words first and implemented per game as
    #: faithfully as that game allows:
    #:
    #: ``bias``          always 1; the baseline value of being in this game
    #: ``progress``      how far through the episode, 0 at the start, 1 at the end
    #: ``my_strength``   how good this position is for me, 0 hopeless to 1 winning
    #: ``their_strength``the same for the opponent, as far as it is visible to me
    #: ``win_available`` 1 when I can finish this right now
    #: ``must_block``    1 when I lose shortly unless I do something about it
    #:
    #: Narrower ideas are namespaced -- ``board:`` for things only board games
    #: have, ``<game>:`` for things only one game has -- so they cannot collide
    #: with anything when one set of weights covers every game.
    UNIVERSAL_FEATURES = (
        "bias",
        "progress",
        "my_strength",
        "their_strength",
        "win_available",
        "must_block",
    )

    def features(self, state: State) -> dict[str, float]:
        """Describe ``state`` from the point of view of the player to move.

        This is the seam between memorising and understanding. :meth:`key`
        answers "which position is this?", and a learner built on it can only
        ever look a position up. ``features`` answers "what is this position
        *like*?", and a learner built on it can carry what it learned in one
        position across to every position that resembles it.

        The default is a single indicator for this exact position, which makes
        a linear learner over it precisely a lookup table -- so a game that
        overrides nothing behaves exactly as it did before, and the difference
        a real description makes can be measured against it.

        Values should sit in roughly [-1, 1] so that one learning rate suits
        every feature.
        """
        return {f"{self.name}:is:{self.key(state)!r}": 1.0}

    def action_space(self) -> tuple[Move, ...]:
        """Every move this game can ever offer, in a fixed order.

        A learner that scores each action directly needs a slot per action that
        means the same thing in every position -- which :meth:`legal_moves`
        cannot give, since it returns only what is available right now. Games
        that cannot enumerate their moves return nothing and can only be played
        by looking ahead.
        """
        return ()

    def encode(self, state: State) -> tuple[float, ...]:
        """``state`` as a plain vector of numbers, for a learner with no words.

        :meth:`features` hands a learner concepts a person chose. This hands it
        the position and nothing else, so that anything resembling a concept
        has to be worked out rather than supplied. A game that does not
        override this falls back to its own features in a fixed order, which
        makes the difference between the two measurable: the same network,
        reading hand-written concepts or reading the raw position.

        The vector must be the same length for every state of a game, and its
        entries should sit in roughly [-1, 1].
        """
        described = self.features(state)
        return tuple(described[name] for name in sorted(described))

    def encoding_names(self) -> tuple[str, ...]:
        """What each slot of :meth:`encode` means, for reading a model back."""
        import random as _random

        return tuple(sorted(self.features(self.initial_state(_random.Random(0)))))

    def all_achievements(self) -> tuple[str, ...]:
        """Every milestone this game can award, declared rather than discovered.

        It has to be declared. Finding the list by watching play means the
        hardest milestones are invisible until something reaches them, and a
        score computed over only what was seen reads 100% for an agent that
        never came close to the top of the ladder.
        """
        return ()

    def achievements(self, state: State) -> frozenset[str]:
        """Everything ``state`` has unlocked, by name.

        A single win-or-lose number says almost nothing about *how far* a
        losing episode got. An achievement tree says it directly: which
        milestones were reached, in a game where later ones depend on earlier
        ones. Games without a natural ladder return nothing and are scored the
        ordinary way.

        This is read from a state rather than accumulated by the game, so it
        stays a pure question about a position.
        """
        return frozenset()

    def redeal(self, state: State, seat: int, rng: random.Random) -> State:
        """A state ``seat`` cannot tell apart from this one, with the rest resampled.

        Searching an imperfect-information game from the true state is
        cheating: the search would see the opponent's cards. Instead a searcher
        samples a world consistent with what it actually knows and searches
        that, repeating over many samples. Perfect-information games have
        nothing to hide, so the default returns the state untouched.
        """
        return state

    def initial_outcomes(self) -> list[tuple[State, float]]:
        """Every possible opening state with its probability.

        Only small games can answer this, and those that do can be analysed
        exactly rather than sampled -- it is what lets a poker agent be scored
        on how exploitable it is instead of on whether it got lucky.
        """
        raise NotImplementedError(f"{self.name} cannot enumerate its openings")

    def heuristic(self, state: State, seat: int) -> float:
        """Rough value of ``state`` for ``seat``, in roughly [-1, 1].

        Only consulted by depth-limited search when it runs out of depth
        before the game ends. The default of zero is honest ignorance, which
        is fine for games small enough to search exhaustively.
        """
        return 0.0

    def render(self, state: State) -> str:
        """A printable board. Overridden by every real game."""
        return repr(state)

    def move_label(self, move: Move) -> str:
        """How a move is written in transcripts and at the prompt."""
        return str(move)

    def parse_move(self, text: str, state: State) -> Move:
        """Parse a human-typed move. Raises ``ValueError`` if it is not legal."""
        text = text.strip()
        for move in self.legal_moves(state):
            if self.move_label(move) == text:
                return move
        raise ValueError(f"{text!r} is not a legal move")

    def zeros(self) -> Rewards:
        """An all-zero reward tuple of the right width for this game."""
        return (0.0,) * self.num_players


def check_legal(game: Game, state: State, move: Move, legal: Sequence[Move] | None = None) -> None:
    """Raise if ``move`` is not legal in ``state``. Used by the episode runner."""
    legal = game.legal_moves(state) if legal is None else legal
    if move not in legal:
        raise ValueError(
            f"{game.name}: illegal move {game.move_label(move)!r}; "
            f"legal moves are {[game.move_label(m) for m in legal]}"
        )
