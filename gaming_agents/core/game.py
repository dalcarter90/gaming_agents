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
