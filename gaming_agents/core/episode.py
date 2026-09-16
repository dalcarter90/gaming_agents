"""Running a single episode, and feeding the result back to learning agents."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from .game import Game, Move, Rewards, State, check_legal
from .transition import Transition

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from ..agents.base import Agent

#: Safety net so a buggy game or agent cannot hang a training run.
MAX_TURNS = 10_000


@dataclass
class TurnRecord:
    """One entry of a played-out game, for transcripts and debugging."""

    seat: int
    board: str
    move: str
    rewards: Rewards


@dataclass
class EpisodeResult:
    """What happened in one episode."""

    outcome: Rewards
    """Per-seat final result, straight from :meth:`Game.outcome`."""
    total_rewards: Rewards
    """Per-seat sum of the shaped rewards handed out during the episode."""
    turns: int
    final_state: State
    transcript: list[TurnRecord] = field(default_factory=list)
    truncated: bool = False
    """True if the episode hit :data:`MAX_TURNS` instead of ending naturally."""

    @property
    def winner(self) -> int | None:
        """Seat with the strictly highest outcome, or None if it is a tie."""
        best = max(self.outcome)
        leaders = [i for i, v in enumerate(self.outcome) if v == best]
        return leaders[0] if len(leaders) == 1 else None


class _Pending:
    """A transition waiting for the seat's next turn to close it."""

    __slots__ = ("state", "key", "move", "reward")

    def __init__(self, state: State, key, move: Move) -> None:
        self.state = state
        self.key = key
        self.move = move
        self.reward = 0.0


def play_episode(
    game: Game,
    agents: Sequence["Agent"],
    rng: random.Random,
    *,
    learn: bool = True,
    record: bool = False,
    max_turns: int = MAX_TURNS,
) -> EpisodeResult:
    """Play one episode of ``game`` with ``agents`` seated in order.

    ``agents[i]`` occupies seat ``i``. The same agent object may occupy several
    seats — that is exactly what self-play is, and the per-seat bookkeeping
    below keeps the two seats' transitions separate.
    """
    if len(agents) != game.num_players:
        raise ValueError(f"{game.name} needs {game.num_players} agents, got {len(agents)}")

    state = game.initial_state(rng)
    for seat, agent in enumerate(agents):
        agent.start_episode(game, seat)

    pending: dict[int, _Pending] = {}
    totals = [0.0] * game.num_players
    transcript: list[TurnRecord] = []
    turns = 0
    truncated = False

    while not game.is_terminal(state):
        if turns >= max_turns:
            truncated = True
            break

        seat = game.current_player(state)
        legal = game.legal_moves(state)
        if not legal:
            raise ValueError(f"{game.name}: non-terminal state with no legal moves")

        # The seat is acting again, so its previous decision can be settled.
        prior = pending.pop(seat, None)
        if prior is not None and learn:
            agents[seat].learn(
                game,
                Transition(
                    seat=seat,
                    state=prior.state,
                    key=prior.key,
                    move=prior.move,
                    reward=prior.reward,
                    next_state=state,
                    next_key=game.key(state),
                    next_legal=tuple(legal),
                    done=False,
                ),
            )

        move = agents[seat].select_move(game, state, legal, rng)
        check_legal(game, state, move, legal)

        next_state, rewards = game.step(state, move, rng)
        if len(rewards) != game.num_players:
            raise ValueError(f"{game.name}.step returned {len(rewards)} rewards")

        if record:
            transcript.append(
                TurnRecord(seat=seat, board=game.render(state), move=game.move_label(move), rewards=rewards)
            )

        pending[seat] = _Pending(state, game.key(state), move)
        # Credit every reward to whichever seat's open decision is responsible
        # for it. A seat that has not moved yet cannot be credited, and that is
        # correct: nothing it did caused the reward.
        for open_seat, entry in pending.items():
            entry.reward += rewards[open_seat]
        for i, r in enumerate(rewards):
            totals[i] += r

        state = next_state
        turns += 1

    if learn:
        for seat, entry in pending.items():
            agents[seat].learn(
                game,
                Transition(
                    seat=seat,
                    state=entry.state,
                    key=entry.key,
                    move=entry.move,
                    reward=entry.reward,
                    next_state=None,
                    next_key=None,
                    next_legal=(),
                    done=not truncated,
                ),
            )

    outcome = game.outcome(state) if game.is_terminal(state) else game.zeros()
    for seat, agent in enumerate(agents):
        agent.end_episode(game, state, outcome, seat)

    return EpisodeResult(
        outcome=outcome,
        total_rewards=tuple(totals),
        turns=turns,
        final_state=state,
        transcript=transcript,
        truncated=truncated,
    )
