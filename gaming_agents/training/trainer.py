"""The self-play training loop.

An agent improves by playing: either against a fixed sparring partner, or
against itself, which is how it keeps facing an opponent at exactly its own
level as it gets better.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..agents.base import Agent
from ..core.episode import play_episode
from ..core.game import Game
from .evaluate import EvalReport, evaluate

#: Called with ``(episodes_done, report_or_None)`` as training progresses.
ProgressHook = Callable[[int, "EvalReport | None"], None]


@dataclass
class TrainReport:
    """What one call to :func:`train` accomplished."""

    game: str
    agent: str
    episodes: int
    seconds: float
    history: list[tuple[int, EvalReport]] = field(default_factory=list)
    """``(episodes_done, evaluation)`` pairs -- the learning curve."""

    @property
    def final(self) -> EvalReport | None:
        return self.history[-1][1] if self.history else None


def train(
    game: Game,
    agent: Agent,
    episodes: int,
    rng: random.Random,
    *,
    opponent: Agent | None = None,
    eval_opponent: Agent | None = None,
    eval_every: int = 0,
    eval_episodes: int = 200,
    progress: ProgressHook | None = None,
) -> TrainReport:
    """Play ``episodes`` learning games of ``game``.

    ``opponent`` is the sparring partner. Leave it as ``None`` in a two-player
    game and the agent plays itself: the same object takes both seats and
    learns from both sides of every game.
    """
    if game.num_players == 1:
        line_up_for = lambda _i: [agent]  # noqa: E731 - trivial and local
    elif opponent is None:
        line_up_for = lambda _i: [agent, agent]  # noqa: E731 - self-play
    else:
        # Alternate seats so the agent learns to play first *and* second.
        line_up_for = lambda i: [agent, opponent] if i % 2 == 0 else [opponent, agent]  # noqa: E731

    report = TrainReport(game=game.name, agent=agent.name, episodes=episodes, seconds=0.0)
    started = time.perf_counter()
    agent.training = True

    for i in range(episodes):
        play_episode(game, line_up_for(i), rng, learn=True)
        done = i + 1
        if eval_every and done % eval_every == 0:
            evaluation = _checkpoint(game, agent, eval_opponent, eval_episodes, rng)
            report.history.append((done, evaluation))
            if progress:
                progress(done, evaluation)
        elif progress:
            progress(done, None)

    if not eval_every or episodes % max(eval_every, 1) != 0:
        evaluation = _checkpoint(game, agent, eval_opponent, eval_episodes, rng)
        report.history.append((episodes, evaluation))
        if progress:
            progress(episodes, evaluation)

    report.seconds = time.perf_counter() - started
    return report


def _checkpoint(
    game: Game,
    agent: Agent,
    eval_opponent: Agent | None,
    eval_episodes: int,
    rng: random.Random,
) -> EvalReport:
    from ..agents.random_agent import RandomAgent

    if game.num_players > 1 and eval_opponent is None:
        eval_opponent = RandomAgent()
    return evaluate(game, agent, eval_opponent, eval_episodes, rng)


def tournament(
    game: Game,
    agents: Sequence[Agent],
    episodes: int,
    rng: random.Random,
) -> list[tuple[str, float, int, int, int]]:
    """Round-robin every agent against every other.

    Returns ``(name, points, wins, draws, losses)`` sorted best first, with a
    win worth 1 point and a draw worth a half.
    """
    if game.num_players != 2:
        raise ValueError("tournaments need a two-player game")

    tally = {a.name: [0.0, 0, 0, 0] for a in agents}
    for i, left in enumerate(agents):
        for right in agents[i + 1 :]:
            report = evaluate(game, left, right, episodes, rng)
            tally[left.name][0] += report.wins + 0.5 * report.draws
            tally[left.name][1] += report.wins
            tally[left.name][2] += report.draws
            tally[left.name][3] += report.losses
            tally[right.name][0] += report.losses + 0.5 * report.draws
            tally[right.name][1] += report.losses
            tally[right.name][2] += report.draws
            tally[right.name][3] += report.wins

    rows = [(name, pts, w, d, ls) for name, (pts, w, d, ls) in tally.items()]
    return sorted(rows, key=lambda row: row[1], reverse=True)
