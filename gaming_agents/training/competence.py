"""How many games it takes to get good, and how sure we are of the answer.

Score-after-N-episodes and episodes-to-reach-a-standard are different
questions, and only the second one asks whether a curriculum made learning
*faster*. An agent that ends up in the same place having needed a tenth of the
practice has gained something a final-score table cannot show.

Everything here reports an interval rather than a number. Training is
stochastic enough that a single run, or a mean of three with no spread quoted,
is not evidence -- a fact this project had to learn the hard way more than
once.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..agents.base import Agent
from ..core.game import Game
from .evaluate import evaluate
from .trainer import train

#: Built an agent, ready to train. Takes the seed so conditions can differ.
AgentFactory = Callable[[int], Agent]
#: Scores a trained agent on the game. Higher is better, always.
Judge = Callable[[Agent, Game], float]

#: Student-t 95% two-sided multipliers, indexed by degrees of freedom. Small
#: seed counts are exactly where the normal approximation lies to you.
_T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def confidence_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float, float]:
    """Return ``(mean, low, high)`` for a small sample, using Student's t.

    With three seeds the interval is wide, and it should be: that width is the
    honest statement of what three runs can tell you.
    """
    if not values:
        raise ValueError("no measurements to summarise")
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, float("-inf"), float("inf")
    if level != 0.95:
        raise ValueError("only the 95% interval is tabulated here")
    spread = statistics.stdev(values) / math.sqrt(len(values))
    degrees = len(values) - 1
    multiplier = _T95.get(degrees) or _T95[min(_T95, key=lambda d: abs(d - degrees))]
    return mean, mean - multiplier * spread, mean + multiplier * spread


@dataclass
class CompetenceResult:
    """How long one condition took to reach the standard, over several seeds."""

    label: str
    game: str
    threshold: float
    #: Episodes needed per seed. ``None`` where the seed never got there.
    samples: list[int | None] = field(default_factory=list)
    #: The score each seed finished on, reached or not.
    finals: list[float] = field(default_factory=list)

    @property
    def reached(self) -> list[int]:
        return [n for n in self.samples if n is not None]

    @property
    def success_rate(self) -> float:
        return len(self.reached) / len(self.samples) if self.samples else 0.0

    def summary(self) -> str:
        if not self.reached:
            return f"{self.label}: never reached {self.threshold:g} in any of {len(self.samples)} runs"
        mean, low, high = confidence_interval([float(n) for n in self.reached])
        got_there = "" if self.success_rate == 1.0 else f", {len(self.reached)}/{len(self.samples)} runs"
        return (f"{self.label}: {mean:,.0f} episodes to reach {self.threshold:g} "
                f"[{max(0.0, low):,.0f}, {high:,.0f}]{got_there}")


def samples_to_competence(
    game: Game,
    build: AgentFactory,
    threshold: float,
    judge: Judge,
    *,
    label: str = "condition",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    step: int = 100,
    ceiling: int = 5_000,
    opponent_name: str | None = "random",
) -> CompetenceResult:
    """Train until ``judge`` clears ``threshold``, and report how long it took.

    ``build`` may return an agent that has already had a childhood elsewhere;
    only the episodes spent on *this* game are counted, which is the point --
    the question is whether prior experience makes the new game quicker.
    """
    from ..registry import make_agent

    result = CompetenceResult(label=label, game=game.name, threshold=threshold)
    for seed in seeds:
        agent = build(seed)
        rng = random.Random(seed)
        spar = make_agent(opponent_name) if (opponent_name and game.num_players > 1) else None

        played, found, score = 0, None, judge(agent, game)
        if score >= threshold:
            found = 0  # the childhood alone was already enough
        while found is None and played < ceiling:
            train(game, agent, step, rng, opponent=spar, eval_episodes=3)
            played += step
            score = judge(agent, game)
            if score >= threshold:
                found = played
        result.samples.append(found)
        result.finals.append(score)
    return result


def default_judge(episodes: int = 300, eval_seed: int = 77) -> Judge:
    """Score by the measure each game is normally judged on, higher-is-better."""
    from ..registry import make_agent

    def judge(agent: Agent, game: Game) -> float:
        rng = random.Random(eval_seed)
        was = agent.training
        agent.training = False
        try:
            if game.num_players == 1:
                return evaluate(game, agent, None, episodes, rng).mean_outcome
            return evaluate(game, agent, make_agent("random"), episodes, rng).win_rate
        finally:
            agent.training = was

    return judge
