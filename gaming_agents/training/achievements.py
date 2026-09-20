"""Scoring by what an agent managed, not by whether it won.

A win-or-lose number cannot tell a near miss from a rout, and an average over
milestones lets an agent bank an easy one over and over while never touching a
hard one. The fix, from Hafner's Crafter benchmark, is a geometric mean over
per-achievement success *rates*: rare achievements move the score far more than
further progress on ones already mastered, so there is no way to inflate it by
repeating what is easy.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from ..agents.base import Agent
from ..core.game import Game

#: Stop a single episode from running away, mirroring the episode runner.
MAX_TURNS = 10_000


@dataclass
class AchievementReport:
    game: str
    agent: str
    episodes: int
    #: How many episodes unlocked each achievement at least once.
    unlocks: Counter = field(default_factory=Counter)
    #: Every achievement the game can award, so never-unlocked ones still count.
    possible: tuple[str, ...] = ()

    def rate(self, name: str) -> float:
        """Share of episodes that reached ``name``, in 0..1."""
        return self.unlocks.get(name, 0) / self.episodes if self.episodes else 0.0

    def arithmetic_score(self) -> float:
        """The naive average, kept only to show what it hides."""
        if not self.possible:
            return 0.0
        return 100.0 * sum(self.rate(name) for name in self.possible) / len(self.possible)

    def geometric_score(self) -> float:
        """Crafter's score: ``exp(mean(ln(1 + rate%))) - 1``.

        Adding one percent to an achievement never reached is worth far more
        than adding one percent to an achievement already reached always, which
        is exactly the incentive a hierarchy is supposed to create.
        """
        if not self.possible:
            return 0.0
        logs = [math.log(1.0 + 100.0 * self.rate(name)) for name in self.possible]
        return math.exp(sum(logs) / len(logs)) - 1.0

    def ladder(self) -> list[tuple[str, float]]:
        """Every achievement with its unlock rate, hardest last."""
        return sorted(((name, self.rate(name)) for name in self.possible),
                      key=lambda pair: -pair[1])

    def summary(self) -> str:
        reached = sum(1 for name in self.possible if self.unlocks.get(name))
        return (f"score {self.geometric_score():.1f} "
                f"(naive average would say {self.arithmetic_score():.1f}), "
                f"{reached}/{len(self.possible)} ever unlocked over {self.episodes} episodes")


def achievable(game: Game, rng: random.Random, probes: int = 200) -> tuple[str, ...]:
    """Everything this game ever awards, found by playing it badly and well.

    A game is asked rather than told, so adding an achievement to a game does
    not mean remembering to register it anywhere else.
    """
    from ..registry import make_agent

    seen: set[str] = set()
    scout = make_agent("random")
    for trial in range(probes):
        source = random.Random(rng.randrange(1 << 30))
        state = game.initial_state(source)
        seen |= game.achievements(state)
        for _ in range(MAX_TURNS):
            if game.is_terminal(state):
                break
            moves = game.legal_moves(state)
            state, _ = game.step(state, scout.select_move(game, state, moves, source), source)
            seen |= game.achievements(state)
    return tuple(sorted(seen))


def measure_achievements(
    game: Game,
    agent: Agent,
    episodes: int,
    rng: random.Random,
    *,
    possible: Sequence[str] | None = None,
) -> AchievementReport:
    """Play ``episodes`` and record which milestones each one reached.

    Achievements are collected from every state passed through, so a lost
    episode still gets credit for how far it got -- which is the whole reason
    for preferring this to a terminal score.
    """
    if game.num_players != 1:
        raise ValueError("achievement scoring is defined for single-player games")

    if possible is not None:
        known = tuple(possible)
    else:
        # What the game says it can award, not what happened to turn up.
        known = game.all_achievements() or achievable(game, random.Random(0))
    if not known:
        raise ValueError(f"{game.name} has no achievements to score")
    report = AchievementReport(game=game.name, agent=agent.name, episodes=episodes,
                               possible=known)
    was = agent.training
    agent.training = False
    try:
        for _ in range(episodes):
            state = game.initial_state(rng)
            earned = set(game.achievements(state))
            for _ in range(MAX_TURNS):
                if game.is_terminal(state):
                    break
                moves = game.legal_moves(state)
                state, _ = game.step(state, agent.select_move(game, state, moves, rng), rng)
                earned |= game.achievements(state)
            report.unlocks.update(earned)
    finally:
        agent.training = was
    return report
