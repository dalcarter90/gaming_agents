"""Measuring how good an agent actually is.

Evaluation always runs with learning and exploration switched off, and in
two-player games it alternates who moves first -- otherwise you are measuring
the first-move advantage as much as the agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..agents.base import Agent
from ..core.episode import play_episode
from ..core.game import Game


@dataclass
class EvalReport:
    """The result of a batch of evaluation episodes."""

    game: str
    agent: str
    opponent: str | None
    episodes: int
    wins: int = 0
    draws: int = 0
    losses: int = 0
    mean_outcome: float = 0.0
    """Average of the agent's own entry in :meth:`Game.outcome`."""
    mean_turns: float = 0.0
    scores: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def non_loss_rate(self) -> float:
        """Share of games not lost. The right bar for a game that is a draw
        with correct play, like Tic-Tac-Toe."""
        return (self.wins + self.draws) / self.episodes if self.episodes else 0.0

    def metric(self, name: str) -> float:
        try:
            return {
                "win_rate": self.win_rate,
                "non_loss_rate": self.non_loss_rate,
                "mean_outcome": self.mean_outcome,
            }[name]
        except KeyError:
            raise ValueError(f"unknown metric {name!r}") from None

    def summary(self) -> str:
        if self.opponent is None:
            # Solo scores span everything from a fraction of a bet in Blackjack
            # to five figures in 2048, so pick the precision to match.
            score = f"{self.mean_outcome:+.3f}" if abs(self.mean_outcome) < 10 else f"{self.mean_outcome:,.1f}"
            return f"{self.episodes} games, mean score {score}, {self.mean_turns:.0f} turns avg"
        return (
            f"{self.wins}W/{self.draws}D/{self.losses}L over {self.episodes} "
            f"(win {self.win_rate:.0%}, non-loss {self.non_loss_rate:.0%})"
        )


def evaluate(
    game: Game,
    agent: Agent,
    opponent: Agent | None,
    episodes: int,
    rng: random.Random,
    *,
    alternate_seats: bool = True,
) -> EvalReport:
    """Play ``episodes`` non-learning games and summarise how ``agent`` did."""
    if game.num_players == 1:
        opponent = None
    elif opponent is None:
        raise ValueError(f"{game.name} needs an opponent to evaluate against")

    was_training = [agent.training] + ([opponent.training] if opponent else [])
    agent.training = False
    if opponent:
        opponent.training = False

    report = EvalReport(
        game=game.name,
        agent=agent.name,
        opponent=opponent.name if opponent else None,
        episodes=episodes,
    )
    total_outcome = 0.0
    total_turns = 0

    try:
        for i in range(episodes):
            if opponent is None:
                seat = 0
                line_up = [agent]
            else:
                seat = (i % 2) if alternate_seats else 0
                line_up = [agent, opponent] if seat == 0 else [opponent, agent]

            result = play_episode(game, line_up, rng, learn=False)
            score = result.outcome[seat]
            total_outcome += score
            total_turns += result.turns
            report.scores.append(score)

            if game.num_players > 1:
                if score > 0:
                    report.wins += 1
                elif score < 0:
                    report.losses += 1
                else:
                    report.draws += 1
    finally:
        agent.training = was_training[0]
        if opponent:
            opponent.training = was_training[1]

    report.mean_outcome = total_outcome / episodes if episodes else 0.0
    report.mean_turns = total_turns / episodes if episodes else 0.0
    return report
