"""The curriculum: a ladder of games an agent climbs one rung at a time.

Each stage trains the agent in rounds and evaluates it after every round. The
agent is only promoted to the next game once it clears that stage's mastery
bar, so "progress from simple games to advanced ones" is an actual gate rather
than a fixed script -- a learner that cannot yet beat random at Nim does not
get to waste its episodes on Connect Four.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agents.base import Agent
from ..registry import make_agent, make_game
from .evaluate import EvalReport, evaluate
from .trainer import TrainReport, train

Reporter = Callable[[str], None]


@dataclass(frozen=True)
class Stage:
    """One rung: a game, who to practise against, and the bar to clear."""

    game: str
    goal: str
    """Plain-English description of what mastering this rung means."""
    train_opponent: str | None = None
    """Agent name to spar with. ``None`` means self-play."""
    eval_opponent: str = "random"
    metric: str = "win_rate"
    mastery: float = 0.9
    episodes_per_round: int = 2_000
    eval_episodes: int = 200
    max_rounds: int = 5
    benchmark_opponent: str | None = None
    """Played against for reporting only -- never gates promotion. Use it to
    watch progress against a much stronger opponent than the bar requires."""
    game_kwargs: dict[str, Any] = field(default_factory=dict)
    train_opponent_kwargs: dict[str, Any] = field(default_factory=dict)
    eval_opponent_kwargs: dict[str, Any] = field(default_factory=dict)
    benchmark_opponent_kwargs: dict[str, Any] = field(default_factory=dict)
    tune: dict[str, Any] = field(default_factory=dict)
    """Agent attributes to set on entering this stage, e.g. a higher epsilon
    for a game whose state space the agent has never seen."""


@dataclass
class StageResult:
    stage: Stage
    rounds: int
    episodes: int
    passed: bool
    best: float
    history: list[tuple[int, EvalReport]] = field(default_factory=list)
    benchmark: EvalReport | None = None
    seconds: float = 0.0

    @property
    def game(self) -> str:
        return self.stage.game


@dataclass
class CurriculumResult:
    curriculum: str
    agent: str
    stages: list[StageResult] = field(default_factory=list)
    completed: bool = False
    """True when every stage was cleared."""

    @property
    def total_episodes(self) -> int:
        return sum(s.episodes for s in self.stages)

    def table(self) -> str:
        lines = [f"{'stage':<14} {'rounds':>6} {'episodes':>9} {'score':>9} {'bar':>7}  result"]
        lines.append("-" * 62)
        for result in self.stages:
            mark = "MASTERED" if result.passed else "not yet"
            lines.append(
                f"{result.stage.game:<14} {result.rounds:>6} {result.episodes:>9,} "
                f"{result.best:>9.3f} {result.stage.mastery:>7.2f}  {mark}"
            )
        return "\n".join(lines)


def run_curriculum(
    curriculum: "NamedCurriculum",
    agent: Agent,
    rng: random.Random,
    *,
    gate: bool = True,
    scale: float = 1.0,
    report: Reporter | None = print,
) -> CurriculumResult:
    """Walk ``agent`` up the ladder.

    ``gate`` is the whole point: with it on, failing a stage ends the run,
    because the later games assume the earlier lessons. Turn it off to see how
    far an agent gets on every rung regardless.

    ``scale`` multiplies every episode budget, which is how you trade wall
    time for strength without editing the ladder.
    """
    say: Reporter = report or (lambda _msg: None)
    result = CurriculumResult(curriculum=curriculum.name, agent=agent.name)

    for index, stage in enumerate(curriculum.stages, start=1):
        game = make_game(stage.game, **stage.game_kwargs)
        say(f"\n=== stage {index}/{len(curriculum.stages)}: {game.title} ===")
        say(f"    goal: {stage.goal}")

        # Stage tuning is advisory: the knobs are named for the tabular learner,
        # and an agent that has no epsilon (MCTS, say) should walk the same
        # ladder rather than fall over on a setting that means nothing to it.
        skipped = [a for a in stage.tune if not hasattr(agent, a)]
        for attribute, value in stage.tune.items():
            if attribute not in skipped:
                setattr(agent, attribute, value)
        if skipped:
            say(f"    (ignoring stage settings {', '.join(sorted(skipped))}: not applicable to {agent.name})")

        sparring = _build(stage.train_opponent, stage.train_opponent_kwargs)
        judge = _build(stage.eval_opponent, stage.eval_opponent_kwargs) if game.num_players > 1 else None
        budget = max(1, int(stage.episodes_per_round * scale))

        stage_result = StageResult(stage=stage, rounds=0, episodes=0, passed=False, best=float("-inf"))
        for round_number in range(1, stage.max_rounds + 1):
            training: TrainReport = train(
                game,
                agent,
                budget,
                rng,
                opponent=sparring,
                eval_opponent=judge,
                eval_episodes=stage.eval_episodes,
            )
            stage_result.rounds = round_number
            stage_result.episodes += budget
            stage_result.seconds += training.seconds

            evaluation = training.final
            assert evaluation is not None  # train always evaluates at least once
            stage_result.history.append((stage_result.episodes, evaluation))
            score = evaluation.metric(stage.metric)
            stage_result.best = max(stage_result.best, score)

            say(
                f"    round {round_number}: {stage_result.episodes:,} episodes  "
                f"{stage.metric}={score:.3f} (bar {stage.mastery:.2f})  {evaluation.summary()}"
            )

            if score >= stage.mastery:
                stage_result.passed = True
                break

        if stage.benchmark_opponent and game.num_players > 1:
            reference = _build(stage.benchmark_opponent, stage.benchmark_opponent_kwargs)
            stage_result.benchmark = evaluate(game, agent, reference, stage.eval_episodes, rng)
            say(f"    benchmark vs {stage.benchmark_opponent}: {stage_result.benchmark.summary()}")

        result.stages.append(stage_result)

        if stage_result.passed:
            say(f"    -> mastered {game.title}; promoting")
        else:
            say(f"    -> did not clear the bar for {game.title} (best {stage_result.best:.3f})")
            if gate:
                say("    -> stopping here: later rungs build on this one")
                return result

    result.completed = all(s.passed for s in result.stages)
    return result


def _build(name: str | None, kwargs: dict[str, Any]) -> Agent | None:
    return make_agent(name, **kwargs) if name else None


@dataclass(frozen=True)
class NamedCurriculum:
    name: str
    description: str
    stages: tuple[Stage, ...]


# The bars below are calibrated, not guessed: each one sits above what a
# non-learner scores on that game and inside what a tabular learner reaches in
# the episode budget it is given. 2048 is the deliberate exception -- see SOLO.

CLASSIC = NamedCurriculum(
    name="classic",
    description="Two-player ladder: Nim -> Tic-Tac-Toe -> mini Connect Four -> Connect Four.",
    stages=(
        Stage(
            game="nim",
            goal="beat a random player essentially every time",
            eval_opponent="random",
            metric="win_rate",
            mastery=0.90,
            episodes_per_round=3_000,
            eval_episodes=300,
            max_rounds=4,
            benchmark_opponent="perfect-nim",
            tune={"epsilon": 0.30, "epsilon_decay": 0.9995, "alpha": 0.3, "gamma": 0.95},
        ),
        Stage(
            game="tictactoe",
            goal="never lose -- correct play here is a draw, so anything less is a mistake",
            eval_opponent="minimax",
            metric="non_loss_rate",
            mastery=0.95,
            episodes_per_round=20_000,
            eval_episodes=200,
            max_rounds=5,
            benchmark_opponent="random",
            tune={"epsilon": 0.35, "epsilon_decay": 0.99995, "alpha": 0.2, "gamma": 0.95},
        ),
        Stage(
            game="connect4-mini",
            # Self-play stalls here: on a 4x5 board it explores lines a random
            # opponent never plays, so the table fills up with positions that
            # never come up. Sparring against the opponent it is judged on is
            # what gets it over the bar.
            goal="win three games in four against a random player",
            train_opponent="random",
            eval_opponent="random",
            metric="win_rate",
            mastery=0.75,
            episodes_per_round=20_000,
            eval_episodes=300,
            max_rounds=5,
            benchmark_opponent="mcts",
            benchmark_opponent_kwargs={"simulations": 60},
            tune={"epsilon": 0.30, "epsilon_decay": 0.99995, "alpha": 0.15, "gamma": 0.95},
        ),
        Stage(
            game="connect4",
            goal="carry the idea to the full 6x7 board",
            train_opponent="random",
            eval_opponent="random",
            metric="win_rate",
            mastery=0.80,
            episodes_per_round=40_000,
            eval_episodes=400,
            max_rounds=5,
            benchmark_opponent="mcts",
            benchmark_opponent_kwargs={"simulations": 60},
            # A bigger board means almost every position is seen once, so the
            # learning rate has to be high enough for a single visit to teach
            # something. alpha=0.1 plateaus below the bar; 0.25 clears it.
            tune={"epsilon": 0.35, "epsilon_decay": 0.99997, "alpha": 0.25, "gamma": 0.95},
        ),
    ),
)

SOLO = NamedCurriculum(
    name="solo",
    description="Single-player ladder against chance: Blackjack -> 2048.",
    stages=(
        Stage(
            game="blackjack",
            goal="lose under a tenth of a bet per hand -- the house always wins a little",
            metric="mean_outcome",
            mastery=-0.10,
            episodes_per_round=50_000,
            eval_episodes=10_000,
            max_rounds=3,
            # Blackjack hands are short and every one ends in a real payoff, so
            # there is nothing to discount: gamma stays at 1.
            tune={"epsilon": 0.25, "epsilon_decay": 0.99999, "alpha": 0.05, "gamma": 1.0},
        ),
        Stage(
            game="2048",
            # A random player already averages around 1,100 here, and a table
            # keyed on exact boards will never see the same board twice. This
            # rung is meant to be where tabular learning runs out of road --
            # run it with `--agent mcts` to watch search clear it instead.
            goal="average 3,000 points, roughly three times what random play stumbles into",
            metric="mean_outcome",
            mastery=3_000.0,
            episodes_per_round=5_000,
            eval_episodes=20,
            max_rounds=3,
            tune={"epsilon": 0.20, "epsilon_decay": 0.9995, "alpha": 0.1, "gamma": 0.95},
        ),
    ),
)

def _by_difficulty(*ladders: NamedCurriculum) -> tuple[Stage, ...]:
    """Merge ladders into one, ordered by how hard each game is.

    Sorting by the games' own ``difficulty`` rather than hard-coding an order
    means a newly registered game slots into the right rung on its own.
    """
    from ..registry import make_game as _make_game

    stages = [stage for ladder in ladders for stage in ladder.stages]
    return tuple(sorted(stages, key=lambda s: _make_game(s.game).difficulty))


FULL = NamedCurriculum(
    name="full",
    description="Every game, interleaved into one ladder from easiest to hardest.",
    stages=_by_difficulty(CLASSIC, SOLO),
)

CURRICULA: dict[str, NamedCurriculum] = {c.name: c for c in (CLASSIC, SOLO, FULL)}


def get_curriculum(name: str) -> NamedCurriculum:
    try:
        return CURRICULA[name]
    except KeyError:
        raise ValueError(f"unknown curriculum {name!r}; try one of {sorted(CURRICULA)}") from None
