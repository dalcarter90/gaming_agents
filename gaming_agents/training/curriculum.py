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
from .exploitability import exploitability
from .trainer import TrainReport, train

#: Metrics where a smaller number is a better one, so the bar is a ceiling
#: rather than a floor. Poker is scored on how much a strategy leaks.
LOWER_IS_BETTER = {"exploitability"}

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
    """What decides promotion. The win-rate metrics come from the evaluation;
    ``"exploitability"`` is computed separately by exact analysis of the game
    tree, and is the only one where a lower score is a better one."""
    mastery: float = 0.9
    episodes_per_round: int = 2_000
    eval_episodes: int = 200
    max_rounds: int = 5
    benchmark: str | None = None
    """A reference agent, reported but never used to gate promotion.

    In a two-player game the learner is played against it. In a solo game it
    is scored on its own, which turns an abstract bar into "here is what the
    best possible play scores"."""
    game_kwargs: dict[str, Any] = field(default_factory=dict)
    train_opponent_kwargs: dict[str, Any] = field(default_factory=dict)
    eval_opponent_kwargs: dict[str, Any] = field(default_factory=dict)
    benchmark_kwargs: dict[str, Any] = field(default_factory=dict)
    benchmark_episodes: int = 0
    """How many games to score the benchmark over. Defaults to
    ``eval_episodes``; set it lower when the reference is an expensive
    searcher and the number only has to be indicative."""
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
        lines = [f"{'stage':<14} {'rounds':>6} {'episodes':>9} {'score':>9} {'bar':>8}  result"]
        lines.append("-" * 63)
        for result in self.stages:
            mark = "MASTERED" if result.passed else "not yet"
            lines.append(
                f"{result.stage.game:<14} {result.rounds:>6} {result.episodes:>9,} "
                f"{result.best:>9g} {result.stage.mastery:>8g}  {mark}"
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

    # Snapshot every setting any stage touches, so each stage starts from the
    # agent as constructed. Without this a stage silently inherits the previous
    # one's settings -- and since the ladder mixes games that want opposite
    # ones (Blackjack needs a 1/n step size, self-play needs a constant one),
    # that leak would hand a stage the wrong learning rule.
    tunable = {attr for stage in curriculum.stages for attr in stage.tune}
    defaults = {attr: getattr(agent, attr) for attr in tunable if hasattr(agent, attr)}

    for index, stage in enumerate(curriculum.stages, start=1):
        game = make_game(stage.game, **stage.game_kwargs)
        say(f"\n=== stage {index}/{len(curriculum.stages)}: {game.title} ===")
        say(f"    goal: {stage.goal}")

        # Stage tuning is advisory: the knobs are named for the tabular learner,
        # and an agent that has no epsilon (MCTS, say) should walk the same
        # ladder rather than fall over on a setting that means nothing to it.
        for attribute, value in defaults.items():
            setattr(agent, attribute, value)
        skipped = [a for a in stage.tune if not hasattr(agent, a)]
        for attribute, value in stage.tune.items():
            if attribute not in skipped:
                setattr(agent, attribute, value)
        if skipped:
            say(f"    (ignoring stage settings {', '.join(sorted(skipped))}: not applicable to {agent.name})")

        sparring = _build(stage.train_opponent, stage.train_opponent_kwargs)
        judge = _build(stage.eval_opponent, stage.eval_opponent_kwargs) if game.num_players > 1 else None
        budget = max(1, int(stage.episodes_per_round * scale))

        worst = float("inf") if stage.metric in LOWER_IS_BETTER else float("-inf")
        stage_result = StageResult(stage=stage, rounds=0, episodes=0, passed=False, best=worst)
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
            if stage.metric == "exploitability":
                # Not something a match can measure: how much an opponent who
                # knew this strategy could take off it, solved exactly.
                score = exploitability(game, agent, rng).exploitability
            else:
                score = evaluation.metric(stage.metric)

            lower_is_better = stage.metric in LOWER_IS_BETTER
            stage_result.best = (
                min(stage_result.best, score) if lower_is_better else max(stage_result.best, score)
            )

            # ``:g`` rather than a fixed width: the bars in this project range
            # from -0.055 to 3000, and rounding either end to two decimals
            # prints a number that is not the bar.
            say(
                f"    round {round_number}: {stage_result.episodes:,} episodes  "
                f"{stage.metric}={score:g} (bar {stage.mastery:g})  {evaluation.summary()}"
            )

            if (score <= stage.mastery) if lower_is_better else (score >= stage.mastery):
                stage_result.passed = True
                break

        if stage.benchmark:
            reference = _build(stage.benchmark, stage.benchmark_kwargs)
            assert reference is not None
            rounds_of = stage.benchmark_episodes or stage.eval_episodes
            if game.num_players > 1:
                stage_result.benchmark = evaluate(game, agent, reference, rounds_of, rng)
                say(f"    benchmark vs {stage.benchmark}: {stage_result.benchmark.summary()}")
            else:
                # Nothing to play against, so score the reference itself: it is
                # the ceiling the learner's number should be read against.
                stage_result.benchmark = evaluate(game, reference, None, rounds_of, rng)
                say(f"    for reference, {stage.benchmark} scores: {stage_result.benchmark.summary()}")

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
            benchmark="perfect-nim",
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
            benchmark="random",
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
            benchmark="mcts:60",
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
            benchmark="mcts:60",
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
            # Perfect play still loses about 4.7 cents a hand here, so the bar
            # is set just above that rather than anywhere near zero. Beating
            # the house is not on the menu; playing the house correctly is.
            goal="get within a cent a hand of the solved optimum of -0.047",
            metric="mean_outcome",
            mastery=-0.055,
            episodes_per_round=100_000,
            eval_episodes=20_000,
            max_rounds=3,
            benchmark="perfect-blackjack",
            # Two settings here are specific to a game of pure chance. gamma=1
            # because a hand is short and ends in a real payoff, so there is
            # nothing to discount. alpha_mode="visits" because the dealer never
            # changes: a fixed step size keeps bouncing around the answer no
            # matter how many hands it sees, while 1/n actually converges.
            tune={
                "epsilon": 0.30,
                "epsilon_decay": 1.0,
                "alpha_mode": "visits",
                "gamma": 1.0,
            },
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
            benchmark="mcts:20",
            # A searching reference costs ~20 seconds a game here, and five
            # games is plenty to show a tenfold gap.
            benchmark_episodes=5,
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


POKER = NamedCurriculum(
    name="poker",
    description="Kuhn poker, scored on how exploitable the agent is rather than on whether it wins.",
    stages=(
        Stage(
            game="kuhn",
            # Beating a bad opponent proves nothing here -- a strategy that
            # never bluffs and folds too often still beats random. The bar is
            # how much a best response could take, which is zero at
            # equilibrium and cannot be reached by getting lucky.
            goal="leak under a hundredth of a chip a hand to an opponent who knows your strategy",
            metric="exploitability",
            mastery=0.01,
            episodes_per_round=5_000,
            eval_episodes=2_000,
            max_rounds=4,
            benchmark="nash-kuhn",
        ),
    ),
)

FULL = NamedCurriculum(
    name="full",
    description="Every game, interleaved into one ladder from easiest to hardest.",
    stages=_by_difficulty(CLASSIC, SOLO, POKER),
)

CURRICULA: dict[str, NamedCurriculum] = {c.name: c for c in (CLASSIC, SOLO, POKER, FULL)}


def get_curriculum(name: str) -> NamedCurriculum:
    try:
        return CURRICULA[name]
    except KeyError:
        raise ValueError(f"unknown curriculum {name!r}; try one of {sorted(CURRICULA)}") from None
