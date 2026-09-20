"""Samples-to-competence, achievement ladders, and intervals.

Three things this project was measuring badly or not at all: how *fast* an
agent gets good rather than how good it ends up, how far a losing episode got,
and how much of any reported difference is just noise.
"""

import math
import random

import pytest

from gaming_agents.registry import make_agent, make_game
from gaming_agents.training.achievements import AchievementReport, measure_achievements
from gaming_agents.training.competence import (
    confidence_interval,
    default_judge,
    samples_to_competence,
)
from gaming_agents.training.trainer import train


# -- intervals ------------------------------------------------------------

def test_an_interval_brackets_the_mean():
    mean, low, high = confidence_interval([100.0, 200.0, 150.0])
    assert low < mean < high
    assert mean == pytest.approx(150.0)


def test_more_seeds_narrow_the_interval():
    """The reason to run five instead of three."""
    _, few_low, few_high = confidence_interval([10.0, 20.0, 30.0])
    _, many_low, many_high = confidence_interval([10.0, 20.0, 30.0] * 4)
    assert (many_high - many_low) < (few_high - few_low)


def test_identical_runs_give_a_point_interval():
    mean, low, high = confidence_interval([7.0, 7.0, 7.0])
    assert (mean, low, high) == (7.0, 7.0, 7.0)


def test_one_run_admits_it_knows_nothing():
    mean, low, high = confidence_interval([42.0])
    assert mean == 42.0 and low == float("-inf") and high == float("inf")


def test_no_measurements_is_an_error():
    with pytest.raises(ValueError):
        confidence_interval([])


# -- samples to competence ------------------------------------------------

def test_an_already_competent_agent_needs_no_episodes(rng):
    """A childhood that already clears the bar should read as zero, not as one
    training step -- the distinction is the whole point of the measure."""
    game = make_game("connect4")

    def build(seed):
        agent = make_agent("linear", alpha=0.02, epsilon=0.2, lookahead_samples=4)
        train(game, agent, 400, random.Random(seed),
              opponent=make_agent("random"), eval_episodes=3)
        return agent

    result = samples_to_competence(game, build, 0.90, default_judge(200),
                                   seeds=(0, 1), step=50, ceiling=200)
    assert result.samples == [0, 0]


def test_an_unreachable_bar_is_reported_as_unreached(rng):
    game = make_game("connect4")
    build = lambda _seed: make_agent("linear", alpha=0.02, lookahead_samples=4)  # noqa: E731
    result = samples_to_competence(game, build, 2.0, default_judge(100),
                                   seeds=(0,), step=50, ceiling=100)
    assert result.samples == [None]
    assert result.success_rate == 0.0
    assert "never reached" in result.summary()


def test_a_harder_bar_takes_longer(rng):
    game = make_game("connect4")
    build = lambda _seed: make_agent("linear", alpha=0.02, epsilon=0.2, lookahead_samples=4)  # noqa: E731
    easy = samples_to_competence(game, build, 0.85, default_judge(300),
                                 seeds=(0, 1), step=25, ceiling=600)
    hard = samples_to_competence(game, build, 0.99, default_judge(300),
                                 seeds=(0, 1), step=25, ceiling=600)
    assert sum(easy.reached) <= sum(hard.reached)


# -- achievements ---------------------------------------------------------

def test_2048_declares_its_whole_ladder():
    """Declared, not discovered: a milestone nothing has reached yet must still
    count against the score."""
    game = make_game("2048")
    declared = game.all_achievements()
    assert "tile_2048" in declared, "the hardest rung has to be in the list"
    assert "tile_32" in declared
    assert len(declared) == len(set(declared))


def test_games_without_a_ladder_have_none(rng):
    for name in ("connect4", "nim", "kuhn"):
        assert make_game(name).all_achievements() == ()


def test_the_ladder_unlocks_in_order():
    from gaming_agents.games.twenty_forty_eight import Game2048State

    game = make_game("2048")
    modest = game.achievements(Game2048State(cells=(64,) + (0,) * 15))
    grand = game.achievements(Game2048State(cells=(512,) + (0,) * 15))
    assert "tile_64" in modest and "tile_512" not in modest
    # A big tile implies every smaller rung; the ladder cannot be skipped.
    assert modest <= grand


def test_how_the_board_was_played_is_credited_too():
    from gaming_agents.games.twenty_forty_eight import Game2048State

    game = make_game("2048")
    anchored = Game2048State(cells=(512, 2, 0, 0) + (0,) * 12)
    stranded = Game2048State(cells=(0, 0, 0, 0, 0, 512, 2, 0) + (0,) * 8)
    assert "anchored_a_big_tile" in game.achievements(anchored)
    assert "anchored_a_big_tile" not in game.achievements(stranded)


def test_the_geometric_score_refuses_to_be_gamed_by_easy_wins():
    """Banking an easy achievement every time must not substitute for ever
    reaching a hard one. That is the whole reason it is not an average."""
    ladder = ("easy", "medium", "hard")
    lopsided = AchievementReport(game="x", agent="a", episodes=100, possible=ladder)
    lopsided.unlocks.update({"easy": 100, "medium": 100})     # hard: never

    balanced = AchievementReport(game="x", agent="a", episodes=100, possible=ladder)
    balanced.unlocks.update({"easy": 70, "medium": 55, "hard": 40})

    assert lopsided.arithmetic_score() > balanced.arithmetic_score()
    assert balanced.geometric_score() > lopsided.geometric_score()


def test_never_unlocking_something_caps_the_score():
    ladder = ("a", "b")
    report = AchievementReport(game="x", agent="a", episodes=50, possible=ladder)
    report.unlocks.update({"a": 50})
    # exp(mean(ln 101, ln 1)) - 1 = sqrt(101) - 1
    assert report.geometric_score() == pytest.approx(math.sqrt(101.0) - 1.0)


def test_a_perfect_run_scores_a_hundred():
    ladder = ("a", "b", "c")
    report = AchievementReport(game="x", agent="a", episodes=10, possible=ladder)
    report.unlocks.update({name: 10 for name in ladder})
    assert report.geometric_score() == pytest.approx(100.0)


def test_training_lifts_the_achievement_score(rng):
    game = make_game("2048")
    agent = make_agent("linear", alpha=0.02, epsilon=0.15, epsilon_decay=0.999,
                       lookahead_samples=4)
    train(game, agent, 200, random.Random(0), eval_episodes=3)

    trained = measure_achievements(game, agent, 20, random.Random(77))
    chance = measure_achievements(game, make_agent("random"), 20, random.Random(77))
    assert trained.geometric_score() > chance.geometric_score() * 2


def test_a_losing_episode_still_gets_credit_for_how_far_it_got(rng):
    """Every 2048 episode ends in a jammed board. If only the ending counted,
    all of them would look identical."""
    game = make_game("2048")
    report = measure_achievements(game, make_agent("random"), 15, random.Random(5))
    assert report.rate("tile_32") > 0.5, "an easy rung should be common"
    assert report.rate("tile_2048") == 0.0, "and a hard one should not"


def test_achievement_scoring_refuses_two_player_games(rng):
    with pytest.raises(ValueError, match="single-player"):
        measure_achievements(make_game("connect4"), make_agent("random"), 2, rng)


def test_a_game_with_no_ladder_cannot_be_scored_this_way(rng):
    with pytest.raises(ValueError, match="no achievements"):
        measure_achievements(make_game("blackjack"), make_agent("random"), 2, rng)
