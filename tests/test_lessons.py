"""The three tasks that are not games.

They exist to be a childhood, so what matters is that they are real tasks --
learnable, failable, and describing themselves in the same vocabulary a game
would use. A lesson nothing can learn teaches nothing, and an experiment built
on it would be measuring noise.
"""

import random

import pytest

from gaming_agents.core.game import Game
from gaming_agents.games.lessons import (
    Grammar,
    GrammarState,
    Inquiry,
    InquiryState,
    Reckoning,
    ReckoningState,
)
from gaming_agents.registry import make_agent, make_game
from gaming_agents.training.evaluate import evaluate
from gaming_agents.training.trainer import train

LESSONS = ["reckoning", "inquiry", "grammar"]


@pytest.mark.parametrize("name", LESSONS)
def test_a_lesson_speaks_the_universal_vocabulary(name, rng):
    game = make_game(name)
    described = set(game.features(game.initial_state(rng)))
    assert set(Game.UNIVERSAL_FEATURES) <= described


@pytest.mark.parametrize("name", LESSONS)
def test_a_lesson_is_learnable(name, rng):
    """If practice does not improve it, using it as a childhood is measuring
    nothing at all."""
    game = make_game(name)
    before = evaluate(game, make_agent("random"), None, 800, random.Random(7)).mean_outcome
    agent = make_agent("linear", alpha=0.02, epsilon=0.2, epsilon_decay=0.999, lookahead_samples=4)
    train(game, agent, 1_500, random.Random(0), eval_episodes=3)
    after = evaluate(game, agent, None, 800, random.Random(7)).mean_outcome
    assert after > before + 0.2, f"{name}: {before:.3f} -> {after:.3f} is not learning"


@pytest.mark.parametrize("name", LESSONS)
def test_a_lesson_can_be_failed(name, rng):
    """A task nobody can lose is not a task."""
    game = make_game(name)
    losses = 0
    for trial in range(40):
        rng = random.Random(trial)
        state = game.initial_state(rng)
        for _ in range(200):  # a cap, so a task with no exit fails loudly
            if game.is_terminal(state):
                break
            state, _ = game.step(state, rng.choice(game.legal_moves(state)), rng)
        else:
            pytest.fail(f"{name}: an episode ran 200 moves without ending")
        losses += game.outcome(state)[0] <= 0
    assert losses > 0


@pytest.mark.parametrize("name", LESSONS)
def test_a_lesson_always_ends(name, rng):
    """Inquiry once had a corner with no exit: narrowed to one candidate with
    the test budget spent, it was neither terminal nor able to progress."""
    game = make_game(name)
    for trial in range(60):
        source = random.Random(trial)
        state = game.initial_state(source)
        for _ in range(200):
            if game.is_terminal(state):
                break
            state, _ = game.step(state, source.choice(game.legal_moves(state)), source)
        assert game.is_terminal(state), f"{name}: stuck after 200 moves"


# -- reckoning ------------------------------------------------------------

def test_hitting_the_number_exactly_is_the_win(rng):
    game = Reckoning()
    state = ReckoningState(value=9, target=10, steps=0)
    nxt, reward = game.step(state, "+1", rng)
    assert game.is_terminal(nxt) and reward == (1.0,)
    assert game.outcome(nxt) == (1.0,)


def test_running_out_of_moves_is_the_loss(rng):
    game = Reckoning()
    state = ReckoningState(value=5, target=99, steps=Reckoning.BUDGET - 1)
    nxt, reward = game.step(state, "+1", rng)
    assert game.is_terminal(nxt) and reward == (-1.0,)


def test_moves_that_leave_the_board_are_not_offered():
    game = Reckoning()
    assert "x2" not in game.legal_moves(ReckoningState(value=150, target=20, steps=0))
    assert "-3" not in game.legal_moves(ReckoningState(value=-18, target=20, steps=0))


def test_being_one_move_from_the_target_is_visible():
    game = Reckoning()
    described = game.features(ReckoningState(value=9, target=10, steps=0))
    assert described["win_available"] == 1.0
    assert described["my_strength"] > 0.9


def test_being_far_away_reads_as_weak():
    game = Reckoning()
    near = game.features(ReckoningState(value=95, target=99, steps=0))
    far = game.features(ReckoningState(value=0, target=99, steps=0))
    assert near["my_strength"] > far["my_strength"]


# -- inquiry --------------------------------------------------------------

def test_a_test_rules_out_roughly_half(rng):
    game = Inquiry()
    state = InquiryState(low=1, high=64, answer=50, tests=0)
    nxt, _ = game.step(state, "probe_middle", rng)
    assert (nxt.high - nxt.low) < (state.high - state.low)
    assert nxt.low <= 50 <= nxt.high, "the answer must stay inside the bracket"


def test_the_hidden_answer_never_reaches_the_state_key():
    """The single thing that would make this task a fiction."""
    game = Inquiry()
    one = InquiryState(low=1, high=64, answer=7, tests=2)
    two = InquiryState(low=1, high=64, answer=61, tests=2)
    assert game.key(one) == game.key(two)


def test_narrowing_to_one_candidate_is_a_certain_win(rng):
    game = Inquiry()
    state = InquiryState(low=40, high=40, answer=40, tests=4)
    assert game.features(state)["win_available"] == 1.0
    nxt, reward = game.step(state, "guess", rng)
    assert nxt.right and reward[0] > 0


def test_too_much_left_to_find_in_the_time_shows_as_danger():
    game = Inquiry()
    doomed = game.features(InquiryState(low=1, high=64, answer=9, tests=Inquiry.BUDGET - 1))
    comfortable = game.features(InquiryState(low=1, high=4, answer=2, tests=1))
    assert doomed["must_block"] == 1.0
    assert comfortable["must_block"] == 0.0


def test_guessing_sooner_is_worth_more(rng):
    game = Inquiry()
    quick = game.step(InquiryState(low=9, high=9, answer=9, tests=1), "guess", rng)[1][0]
    slow = game.step(InquiryState(low=9, high=9, answer=9, tests=6), "guess", rng)[1][0]
    assert quick > slow


# -- grammar --------------------------------------------------------------

def test_breaking_the_rule_ends_it(rng):
    game = Grammar()
    nxt, reward = game.step(GrammarState(written=("a",)), "a", rng)
    assert nxt.broken and reward == (-1.0,)
    assert game.outcome(nxt) == (0.0,)


def test_following_the_rule_pays(rng):
    game = Grammar()
    nxt, reward = game.step(GrammarState(written=("a",)), "b", rng)
    assert not nxt.broken and reward[0] > 0


def test_every_letter_may_be_attempted():
    """The rule is never announced, so it has to be learnable by being cut off."""
    game = Grammar()
    assert set(game.legal_moves(GrammarState(written=("a",)))) == set(Grammar.ALPHABET)


def test_a_cramped_position_reads_as_weaker():
    game = Grammar()
    roomy = game.features(GrammarState(written=("a",)))       # b or c may follow
    cramped = game.features(GrammarState(written=("d",)))     # only a may follow
    assert roomy["my_strength"] > cramped["my_strength"]
    assert cramped["grammar:at_dead_end"] == 1.0


def test_only_the_last_letter_decides_what_may_come_next():
    game = Grammar()
    assert game.key(GrammarState(written=("a", "b", "c"))) == game.key(
        GrammarState(written=("d", "a", "c"))
    )
