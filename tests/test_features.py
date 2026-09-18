"""Describing positions instead of memorising them.

The properties that matter: a description must be blind to things that do not
change the position's character (which mirror image it is), sharp about things
that do (an immediate threat), and the default must reduce exactly to the
lookup table it replaces -- otherwise no comparison between them means anything.
"""

import random

import pytest

from gaming_agents.games.connect_four import ConnectFour, ConnectFourState
from gaming_agents.games.tictactoe import TicTacToeState
from gaming_agents.registry import make_agent, make_game
from gaming_agents.training.evaluate import evaluate
from gaming_agents.training.trainer import train


# -- the contract every game's description must meet ----------------------

def test_every_game_can_describe_a_position(any_game, rng):
    state = any_game.initial_state(rng)
    described = any_game.features(state)
    assert described, "a position must have some description"
    assert all(isinstance(v, float) or isinstance(v, int) for v in described.values())


def test_feature_values_stay_in_a_single_scale(any_game, rng):
    """One learning rate has to suit every feature, so they share a scale."""
    state = any_game.initial_state(rng)
    for _ in range(40):
        if any_game.is_terminal(state):
            break
        for name, value in any_game.features(state).items():
            assert -1.0 <= value <= 1.0, f"{any_game.name}: {name} is off-scale at {value}"
        state, _ = any_game.step(state, rng.choice(any_game.legal_moves(state)), rng)


def test_the_default_description_is_the_lookup_table(rng):
    """Games that describe nothing must behave exactly as they always did."""
    game = make_game("nim")
    state = game.initial_state(rng)
    assert game.features(state) == {f"is:{game.key(state)!r}": 1.0}


def test_two_different_positions_get_two_different_default_keys(rng):
    game = make_game("nim")
    first = game.initial_state(rng)
    second, _ = game.step(first, game.legal_moves(first)[0], rng)
    assert game.features(first) != game.features(second)


# -- Connect Four ---------------------------------------------------------

def test_a_mirrored_board_describes_identically():
    """The saving that comes free: a board and its reflection stop being
    learned twice over."""
    game = ConnectFour()
    left = ConnectFourState(columns=((1, 1, 1), (2, 2), (), (), (), (), ()), player=1, filled=5)
    right = ConnectFourState(columns=((), (), (), (), (), (2, 2), (1, 1, 1)), player=1, filled=5)
    assert game.features(left) == game.features(right)


def test_an_immediate_threat_is_seen():
    game = ConnectFour()
    # X has three stacked in column 0 and it is O's turn: O must block.
    state = ConnectFourState(columns=((1, 1, 1), (2, 2), (), (), (), (), ()), player=1, filled=5)
    described = game.features(state)
    assert described["must_block"] == 1.0
    assert described["win_available"] == 0.0


def test_a_winning_move_of_my_own_is_seen():
    game = ConnectFour()
    state = ConnectFourState(columns=((1, 1, 1), (2, 2), (), (), (), (), ()), player=0, filled=5)
    described = game.features(state)
    assert described["win_available"] == 1.0


def test_a_line_that_cannot_be_reached_yet_is_not_a_threat():
    """Three in a row along the top of a column nobody can reach is a promise,
    not a threat, and the description has to tell the difference."""
    game = ConnectFour()
    # X has three in a row on row 0, with the gap in column 3 -- reachable.
    reachable = ConnectFourState(columns=((1,), (1,), (1,), (), (), (), ()), player=1, filled=3)
    assert game.features(reachable)["must_block"] == 1.0

    # The same three, but the gap sits one row up, so it cannot be played yet.
    stacked = ConnectFourState(
        columns=((2, 1), (2, 1), (2, 1), (), (), (), ()), player=1, filled=6
    )
    described = game.features(stacked)
    assert described["their_near_wins"] > 0, "the line is still worth noticing"
    assert described["must_block"] == 0.0, "but it is not playable this turn"


def test_a_contested_line_counts_for_nobody():
    game = ConnectFour()
    blocked = ConnectFourState(columns=((1,), (1,), (2,), (), (), (), ()), player=0, filled=3)
    described = game.features(blocked)
    assert described["my_near_wins"] == 0.0
    assert described["their_near_wins"] == 0.0


def test_the_description_is_the_same_size_whatever_the_position(rng):
    game = ConnectFour()
    state = game.initial_state(rng)
    names = set(game.features(state))
    for _ in range(20):
        if game.is_terminal(state):
            break
        state, _ = game.step(state, rng.choice(game.legal_moves(state)), rng)
        assert set(game.features(state)) == names


def test_describing_a_board_does_not_depend_on_who_is_asking():
    """The description is from the mover's point of view, so swapping colours
    and swapping turn should give back the same numbers."""
    game = ConnectFour()
    as_x = ConnectFourState(columns=((1, 1), (2,), (), (), (), (), ()), player=0, filled=3)
    as_o = ConnectFourState(columns=((2, 2), (1,), (), (), (), (), ()), player=1, filled=3)
    assert game.features(as_x) == game.features(as_o)


# -- Tic-Tac-Toe ----------------------------------------------------------

def test_all_eight_symmetries_of_a_board_describe_identically():
    """86% of the old lookup table was duplicates of this kind."""
    game = make_game("tictactoe")
    corners = [
        (1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 1),
    ]
    described = {tuple(sorted(game.features(TicTacToeState(cells=c, player=1)).items())) for c in corners}
    assert len(described) == 1, "all four opening corners are the same move"


def test_the_centre_is_described_separately_from_a_corner():
    game = make_game("tictactoe")
    centre = game.features(TicTacToeState(cells=(0, 0, 0, 0, 1, 0, 0, 0, 0), player=1))
    corner = game.features(TicTacToeState(cells=(1, 0, 0, 0, 0, 0, 0, 0, 0), player=1))
    assert centre != corner


def test_a_winning_line_is_visible_to_the_player_who_can_take_it():
    game = make_game("tictactoe")
    state = TicTacToeState(cells=(1, 1, 0, 2, 2, 0, 0, 0, 0), player=0)
    assert game.features(state)["win_available"] == 1.0
    assert game.features(TicTacToeState(cells=state.cells, player=1))["must_block"] == 1.0


# -- the learner ----------------------------------------------------------

def test_a_one_hot_learner_is_a_lookup_table_in_disguise(rng):
    """The control the comparison rests on: with the default description this
    agent has exactly the information the tabular one has."""
    game = make_game("nim")
    agent = make_agent("linear", alpha=0.3, epsilon=0.3)
    train(game, agent, 3_000, rng, eval_episodes=10)
    learned = agent.stats()["weights_learned"]

    table = make_agent("qlearner")
    train(game, table, 3_000, rng, eval_episodes=10)
    # One weight per position, against one table row per position.
    assert abs(learned - table.known_states("nim")) < learned * 0.2


def test_the_feature_learner_beats_random_at_connect_four(rng):
    game = make_game("connect4")
    agent = make_agent("linear", alpha=0.02, epsilon=0.2)
    train(game, agent, 2_000, rng, opponent=make_agent("random"), eval_episodes=10)
    assert evaluate(game, agent, make_agent("random"), 300, rng).win_rate > 0.85


def test_it_learns_far_faster_than_the_table(rng):
    """The whole point: the same games should teach it more."""
    game = make_game("connect4")
    budget = 2_000

    described = make_agent("linear", alpha=0.02, epsilon=0.2)
    train(game, described, budget, random.Random(0), opponent=make_agent("random"), eval_episodes=10)

    memorised = make_agent("qlearner", alpha=0.25, epsilon=0.35)
    train(game, memorised, budget, random.Random(0), opponent=make_agent("random"), eval_episodes=10)

    judge = lambda a: evaluate(game, a, make_agent("random"), 400, random.Random(1)).win_rate  # noqa: E731
    assert judge(described) > judge(memorised) + 0.2


def test_it_keeps_one_set_of_weights_per_game(rng):
    agent = make_agent("linear", alpha=0.1)
    train(make_game("nim"), agent, 200, rng, eval_episodes=5)
    train(make_game("tictactoe"), agent, 200, rng, eval_episodes=5)
    assert set(agent.weights) == {"nim", "tictactoe"}


def test_a_terminal_position_is_worth_nothing_in_itself(rng):
    """Its value is in the reward already paid out, not in the position."""
    game = make_game("tictactoe")
    agent = make_agent("linear")
    state = TicTacToeState(cells=(1, 1, 1, 2, 2, 0, 0, 0, 0), player=1, winner=0)
    assert agent.value(game, state) == 0.0


def test_weights_survive_a_round_trip(tmp_path, rng):
    from gaming_agents.training.persistence import load_agent, save_agent

    agent = make_agent("linear", alpha=0.07, epsilon=0.11)
    train(make_game("connect4"), agent, 300, rng, opponent=make_agent("random"), eval_episodes=5)
    restored = load_agent(save_agent(agent, tmp_path / "linear.json"))

    assert restored.alpha == pytest.approx(0.07)
    assert dict(restored.weights["connect4"]) == dict(agent.weights["connect4"])


def test_what_it_learned_can_be_read_back(rng):
    """A table of 400,000 numbers explains nothing; ten weights can."""
    game = make_game("connect4")
    agent = make_agent("linear", alpha=0.02, epsilon=0.2)
    train(game, agent, 2_000, rng, opponent=make_agent("random"), eval_episodes=10)

    weights = dict(agent.explain("connect4", top=20))
    assert len(weights) <= 10, "the whole point is that there are few enough to read"
    assert "my_centre" in weights and "their_near_wins" in weights


def test_it_values_positions_in_the_right_order(rng):
    """What the agent has understood shows in how it ranks positions, not in
    any single weight: the features overlap, so a linear model is free to
    split the credit between correlated ones however it likes."""
    game = make_game("connect4")
    agent = make_agent("linear", alpha=0.02, epsilon=0.2)
    train(game, agent, 3_000, random.Random(0), opponent=make_agent("random"), eval_episodes=10)

    # X to move, with three in a column and the win there for the taking.
    winning = ConnectFourState(columns=((1, 1, 1), (2, 2), (), (), (), (), ()), player=0, filled=5)
    # The same board with O to move, who must block or lose.
    losing = ConnectFourState(columns=((1, 1, 1), (2, 2), (), (), (), (), ()), player=1, filled=5)
    quiet = ConnectFourState(columns=((1,), (2,), (), (), (), (), ()), player=0, filled=2)

    assert agent.value(game, winning) > agent.value(game, quiet) > agent.value(game, losing)


def test_it_takes_a_win_and_blocks_a_loss(rng):
    """The behaviour the features are there to produce."""
    game = make_game("connect4")
    agent = make_agent("linear", alpha=0.02, epsilon=0.2)
    train(game, agent, 3_000, random.Random(0), opponent=make_agent("random"), eval_episodes=10)
    agent.training = False

    winning = ConnectFourState(columns=((1, 1, 1), (2, 2), (), (), (), (), ()), player=0, filled=5)
    assert agent.select_move(game, winning, game.legal_moves(winning), rng) == 0, "take the win"

    blocking = ConnectFourState(columns=((1, 1, 1), (2, 2), (), (), (), (), ()), player=1, filled=5)
    assert agent.select_move(game, blocking, game.legal_moves(blocking), rng) == 0, "block the loss"
