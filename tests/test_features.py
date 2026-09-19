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
    assert game.features(state) == {f"{game.name}:is:{game.key(state)!r}": 1.0}


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
    assert described["their_strength"] > 0, "the line is still worth noticing"
    assert described["must_block"] == 0.0, "but it is not playable this turn"


def test_a_contested_line_counts_for_nobody():
    game = ConnectFour()
    blocked = ConnectFourState(columns=((1,), (1,), (2,), (), (), (), ()), player=0, filled=3)
    described = game.features(blocked)
    assert described["my_strength"] == 0.0
    assert described["their_strength"] == 0.0


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
    assert "board:centre_mine" in weights and "their_strength" in weights


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


# -- a vocabulary shared between games ------------------------------------

#: Every game speaks these. Their definitions live on ``Game.UNIVERSAL_FEATURES``.
UNIVERSAL = {"bias", "progress", "my_strength", "their_strength", "win_available", "must_block"}
#: Ideas only board games have, so shared between them but not with the cards.
BOARD = {"board:building_mine", "board:building_theirs", "board:centre_mine", "board:centre_theirs"}


def test_every_game_speaks_the_universal_vocabulary(rng):
    """Transfer is only possible if the same idea has the same name everywhere.
    Card games included -- that is what makes the cross-family question
    answerable at all."""
    for name in ("connect4", "tictactoe", "connect4-mini", "blackjack", "kuhn"):
        game = make_game(name)
        described = set(game.features(game.initial_state(rng)))
        assert UNIVERSAL <= described, f"{name} is missing {UNIVERSAL - described}"


def test_the_universal_vocabulary_is_the_one_the_interface_declares():
    from gaming_agents.core.game import Game

    assert set(Game.UNIVERSAL_FEATURES) == UNIVERSAL


def test_board_games_share_a_vocabulary_the_card_games_do_not(rng):
    for name in ("connect4", "tictactoe", "connect4-mini"):
        assert BOARD <= set(make_game(name).features(make_game(name).initial_state(rng)))
    for name in ("blackjack", "kuhn"):
        described = set(make_game(name).features(make_game(name).initial_state(rng)))
        assert not (BOARD & described), f"{name} should not claim board-only ideas"


@pytest.mark.parametrize("game_name", ["tictactoe", "connect4", "blackjack", "kuhn"])
def test_narrower_ideas_are_namespaced(game_name, rng):
    """Corners mean something in Tic-Tac-Toe and nothing in Blackjack, so they
    must not collide with anything when one set of weights covers every game."""
    game = make_game(game_name)
    for name in set(game.features(game.initial_state(rng))) - UNIVERSAL:
        assert ":" in name, f"{name} is neither universal nor marked as narrower"


def test_games_with_no_description_stay_out_of_each_others_way(rng):
    """Two games' lookup keys could read identically by coincidence; prefixing
    keeps them apart once one set of weights covers everything."""
    for name in ("nim",):
        game = make_game(name)
        assert all(f.startswith(f"{name}:") for f in game.features(game.initial_state(rng)))


def test_every_game_the_learner_is_measured_on_has_a_description(rng):
    """Nim is left on a lookup key deliberately -- it has 266 positions and a
    table handles it perfectly. Everything big enough to need describing is
    described."""
    for name in ("connect4", "connect4-mini", "tictactoe", "blackjack", "kuhn", "2048"):
        game = make_game(name)
        described = set(game.features(game.initial_state(rng)))
        assert not any(f.startswith(f"{name}:is:") for f in described), f"{name} is still a lookup"


def test_sharing_is_off_unless_asked_for(rng):
    agent = make_agent("linear")
    assert agent.shared is False
    train(make_game("nim"), agent, 200, rng, eval_episodes=5)
    train(make_game("tictactoe"), agent, 200, rng, eval_episodes=5)
    assert set(agent.weights) == {"nim", "tictactoe"}


def test_sharing_puts_every_game_in_one_pool(rng):
    agent = make_agent("linear", shared=True)
    train(make_game("nim"), agent, 200, rng, eval_episodes=5)
    train(make_game("tictactoe"), agent, 200, rng, eval_episodes=5)
    assert set(agent.weights) == {agent.SHARED}


def test_what_one_game_teaches_shows_up_in_another(rng):
    """The point of the shared vocabulary: an opinion formed about being one
    move from losing is the same opinion in the next game."""
    agent = make_agent("linear", alpha=0.02, epsilon=0.2, shared=True)
    train(make_game("connect4"), agent, 1_000, random.Random(0),
          opponent=make_agent("random"), eval_episodes=5)

    learned = agent.weights[agent.SHARED]
    assert UNIVERSAL <= set(learned), "Connect Four should have taught the universal concepts"

    # Those weights are what it brings to a game it has never played.
    ttt = make_game("tictactoe")
    fresh = make_agent("linear", shared=True)
    state = ttt.initial_state(rng)
    assert agent.value(ttt, state) != fresh.value(ttt, state)


def test_prior_experience_helps_a_game_never_played(rng):
    """It should be better than an untrained agent at tic-tac-toe on the
    strength of Connect Four alone."""
    ttt = make_game("tictactoe")
    seasoned = make_agent("linear", alpha=0.02, epsilon=0.2, shared=True)
    train(make_game("connect4"), seasoned, 2_000, random.Random(0),
          opponent=make_agent("random"), eval_episodes=5)

    blank = make_agent("linear", shared=True)
    judge = lambda a: evaluate(ttt, a, make_agent("random"), 400, random.Random(77)).win_rate  # noqa: E731
    assert judge(seasoned) > judge(blank) + 0.1


def test_sharing_survives_a_round_trip(tmp_path, rng):
    from gaming_agents.training.persistence import load_agent, save_agent

    agent = make_agent("linear", shared=True, alpha=0.05)
    train(make_game("tictactoe"), agent, 300, rng, eval_episodes=5)
    restored = load_agent(save_agent(agent, tmp_path / "shared.json"))
    assert restored.shared is True
    assert dict(restored.weights[agent.SHARED]) == dict(agent.weights[agent.SHARED])


# -- 2048, where memorising cannot work at all ----------------------------

def test_2048_speaks_the_universal_vocabulary(rng):
    game = make_game("2048")
    assert UNIVERSAL <= set(game.features(game.initial_state(rng)))


def test_a_tidy_board_and_a_fragmented_one_read_differently():
    """Same score, same biggest tile, opposite positions. If the description
    cannot tell these apart it is describing nothing."""
    from gaming_agents.games.twenty_forty_eight import Game2048State

    game = make_game("2048")
    tidy = Game2048State(cells=(1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 0, 0, 0, 0, 0, 0), score=9000)
    fragmented = Game2048State(
        cells=(2, 1024, 4, 256, 512, 8, 128, 16, 4, 64, 2, 32, 8, 4, 16, 2), score=9000
    )
    order, mess = game.features(tidy), game.features(fragmented)

    assert order["progress"] == mess["progress"], "the two boards are equally far along"
    assert order["2048:ordered"] > mess["2048:ordered"]
    assert order["2048:cornered"] > mess["2048:cornered"]
    assert order["my_strength"] > mess["my_strength"], "room to move is the difference"
    assert mess["must_block"] == 1.0, "a full board is one spawn from stuck"


def test_a_cornered_biggest_tile_is_noticed():
    from gaming_agents.games.twenty_forty_eight import Game2048State

    game = make_game("2048")
    corner = Game2048State(cells=(64, 2, 0, 0) + (0,) * 12)
    middle = Game2048State(cells=(0, 0, 0, 0, 0, 64, 2, 0) + (0,) * 8)
    assert game.features(corner)["2048:cornered"] == 1.0
    assert game.features(middle)["2048:cornered"] == 0.0


def test_room_to_move_is_what_my_strength_means():
    from gaming_agents.games.twenty_forty_eight import Game2048State

    game = make_game("2048")
    roomy = game.features(Game2048State(cells=(2, 4) + (0,) * 14))
    cramped = game.features(Game2048State(cells=tuple(range(2, 34, 2))))
    assert roomy["my_strength"] > cramped["my_strength"]
    assert roomy["must_block"] < cramped["must_block"]


def test_progress_tracks_the_biggest_tile():
    from gaming_agents.games.twenty_forty_eight import Game2048State

    game = make_game("2048")
    early = game.features(Game2048State(cells=(2,) + (0,) * 15))
    late = game.features(Game2048State(cells=(1024,) + (0,) * 15))
    assert late["progress"] > early["progress"]
    assert game.features(Game2048State(cells=(2048,) + (0,) * 15))["win_available"] == 1.0


def test_the_description_beats_memorising_on_2048(rng):
    """The claim this game exists to test: a board that never comes round
    twice cannot be learned by looking it up, but can be learned by
    description."""
    from gaming_agents.core.game import Game as BaseGame
    from gaming_agents.games.twenty_forty_eight import TwentyFortyEight

    class Undescribed(TwentyFortyEight):
        features = BaseGame.features

    budget = 120
    described = make_agent("linear", lookahead_samples=2, alpha=0.02, epsilon=0.15)
    train(make_game("2048"), described, budget, random.Random(0), eval_episodes=2)

    memorised = make_agent("linear", lookahead_samples=2, alpha=0.02, epsilon=0.15)
    train(Undescribed(), memorised, budget, random.Random(0), eval_episodes=2)

    judge = lambda a, g: evaluate(g, a, None, 15, random.Random(5)).mean_outcome  # noqa: E731
    assert judge(described, make_game("2048")) > judge(memorised, Undescribed()) * 1.5
