"""The learned value function, and the two things it changed.

The interesting tests here are the ones about *where* a network helps. It is
not a general upgrade: it wins where the hand-written concepts were provably
inadequate and loses where they were good.
"""

import random

import pytest

from gaming_agents.agents.network import ValueNetwork
from gaming_agents.core.game import Game
from gaming_agents.registry import make_agent, make_game
from gaming_agents.training.evaluate import evaluate
from gaming_agents.training.trainer import train


# -- the network ----------------------------------------------------------

def test_it_learns_a_shape_no_straight_line_can_take():
    """The reason for having a hidden layer at all: high in the middle and low
    at both ends is not something weights over one input can say."""
    net = ValueNetwork(1, hidden=12, rng=random.Random(0))
    rng = random.Random(1)
    for _ in range(12_000):
        x = rng.uniform(0.0, 1.0)
        want = 1.0 if 0.35 < x < 0.65 else 0.0
        out, hidden = net.forward([x])
        net.learn([x], hidden, want - out[0], 0.05)

    assert net.value([0.5]) > 0.7
    assert net.value([0.05]) < 0.4
    assert net.value([0.95]) < 0.4


def test_each_head_learns_its_own_answer():
    net = ValueNetwork(2, hidden=10, outputs=2, rng=random.Random(0))
    rng = random.Random(1)
    for _ in range(6_000):
        x = [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
        outs, hidden = net.forward(x)
        net.learn(x, hidden, {0: x[0] - outs[0], 1: -x[1] - outs[1]}, 0.05)

    outs, _ = net.forward([0.9, 0.9])
    assert outs[0] == pytest.approx(0.9, abs=0.15)
    assert outs[1] == pytest.approx(-0.9, abs=0.15)


def test_an_untouched_head_stays_put():
    """In Q-learning only the action taken has a target. Moving the others
    would be inventing evidence about moves nobody played."""
    net = ValueNetwork(3, hidden=8, outputs=3, rng=random.Random(0))
    before, hidden = net.forward([1.0, 0.0, 1.0])
    net.learn([1.0, 0.0, 1.0], hidden, {1: 5.0}, 0.1)
    after, _ = net.forward([1.0, 0.0, 1.0])

    assert after[1] > before[1], "the head with a target should move"
    # The others shift a little through the shared hidden layer, but are not
    # dragged toward a target of their own.
    assert abs(after[0] - before[0]) < abs(after[1] - before[1])


def test_weights_scale_with_the_width_of_the_input():
    """An eighty-four-square board would saturate tanh on the first pass if the
    starting weights did not shrink to match."""
    wide = ValueNetwork(84, hidden=16, rng=random.Random(0))
    assert abs(wide.value([1.0] * 84)) < 3.0


def test_a_network_survives_a_round_trip():
    net = ValueNetwork(4, hidden=6, outputs=2, rng=random.Random(3))
    same = ValueNetwork.from_state_dict(net.state_dict())
    assert same.forward([0.2, 0.4, 0.6, 0.8])[0] == net.forward([0.2, 0.4, 0.6, 0.8])[0]


# -- what a game hands the learner ---------------------------------------

@pytest.mark.parametrize("name", ["blackjack", "tictactoe", "connect4", "2048"])
def test_a_raw_encoding_is_fixed_width_and_in_range(name, rng):
    game = make_game(name)
    state = game.initial_state(rng)
    width = len(game.encode(state))
    for _ in range(30):
        if game.is_terminal(state):
            break
        vector = game.encode(state)
        assert len(vector) == width
        assert all(-1.0 <= value <= 1.0 for value in vector)
        state, _ = game.step(state, rng.choice(game.legal_moves(state)), rng)


def test_a_game_with_no_raw_encoding_falls_back_to_its_concepts(rng):
    game = make_game("kuhn")
    assert type(game).encode is Game.encode
    assert len(game.encode(game.initial_state(rng))) == len(game.encoding_names())


def test_the_blackjack_upcard_gets_a_slot_per_rank(rng):
    """Given one number it could only say 'higher is worse', which is the
    precise reason the linear agent cannot play this game."""
    from gaming_agents.games.blackjack import BlackjackState

    game = make_game("blackjack")
    six = game.encode(BlackjackState(player_total=16, player_ace=False, dealer_upcard=6,
                                     dealer_total=6, dealer_ace=False))
    ten = game.encode(BlackjackState(player_total=16, player_ace=False, dealer_upcard=10,
                                     dealer_total=10, dealer_ace=False))
    assert sum(six[2:]) == 1.0 and sum(ten[2:]) == 1.0
    assert six[2:] != ten[2:], "different upcards must be different inputs"


@pytest.mark.parametrize("name", ["blackjack", "tictactoe", "connect4", "2048", "kuhn"])
def test_declared_actions_cover_everything_ever_offered(name, rng):
    game = make_game(name)
    space = set(game.action_space())
    state = game.initial_state(rng)
    for _ in range(40):
        if game.is_terminal(state):
            break
        assert set(game.legal_moves(state)) <= space
        state, _ = game.step(state, rng.choice(game.legal_moves(state)), rng)


# -- the agent ------------------------------------------------------------

def test_it_refuses_direct_scoring_without_an_action_space(rng):
    game = make_game("nim")
    agent = make_agent("neural", plan=False)
    with pytest.raises(ValueError, match="looking ahead"):
        agent.select_move(game, game.initial_state(rng), game.legal_moves(game.initial_state(rng)), rng)


def test_an_unknown_reading_is_rejected():
    with pytest.raises(ValueError, match="position.*concepts"):
        make_agent("neural", read="tea leaves")


def test_it_learns_tic_tac_toe_from_the_bare_board(rng):
    """No concepts supplied: nine cells mine, nine theirs, and nothing else."""
    game = make_game("tictactoe")
    agent = make_agent("neural", plan=False, read="position", alpha=0.02,
                       epsilon=0.25, epsilon_decay=0.999)
    before = evaluate(game, agent, make_agent("random"), 200, random.Random(77)).win_rate
    train(game, agent, 4_000, random.Random(0), opponent=make_agent("random"), eval_episodes=5)
    after = evaluate(game, agent, make_agent("random"), 200, random.Random(77)).win_rate
    assert after > before + 0.15


def test_scoring_directly_beats_planning_where_planning_is_noisy(rng):
    """Blackjack's "stick" is valued by sampling the dealer's whole hand, and
    that estimate is noisier than the gap it is meant to resolve."""
    game = make_game("blackjack")
    results = {}
    for plan in (True, False):
        agent = make_agent("neural", plan=plan, read="position", alpha=0.01,
                           epsilon=0.25, gamma=1.0, lookahead_samples=8)
        train(game, agent, 40_000, random.Random(0), eval_episodes=5)
        results[plan] = evaluate(game, agent, None, 20_000, random.Random(77)).mean_outcome
    assert results[False] > results[True] + 0.01


def test_a_brain_survives_a_round_trip(tmp_path, rng):
    from gaming_agents.training.persistence import load_agent, save_agent

    game = make_game("blackjack")
    agent = make_agent("neural", plan=False, read="position", alpha=0.02, hidden=8)
    train(game, agent, 500, rng, eval_episodes=3)
    restored = load_agent(save_agent(agent, tmp_path / "net.json"))

    assert restored.plan is False and restored.read == "position"
    state = game.initial_state(random.Random(5))
    assert restored.value(game, state) == pytest.approx(agent.value(game, state))
