"""Rules every game has to obey, checked against all of them.

These are the invariants the episode runner, the agents and the search all
assume. A new game is wired up correctly when it passes this file.
"""


import pytest

from gaming_agents.core.episode import play_episode
from gaming_agents.registry import make_agent


def test_initial_state_is_playable(any_game, rng):
    state = any_game.initial_state(rng)
    assert not any_game.is_terminal(state)
    assert any_game.legal_moves(state)
    assert 0 <= any_game.current_player(state) < any_game.num_players


def test_terminal_states_have_no_moves(any_game, rng):
    state = any_game.initial_state(rng)
    while not any_game.is_terminal(state):
        moves = any_game.legal_moves(state)
        assert moves, "non-terminal state must offer a move"
        state, _ = any_game.step(state, rng.choice(moves), rng)
    assert any_game.legal_moves(state) == []


def test_step_does_not_mutate_the_state_it_was_given(any_game, rng):
    state = any_game.initial_state(rng)
    before = any_game.key(state)
    for move in any_game.legal_moves(state):
        any_game.step(state, move, rng)
    assert any_game.key(state) == before


def test_step_returns_one_reward_per_player(any_game, rng):
    state = any_game.initial_state(rng)
    move = any_game.legal_moves(state)[0]
    _, rewards = any_game.step(state, move, rng)
    assert len(rewards) == any_game.num_players
    assert all(isinstance(r, float) for r in rewards)


def test_state_keys_are_hashable_and_renderable(any_game, rng):
    state = any_game.initial_state(rng)
    hash(any_game.key(state))
    assert isinstance(any_game.render(state), str)


def test_move_labels_are_unique(any_game, rng):
    """Agents index tables by label, so two moves must never share one."""
    state = any_game.initial_state(rng)
    labels = [any_game.move_label(m) for m in any_game.legal_moves(state)]
    assert len(set(labels)) == len(labels)


def test_labels_round_trip_through_parse_move(any_game, rng):
    state = any_game.initial_state(rng)
    for move in any_game.legal_moves(state):
        assert any_game.parse_move(any_game.move_label(move), state) == move


def test_illegal_moves_raise(any_game, rng):
    state = any_game.initial_state(rng)
    with pytest.raises(ValueError):
        any_game.parse_move("definitely-not-a-move", state)


def test_two_player_games_are_zero_sum(any_game, rng):
    if any_game.num_players != 2:
        pytest.skip("only meaningful for adversarial games")
    for _ in range(5):
        result = play_episode(any_game, [make_agent("random")] * 2, rng, learn=False)
        assert sum(result.outcome) == pytest.approx(0.0)


def test_random_play_always_terminates(any_game, rng):
    for _ in range(5):
        result = play_episode(
            any_game, [make_agent("random")] * any_game.num_players, rng, learn=False
        )
        assert not result.truncated
        assert any_game.is_terminal(result.final_state)
