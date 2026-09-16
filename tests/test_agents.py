"""Agent behaviour: search that is actually correct, and learning that learns."""


import random

import pytest

from gaming_agents.registry import make_agent, make_game
from gaming_agents.training.evaluate import evaluate
from gaming_agents.training.trainer import train


def test_random_agent_only_plays_legal_moves(any_game, rng):
    agent = make_agent("random")
    state = any_game.initial_state(rng)
    while not any_game.is_terminal(state):
        legal = any_game.legal_moves(state)
        move = agent.select_move(any_game, state, legal, rng)
        assert move in legal
        state, _ = any_game.step(state, move, rng)


# -- minimax --------------------------------------------------------------

def test_minimax_never_loses_at_tictactoe(rng):
    """Tic-Tac-Toe is a draw with correct play, so a loss means a search bug."""
    game = make_game("tictactoe")
    report = evaluate(game, make_agent("minimax"), make_agent("random"), 60, rng)
    assert report.losses == 0


def test_two_perfect_players_always_draw(rng):
    game = make_game("tictactoe")
    report = evaluate(game, make_agent("minimax", name="a"), make_agent("minimax", name="b"), 12, rng)
    assert report.draws == report.episodes


def test_minimax_takes_an_immediate_win(rng):
    from gaming_agents.games.tictactoe import TicTacToeState

    game = make_game("tictactoe")
    state = TicTacToeState(cells=(1, 1, 0, 2, 2, 0, 0, 0, 0), player=0)
    assert make_agent("minimax").select_move(game, state, game.legal_moves(state), rng) == 2


def test_minimax_blocks_an_immediate_loss(rng):
    from gaming_agents.games.tictactoe import TicTacToeState

    game = make_game("tictactoe")
    state = TicTacToeState(cells=(2, 2, 0, 1, 0, 0, 0, 0, 0), player=0)
    assert make_agent("minimax").select_move(game, state, game.legal_moves(state), rng) == 2


def test_minimax_refuses_games_it_cannot_search(rng):
    game = make_game("blackjack")
    state = game.initial_state(rng)
    with pytest.raises(ValueError):
        make_agent("minimax").select_move(game, state, game.legal_moves(state), rng)


def test_perfect_nim_wins_every_winnable_game(rng):
    """(1, 3, 5) has a non-zero nim-sum, so the first player wins outright."""
    from gaming_agents.games.nim import Nim

    game = Nim(heaps=(1, 3, 5))
    report = evaluate(
        game, make_agent("perfect-nim"), make_agent("random"), 100, rng, alternate_seats=False
    )
    assert report.wins == report.episodes


# -- MCTS -----------------------------------------------------------------

def test_mcts_beats_random_at_tictactoe(rng):
    game = make_game("tictactoe")
    report = evaluate(game, make_agent("mcts", simulations=120), make_agent("random"), 40, rng)
    assert report.win_rate > 0.75


def test_mcts_finds_a_win_in_one(rng):
    from gaming_agents.games.tictactoe import TicTacToeState

    game = make_game("tictactoe")
    state = TicTacToeState(cells=(1, 1, 0, 2, 2, 0, 0, 0, 0), player=0)
    agent = make_agent("mcts", simulations=300)
    assert agent.select_move(game, state, game.legal_moves(state), rng) == 2


def test_mcts_handles_a_stochastic_game(rng):
    """2048 resamples every child, so this exercises the open-loop path."""
    game = make_game("2048")
    report = evaluate(game, make_agent("mcts", simulations=15), None, 3, rng)
    assert report.mean_outcome > 0


def test_a_forced_move_needs_no_search(rng):
    from gaming_agents.games.tictactoe import TicTacToeState

    game = make_game("tictactoe")
    state = TicTacToeState(cells=(1, 2, 1, 1, 2, 2, 2, 1, 0), player=0)
    agent = make_agent("mcts", simulations=1)
    assert agent.select_move(game, state, game.legal_moves(state), rng) == 8


# -- the learner ----------------------------------------------------------

def test_the_learner_solves_nim_by_self_play(rng):
    game = make_game("nim")
    agent = make_agent("qlearner", epsilon=0.3, epsilon_decay=0.9995, alpha=0.3)
    before = evaluate(game, agent, make_agent("random"), 200, rng).win_rate
    train(game, agent, 3_000, rng, eval_episodes=50)
    after = evaluate(game, agent, make_agent("random"), 200, rng).win_rate
    assert after > before
    assert after > 0.9


def test_a_trained_learner_draws_with_perfect_play_from_a_won_seat(rng):
    """The (1,3,5,7) opening is lost for whoever moves first, so a learner that
    has really solved Nim wins exactly the half of the games it is dealt."""
    game = make_game("nim")
    agent = make_agent("qlearner", epsilon=0.3, epsilon_decay=0.9995, alpha=0.3)
    train(game, agent, 6_000, rng, eval_episodes=50)
    report = evaluate(game, agent, make_agent("perfect-nim"), 200, rng)
    assert report.win_rate == pytest.approx(0.5, abs=0.02)


def test_learning_stops_when_training_is_off(rng):
    game = make_game("tictactoe")
    agent = make_agent("qlearner")
    train(game, agent, 200, rng, eval_episodes=10)
    updates = agent.updates
    evaluate(game, agent, make_agent("random"), 50, rng)
    assert agent.updates == updates


def test_evaluation_does_not_age_the_exploration_schedule(rng):
    game = make_game("tictactoe")
    agent = make_agent("qlearner", epsilon=0.5, epsilon_decay=0.99)
    train(game, agent, 100, rng, eval_episodes=100)
    assert agent.episodes == 100


def test_self_play_counts_each_game_once(rng):
    game = make_game("tictactoe")
    agent = make_agent("qlearner")
    train(game, agent, 50, rng, eval_episodes=1)
    assert agent.episodes == 50


def test_alternating_seats_counts_each_game_once(rng):
    game = make_game("tictactoe")
    agent = make_agent("qlearner")
    train(game, agent, 50, rng, opponent=make_agent("random"), eval_episodes=1)
    assert agent.episodes == 50


def test_one_learner_keeps_separate_tables_per_game(rng):
    """This is what lets a single agent walk the whole curriculum."""
    agent = make_agent("qlearner")
    train(make_game("nim"), agent, 200, rng, eval_episodes=5)
    train(make_game("tictactoe"), agent, 200, rng, eval_episodes=5)
    assert agent.known_states("nim") > 0
    assert agent.known_states("tictactoe") > 0
    assert set(agent.q) == {"nim", "tictactoe"}


def test_the_learner_improves_at_blackjack(rng):
    game = make_game("blackjack")
    agent = make_agent("qlearner", epsilon=0.25, alpha=0.05, gamma=1.0)
    train(game, agent, 30_000, rng, eval_episodes=10)
    trained = evaluate(game, agent, None, 5_000, rng).mean_outcome
    baseline = evaluate(game, make_agent("random"), None, 5_000, rng).mean_outcome
    assert trained > baseline + 0.2


def test_a_human_agent_reads_moves_from_its_prompt(rng, capsys):
    game = make_game("tictactoe")
    answers = iter(["not a cell", "4"])
    agent = make_agent("human", prompt=lambda _text: next(answers))
    state = game.initial_state(rng)
    assert agent.select_move(game, state, game.legal_moves(state), rng) == 4


def test_a_human_agent_falls_back_when_input_runs_out(rng):
    game = make_game("tictactoe")

    def closed(_text):
        raise EOFError

    agent = make_agent("human", prompt=closed)
    state = game.initial_state(rng)
    assert agent.select_move(game, state, game.legal_moves(state), rng) in game.legal_moves(state)


# -- solved-game reference agents -----------------------------------------

def test_the_blackjack_solver_matches_its_own_theory(rng):
    """The DP is computed analytically; this plays it out and checks they agree."""
    from gaming_agents.games.blackjack import optimal_value

    game = make_game("blackjack")
    report = evaluate(game, make_agent("perfect-blackjack"), None, 40_000, rng)
    assert report.mean_outcome == pytest.approx(optimal_value(), abs=0.015)


def test_the_solver_beats_a_random_player_by_a_wide_margin(rng):
    game = make_game("blackjack")
    solved = evaluate(game, make_agent("perfect-blackjack"), None, 20_000, rng).mean_outcome
    chance = evaluate(game, make_agent("random"), None, 20_000, rng).mean_outcome
    assert solved > chance + 0.25


def test_reference_agents_refuse_games_they_do_not_know(rng):
    game = make_game("tictactoe")
    state = game.initial_state(rng)
    for spec in ("perfect-nim", "perfect-blackjack"):
        with pytest.raises(ValueError):
            make_agent(spec).select_move(game, state, game.legal_moves(state), rng)


# -- learning-rate schedules ----------------------------------------------

def test_a_visits_schedule_reaches_the_solved_optimum_where_a_fixed_step_does_not(rng):
    """The point of alpha_mode: against a fixed environment, a constant step
    size keeps bouncing around the answer however long it trains, and 1/n
    converges to it."""
    from gaming_agents.games.blackjack import optimal_value

    game = make_game("blackjack")
    fixed = make_agent("qlearner", epsilon=0.3, epsilon_decay=1.0, alpha=0.05, gamma=1.0)
    scheduled = make_agent("qlearner", epsilon=0.3, epsilon_decay=1.0, alpha_mode="visits", gamma=1.0)
    for agent in (fixed, scheduled):
        train(game, agent, 150_000, rng, eval_episodes=10)

    judge = lambda a: evaluate(game, a, None, 40_000, random.Random(11)).mean_outcome  # noqa: E731
    assert judge(scheduled) > judge(fixed)
    assert judge(scheduled) == pytest.approx(optimal_value(), abs=0.015)


def test_a_visits_schedule_shrinks_the_step_as_it_learns(rng):
    agent = make_agent("qlearner", alpha_mode="visits")
    train(make_game("nim"), agent, 300, rng, eval_episodes=5)
    counts = [n for table in agent.visits.values() for row in table.values() for n in row.values()]
    assert counts and max(counts) > 1


def test_a_constant_schedule_keeps_no_counts(rng):
    """Counting every state-action pair on a big game would double memory."""
    agent = make_agent("qlearner", alpha_mode="constant")
    train(make_game("nim"), agent, 300, rng, eval_episodes=5)
    assert not agent.visits


def test_an_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="alpha_mode"):
        make_agent("qlearner", alpha_mode="magic")


def test_visit_counts_survive_a_save(tmp_path, rng):
    """Reloading must not restart the 1/n schedule and undo what it settled on."""
    from gaming_agents.training.persistence import load_agent, save_agent

    agent = make_agent("qlearner", alpha_mode="visits")
    train(make_game("nim"), agent, 500, rng, eval_episodes=5)
    restored = load_agent(save_agent(agent, tmp_path / "brain.json"))
    assert restored.alpha_mode == "visits"
    assert restored.visits == agent.visits
