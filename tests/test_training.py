"""The training loop, the evaluator, the curriculum and saved brains."""

import json
import random

import pytest

from gaming_agents.core.episode import play_episode
from gaming_agents.registry import make_agent, make_game
from gaming_agents.training.curriculum import NamedCurriculum, Stage, get_curriculum, run_curriculum
from gaming_agents.training.evaluate import evaluate
from gaming_agents.training.persistence import load_agent, peek, save_agent
from gaming_agents.training.trainer import tournament, train


# -- the episode runner ---------------------------------------------------

class Recorder(make_agent("random").__class__):
    """A random agent that keeps every transition it is handed."""

    def __init__(self, name=None):
        super().__init__(name)
        self.seen = []

    def learn(self, game, transition):
        self.seen.append(transition)


def test_transitions_span_the_opponents_reply(rng):
    """A seat's next_state must be its own next turn, not the position right
    after its own move -- otherwise TD updates read the wrong side's board."""
    game = make_game("tictactoe")
    left = Recorder("left")
    play_episode(game, [left, make_agent("random")], rng, learn=True)
    for transition in left.seen:
        if transition.next_state is not None:
            assert game.current_player(transition.next_state) == 0


def test_the_last_transition_of_an_episode_is_marked_done(rng):
    game = make_game("tictactoe")
    left = Recorder("left")
    play_episode(game, [left, make_agent("random")], rng, learn=True)
    assert left.seen[-1].done
    assert left.seen[-1].next_state is None
    assert not any(t.done for t in left.seen[:-1])


def test_terminal_reward_reaches_the_seat_that_earned_it(rng):
    game = make_game("nim")
    left, right = Recorder("left"), Recorder("right")
    result = play_episode(game, [left, right], rng, learn=True)
    winner = result.winner
    assert winner is not None
    rewards = {0: left.seen[-1].reward, 1: right.seen[-1].reward}
    assert rewards[winner] == 1.0
    assert rewards[1 - winner] == -1.0


def test_an_agent_playing_itself_gets_both_seats_transitions(rng):
    game = make_game("tictactoe")
    agent = Recorder("both")
    play_episode(game, [agent, agent], rng, learn=True)
    assert {t.seat for t in agent.seen} == {0, 1}


def test_nothing_is_learned_when_learning_is_off(rng):
    game = make_game("tictactoe")
    agent = Recorder("quiet")
    play_episode(game, [agent, make_agent("random")], rng, learn=False)
    assert agent.seen == []


def test_a_recorded_episode_keeps_a_transcript(rng):
    game = make_game("tictactoe")
    result = play_episode(game, [make_agent("random")] * 2, rng, learn=False, record=True)
    assert len(result.transcript) == result.turns
    assert all(isinstance(turn.board, str) for turn in result.transcript)


def test_the_wrong_number_of_agents_is_rejected(rng):
    game = make_game("tictactoe")
    with pytest.raises(ValueError):
        play_episode(game, [make_agent("random")], rng)


def test_an_illegal_move_is_caught(rng):
    class Cheater(make_agent("random").__class__):
        def select_move(self, game, state, legal_moves, rng):
            return "no such move"

    game = make_game("tictactoe")
    with pytest.raises(ValueError, match="illegal move"):
        play_episode(game, [Cheater(), make_agent("random")], rng)


# -- evaluation -----------------------------------------------------------

def test_evaluation_alternates_who_moves_first(rng):
    """First-move advantage in Tic-Tac-Toe is large, so a lopsided result here
    would mean the seats are not really being swapped."""
    game = make_game("tictactoe")
    report = evaluate(game, make_agent("random", name="a"), make_agent("random", name="b"), 400, rng)
    assert 0.35 < report.win_rate < 0.65


def test_evaluation_restores_training_flags(rng):
    game = make_game("tictactoe")
    agent = make_agent("qlearner")
    agent.training = True
    evaluate(game, agent, make_agent("random"), 5, rng)
    assert agent.training is True


def test_a_two_player_game_needs_an_opponent(rng):
    with pytest.raises(ValueError):
        evaluate(make_game("tictactoe"), make_agent("random"), None, 5, rng)


def test_win_draw_and_loss_counts_add_up(rng):
    game = make_game("tictactoe")
    report = evaluate(game, make_agent("random", name="a"), make_agent("random", name="b"), 50, rng)
    assert report.wins + report.draws + report.losses == report.episodes
    assert report.non_loss_rate >= report.win_rate


def test_unknown_metrics_are_rejected(rng):
    report = evaluate(make_game("blackjack"), make_agent("random"), None, 5, rng)
    with pytest.raises(ValueError):
        report.metric("vibes")


# -- training and tournaments ---------------------------------------------

def test_training_records_a_learning_curve(rng):
    game = make_game("nim")
    report = train(game, make_agent("qlearner"), 400, rng, eval_every=100, eval_episodes=20)
    assert [episodes for episodes, _ in report.history] == [100, 200, 300, 400]


def test_a_tournament_ranks_the_stronger_agent_first(rng):
    game = make_game("tictactoe")
    table = tournament(
        game, [make_agent("random"), make_agent("minimax"), make_agent("first")], 10, rng
    )
    assert table[0][0] == "minimax"
    assert [row[0] for row in table][-1] in {"random", "first"}


def test_tournaments_need_two_players(rng):
    with pytest.raises(ValueError):
        tournament(make_game("2048"), [make_agent("random")], 2, rng)


# -- persistence ----------------------------------------------------------

def test_a_brain_survives_a_round_trip(tmp_path, rng):
    game = make_game("nim")
    agent = make_agent("qlearner", epsilon=0.11, alpha=0.33)
    train(game, agent, 1_000, rng, eval_episodes=20)
    path = save_agent(agent, tmp_path / "brain.json")

    restored = load_agent(path)
    assert restored.q == agent.q
    assert restored.epsilon == pytest.approx(0.11)
    assert restored.alpha == pytest.approx(0.33)
    assert restored.episodes == agent.episodes


def test_a_restored_brain_plays_the_same_way(tmp_path, rng):
    game = make_game("nim")
    agent = make_agent("qlearner", epsilon=0.3, alpha=0.3)
    train(game, agent, 2_000, rng, eval_episodes=20)
    restored = load_agent(save_agent(agent, tmp_path / "brain.json"))

    before = evaluate(game, agent, make_agent("perfect-nim"), 100, random.Random(9)).win_rate
    after = evaluate(game, restored, make_agent("perfect-nim"), 100, random.Random(9)).win_rate
    assert before == after


def test_a_brain_can_be_loaded_into_an_existing_agent(tmp_path, rng):
    agent = make_agent("qlearner")
    train(make_game("nim"), agent, 500, rng, eval_episodes=10)
    path = save_agent(agent, tmp_path / "brain.json")

    fresh = make_agent("qlearner")
    load_agent(path, into=fresh)
    assert fresh.known_states("nim") == agent.known_states("nim")


def test_a_brain_is_readable_json(tmp_path, rng):
    agent = make_agent("qlearner")
    train(make_game("nim"), agent, 200, rng, eval_episodes=5)
    path = save_agent(agent, tmp_path / "brain.json")
    payload = json.loads(path.read_text())
    assert payload["kind"] == "qlearner"
    assert "nim" in payload["agent"]["q"]


def test_peek_summarises_without_loading(tmp_path, rng):
    agent = make_agent("qlearner")
    train(make_game("nim"), agent, 200, rng, eval_episodes=5)
    info = peek(save_agent(agent, tmp_path / "brain.json"))
    assert info["games"] == ["nim"]
    assert info["kind"] == "qlearner"


def test_a_brain_from_the_future_is_refused(tmp_path):
    path = tmp_path / "brain.json"
    path.write_text(json.dumps({"format": 999, "agent": {}}))
    with pytest.raises(ValueError, match="format"):
        load_agent(path)


def test_saving_leaves_no_temporary_file_behind(tmp_path, rng):
    agent = make_agent("qlearner")
    save_agent(agent, tmp_path / "brain.json")
    assert [p.name for p in tmp_path.iterdir()] == ["brain.json"]


# -- the curriculum -------------------------------------------------------

TINY = NamedCurriculum(
    name="tiny",
    description="A two-rung ladder small enough to run in a test.",
    stages=(
        Stage(
            game="nim",
            goal="beat random at Nim",
            metric="win_rate",
            mastery=0.85,
            episodes_per_round=2_000,
            eval_episodes=100,
            max_rounds=3,
            tune={"epsilon": 0.3, "alpha": 0.3},
        ),
        Stage(
            game="tictactoe",
            goal="beat random at Tic-Tac-Toe",
            metric="win_rate",
            mastery=0.60,
            episodes_per_round=5_000,
            eval_episodes=100,
            max_rounds=3,
            tune={"epsilon": 0.3, "alpha": 0.2},
        ),
    ),
)


def test_a_learner_climbs_a_small_ladder(rng):
    agent = make_agent("qlearner")
    result = run_curriculum(TINY, agent, rng, report=None)
    assert result.completed
    assert [s.stage.game for s in result.stages] == ["nim", "tictactoe"]
    assert agent.known_states("nim") > 0
    assert agent.known_states("tictactoe") > 0


def test_an_unreachable_bar_stops_the_climb(rng):
    impossible = NamedCurriculum(
        name="impossible",
        description="A bar nothing can clear.",
        stages=(
            Stage(
                game="nim",
                goal="win every single game against random, from both seats",
                metric="win_rate",
                mastery=1.01,
                episodes_per_round=200,
                eval_episodes=50,
                max_rounds=2,
            ),
        ) + TINY.stages[1:],
    )
    result = run_curriculum(impossible, make_agent("qlearner"), rng, report=None)
    assert not result.completed
    assert len(result.stages) == 1, "gating should stop before the second rung"
    assert result.stages[0].rounds == 2


def test_gating_can_be_turned_off(rng):
    impossible = NamedCurriculum(
        name="ungated",
        description="Keeps going regardless.",
        stages=(
            Stage(
                game="nim",
                goal="an impossible bar",
                metric="win_rate",
                mastery=1.01,
                episodes_per_round=200,
                eval_episodes=50,
                max_rounds=1,
            ),
        ) + TINY.stages[1:],
    )
    result = run_curriculum(impossible, make_agent("qlearner"), rng, gate=False, report=None)
    assert len(result.stages) == 2


def test_scale_shrinks_every_episode_budget(rng):
    result = run_curriculum(TINY, make_agent("qlearner"), rng, scale=0.01, gate=False, report=None)
    assert result.stages[0].episodes <= 3 * 20


def test_stage_settings_that_mean_nothing_to_an_agent_are_skipped(rng):
    """MCTS has no epsilon, but it should still be able to walk the ladder."""
    ladder = NamedCurriculum(
        name="search",
        description="One rung, walked by a searcher.",
        stages=(
            Stage(
                game="nim",
                goal="beat random",
                metric="win_rate",
                mastery=0.8,
                episodes_per_round=10,
                eval_episodes=40,
                max_rounds=1,
                tune={"epsilon": 0.5, "alpha": 0.1},
            ),
        ),
    )
    result = run_curriculum(ladder, make_agent("mcts", simulations=30), rng, report=None)
    assert result.completed


def test_the_shipped_curricula_are_well_formed():
    for name in ("classic", "solo", "full"):
        ladder = get_curriculum(name)
        assert ladder.stages
        for stage in ladder.stages:
            game = make_game(stage.game)
            assert stage.goal
            assert stage.metric in {"win_rate", "non_loss_rate", "mean_outcome"}
            if game.num_players == 1:
                assert stage.metric == "mean_outcome", f"{stage.game}: solo games have no win rate"


def test_the_ladder_runs_from_easiest_to_hardest():
    stages = get_curriculum("full").stages
    rungs = [make_game(s.game).difficulty for s in stages]
    assert rungs == sorted(rungs), f"curriculum is out of order: {rungs}"


def test_unknown_curricula_are_rejected():
    with pytest.raises(ValueError):
        get_curriculum("nope")
