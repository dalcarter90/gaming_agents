"""The command line surface: every subcommand runs and says something useful."""

import pytest

from gaming_agents.cli import main
from gaming_agents.registry import parse_spec


def run(capsys, *argv) -> str:
    """Run a subcommand, assert it succeeded, and return what it printed."""
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def run_err(capsys, *argv) -> str:
    """Same, but return stderr. ``readouterr`` drains the buffer, so a test
    that wants both streams has to ask for them in one go."""
    assert main(list(argv)) == 0
    return capsys.readouterr().err


def test_games_lists_everything_available(capsys):
    out = run(capsys, "games")
    for expected in ("nim", "tictactoe", "connect4", "blackjack", "2048"):
        assert expected in out
    assert "qlearner" in out and "classic" in out


def test_eval_reports_a_result(capsys):
    out = run(capsys, "eval", "--game", "tictactoe", "--agent", "minimax",
              "--opponent", "random", "--episodes", "6")
    assert "Tic-Tac-Toe" in out
    assert "mean outcome" in out


def test_eval_works_on_a_solo_game(capsys):
    out = run(capsys, "eval", "--game", "blackjack", "--agent", "random", "--episodes", "50")
    assert "solo" in out


def test_train_runs_and_saves(capsys, tmp_path):
    brain = tmp_path / "brain.json"
    out = run(capsys, "train", "--game", "nim", "--episodes", "500",
              "--eval-episodes", "50", "--eval-every", "250", "--save", str(brain))
    assert "training qlearner" in out
    assert brain.exists()
    assert "saved brain" in out


def test_train_can_warm_start_from_a_saved_brain(capsys, tmp_path):
    brain = tmp_path / "brain.json"
    run(capsys, "train", "--game", "nim", "--episodes", "400",
        "--eval-episodes", "20", "--save", str(brain))
    out = run(capsys, "train", "--game", "nim", "--episodes", "100",
              "--eval-episodes", "20", "--load", str(brain))
    assert "'episodes': 500" in out


def test_a_missing_brain_is_a_warning_not_a_crash(capsys, tmp_path):
    err = run_err(capsys, "train", "--game", "nim", "--episodes", "50",
                  "--eval-episodes", "10", "--load", str(tmp_path / "absent.json"))
    assert "no brain at" in err


def test_inspect_summarises_a_brain(capsys, tmp_path):
    brain = tmp_path / "brain.json"
    run(capsys, "train", "--game", "nim", "--episodes", "300",
        "--eval-episodes", "20", "--save", str(brain))
    out = run(capsys, "inspect", str(brain))
    assert "games learned: nim" in out


def test_watch_prints_a_whole_game(capsys):
    out = run(capsys, "watch", "--game", "tictactoe", "--agent", "random", "--opponent", "random")
    assert "=== final ===" in out
    assert "plays" in out


def test_watch_handles_a_solo_game(capsys):
    out = run(capsys, "watch", "--game", "blackjack", "--agent", "random")
    assert "=== final ===" in out


def test_tournament_ranks_the_field(capsys):
    out = run(capsys, "tournament", "--game", "nim", "--episodes", "6",
              "--agents", "random", "perfect-nim")
    assert "points" in out
    assert out.index("perfect-nim") < out.index("random"), "the stronger agent should rank first"


def test_a_tournament_rejects_duplicate_entrants(capsys):
    with pytest.raises(SystemExit):
        main(["tournament", "--game", "nim", "--episodes", "2", "--agents", "random", "random"])


def test_curriculum_runs_and_reports(capsys, tmp_path):
    brain = tmp_path / "brain.json"
    code = main(["curriculum", "--curriculum", "classic", "--scale", "0.002",
                 "--no-gate", "--save", str(brain)])
    out = capsys.readouterr().out
    assert "stage 1/4" in out
    assert "total episodes played" in out
    assert brain.exists()
    assert code in (0, 1)


def test_an_unknown_game_is_a_clean_error(capsys):
    assert main(["eval", "--game", "chess", "--episodes", "1"]) == 2
    assert "unknown game" in capsys.readouterr().err


def test_an_unknown_agent_is_a_clean_error(capsys):
    assert main(["eval", "--game", "nim", "--agent", "alphazero", "--episodes", "1"]) == 2
    assert "unknown agent" in capsys.readouterr().err


def test_the_same_seed_gives_the_same_game(capsys):
    first = run(capsys, "--seed", "42", "watch", "--game", "tictactoe",
                "--agent", "random", "--opponent", "random")
    second = run(capsys, "--seed", "42", "watch", "--game", "tictactoe",
                 "--agent", "random", "--opponent", "random")
    assert first == second


# -- agent strength specs -------------------------------------------------

def test_a_bare_agent_name_has_no_tuning():
    assert parse_spec("mcts") == ("mcts", {})


def test_a_spec_sets_the_agents_one_strength_knob():
    assert parse_spec("mcts:800") == ("mcts", {"simulations": 800})
    assert parse_spec("minimax:6") == ("minimax", {"depth": 6})


@pytest.mark.parametrize("spec", ["random:5", "mcts:lots", "mcts:0"])
def test_nonsense_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        parse_spec(spec)


def test_a_spec_reaches_the_agent_it_names(capsys):
    out = run(capsys, "eval", "--game", "tictactoe", "--agent", "mcts:30",
              "--opponent", "minimax:2", "--episodes", "4")
    assert "Tic-Tac-Toe" in out


def test_a_bad_spec_is_a_clean_error(capsys):
    assert main(["eval", "--game", "nim", "--agent", "random:9"]) == 2
    assert "no strength setting" in capsys.readouterr().err
