"""Kuhn poker, and the hidden-information machinery it forced into the project.

The tests that matter most here are the ones checking that agents *cannot* see
what they should not. A poker result measured against an agent that peeks is
not a weak result, it is a meaningless one.
"""

import random

import pytest

from gaming_agents.games.kuhn_poker import (
    BET,
    CALL,
    CHECK,
    FOLD,
    JACK,
    KING,
    QUEEN,
    KuhnPoker,
    KuhnState,
)
from gaming_agents.registry import make_agent, make_game
from gaming_agents.training.evaluate import evaluate
from gaming_agents.training.exploitability import (
    best_response_value,
    exploitability,
    information_sets,
    strategy_of,
)

#: The value of Kuhn poker to the player who acts first, under correct play.
KUHN_VALUE = -1 / 18


# -- the rules ------------------------------------------------------------

def test_the_hand_ends_on_every_terminal_betting_line(rng):
    game = KuhnPoker()
    for history in [
        (CHECK, CHECK),
        (BET, FOLD),
        (BET, CALL),
        (CHECK, BET, FOLD),
        (CHECK, BET, CALL),
    ]:
        assert game.is_terminal(KuhnState(cards=(KING, JACK), history=history))


def test_betting_lines_that_are_still_live(rng):
    game = KuhnPoker()
    for history in [(), (CHECK,), (BET,)]:
        assert not game.is_terminal(KuhnState(cards=(KING, JACK), history=history))


def test_facing_a_bet_you_may_only_fold_or_call(rng):
    game = KuhnPoker()
    state = KuhnState(cards=(JACK, KING), history=(BET,))
    assert game.legal_moves(state) == [FOLD, CALL]


def test_with_no_bet_outstanding_you_may_check_or_bet(rng):
    game = KuhnPoker()
    assert game.legal_moves(KuhnState(cards=(JACK, KING))) == [CHECK, BET]


@pytest.mark.parametrize(
    "cards,history,expected",
    [
        ((KING, JACK), (CHECK, CHECK), (1.0, -1.0)),    # showdown for the antes
        ((JACK, KING), (CHECK, CHECK), (-1.0, 1.0)),
        ((JACK, KING), (BET, FOLD), (1.0, -1.0)),       # a bluff that worked
        ((KING, JACK), (BET, CALL), (2.0, -2.0)),       # showdown after a call
        ((JACK, KING), (BET, CALL), (-2.0, 2.0)),
        ((KING, JACK), (CHECK, BET, FOLD), (-1.0, 1.0)),
        ((KING, JACK), (CHECK, BET, CALL), (2.0, -2.0)),
    ],
)
def test_payoffs(cards, history, expected):
    assert KuhnPoker().outcome(KuhnState(cards=cards, history=history)) == expected


def test_the_pot_grows_with_the_betting():
    game = KuhnPoker()
    assert game.pot(KuhnState(cards=(JACK, KING))) == 2
    assert game.pot(KuhnState(cards=(JACK, KING), history=(BET,))) == 3
    assert game.pot(KuhnState(cards=(JACK, KING), history=(BET, CALL))) == 4


def test_the_two_seats_alternate():
    game = KuhnPoker()
    assert game.current_player(KuhnState(cards=(JACK, KING))) == 0
    assert game.current_player(KuhnState(cards=(JACK, KING), history=(CHECK,))) == 1
    assert game.current_player(KuhnState(cards=(JACK, KING), history=(CHECK, BET))) == 0


def test_all_six_deals_are_enumerated():
    deals = KuhnPoker().initial_outcomes()
    assert len(deals) == 6
    assert sum(probability for _, probability in deals) == pytest.approx(1.0)
    assert len({state.cards for state, _ in deals}) == 6
    assert all(a != b for (state, _) in deals for a, b in [state.cards])


# -- hidden information ---------------------------------------------------

def test_the_state_key_hides_the_opponents_card():
    """The single most important test in this file: if this fails, every other
    poker number in the project is measuring an agent that can see through
    the backs of the cards."""
    game = KuhnPoker()
    against_jack = KuhnState(cards=(QUEEN, JACK), history=(CHECK,))
    against_king = KuhnState(cards=(QUEEN, KING), history=(CHECK,))
    # Seat 1 is to move in both, holding different cards -- so these differ.
    assert game.key(against_jack) != game.key(against_king)
    # But from seat 0's side of the table, holding a queen, they are the same.
    seat0 = KuhnState(cards=(QUEEN, JACK)), KuhnState(cards=(QUEEN, KING))
    assert game.key(seat0[0]) == game.key(seat0[1])


def test_redealing_keeps_your_card_and_changes_theirs(rng):
    game = KuhnPoker()
    state = KuhnState(cards=(QUEEN, JACK), history=(CHECK,))
    seen = set()
    for _ in range(50):
        resampled = game.redeal(state, 0, rng)
        assert resampled.cards[0] == QUEEN, "your own card must survive a redeal"
        assert resampled.cards[1] != QUEEN, "the deck holds one of each card"
        assert resampled.history == state.history
        seen.add(resampled.cards[1])
    assert seen == {JACK, KING}, "both consistent worlds should come up"


def test_redealing_from_the_other_seat_preserves_that_seats_card(rng):
    game = KuhnPoker()
    state = KuhnState(cards=(QUEEN, JACK), history=(CHECK,))
    for _ in range(20):
        resampled = game.redeal(state, 1, rng)
        assert resampled.cards[1] == JACK


def test_perfect_information_games_have_nothing_to_redeal(rng):
    game = make_game("tictactoe")
    state = game.initial_state(rng)
    assert game.redeal(state, 0, rng) is state


def test_search_does_not_read_the_opponents_hand():
    """MCTS searches sampled worlds, so its play must not vary with a card it
    is not allowed to see."""
    game = make_game("kuhn")
    agent = make_agent("mcts:150")

    def opening_bet_rate(their_card):
        bets = 0
        for trial in range(120):
            state = KuhnState(cards=(QUEEN, their_card))
            bets += agent.select_move(game, state, game.legal_moves(state), random.Random(trial)) == BET
        return bets / 120

    assert opening_bet_rate(JACK) == opening_bet_rate(KING)


def test_minimax_refuses_poker_rather_than_cheating_at_it(rng):
    game = make_game("kuhn")
    state = game.initial_state(rng)
    with pytest.raises(ValueError, match="hidden information"):
        make_agent("minimax").select_move(game, state, game.legal_moves(state), rng)


# -- exploitability -------------------------------------------------------

def test_each_seat_has_six_information_sets():
    game = make_game("kuhn")
    for seat in (0, 1):
        spots = information_sets(game, seat)
        assert len(spots) == 6
        # Three cards times two betting lines the seat can face.
        assert all(len(spot.worlds) == 2 for spot in spots.values())
        assert all(len(spot.moves) == 2 for spot in spots.values())


def test_the_known_equilibrium_is_unexploitable(rng):
    """Kuhn's equilibrium is a one-parameter family, and every member of it
    should measure as exactly unexploitable. This validates the measurement
    against theory as much as it validates the agent."""
    game = make_game("kuhn")
    for alpha in (0.0, 1 / 6, 1 / 3):
        report = exploitability(game, make_agent("nash-kuhn", alpha=alpha), rng)
        assert report.exploitability == pytest.approx(0.0, abs=1e-9)
        assert report.best_response_values[0] == pytest.approx(KUHN_VALUE, abs=1e-9)
        assert report.best_response_values[1] == pytest.approx(-KUHN_VALUE, abs=1e-9)


def test_the_equilibrium_holds_its_value_when_actually_played(rng):
    game = make_game("kuhn")
    report = evaluate(
        game, make_agent("nash-kuhn", name="a"), make_agent("nash-kuhn", name="b"),
        200_000, rng, alternate_seats=False,
    )
    assert report.mean_outcome == pytest.approx(KUHN_VALUE, abs=0.01)


def test_alpha_outside_the_equilibrium_family_is_rejected():
    for alpha in (-0.01, 0.5, 1.0):
        with pytest.raises(ValueError, match="equilibrium family"):
            make_agent("nash-kuhn", alpha=alpha)


def test_a_predictable_strategy_is_punished(rng):
    """An agent that always checks and always folds cannot be beaten by luck,
    only by being read -- which is exactly what exploitability measures."""
    game = make_game("kuhn")
    always_passive = {
        key: {CHECK: 1.0, FOLD: 1.0} for key in information_sets(game, 1)
    }
    value = best_response_value(game, 0, always_passive)
    assert value > 0.4, "folding every hand should be worth a fortune to the opponent"


def test_random_play_is_wildly_exploitable(rng):
    game = make_game("kuhn")
    report = exploitability(game, make_agent("random"), rng, samples=3000)
    assert report.exploitability > 0.3


def test_exploitability_needs_two_players(rng):
    with pytest.raises(ValueError, match="two-player"):
        exploitability(make_game("blackjack"), make_agent("random"), rng)


def test_an_exact_strategy_is_read_rather_than_sampled(rng):
    """Sampling would put a noise floor under a measurement meant to hit zero."""
    game = make_game("kuhn")
    strategy = strategy_of(game, make_agent("nash-kuhn", alpha=1 / 3), 0, rng, samples=1)
    bluff = next(mix for key, mix in strategy.items() if key == (JACK, ()))
    assert bluff[BET] == pytest.approx(1 / 3)


# -- CFR ------------------------------------------------------------------

def test_cfr_converges_towards_equilibrium(rng):
    game = make_game("kuhn")
    agent = make_agent("cfr")

    agent.solve(game, 20)
    early = exploitability(game, agent, rng).exploitability
    agent.solve(game, 5_000)
    late = exploitability(game, agent, rng).exploitability

    assert late < early / 10
    assert late < 0.01


def test_cfr_finds_the_value_of_the_game(rng):
    game = make_game("kuhn")
    agent = make_agent("cfr")
    agent.solve(game, 5_000)
    report = exploitability(game, agent, rng)
    assert report.best_response_values[0] == pytest.approx(KUHN_VALUE, abs=0.01)


def test_cfr_learns_to_bluff(rng):
    """The tell that it has understood the game: betting a Jack, the worst
    card, which no value-maximising-per-hand method would ever choose."""
    game = make_game("kuhn")
    agent = make_agent("cfr")
    agent.solve(game, 5_000)
    state = KuhnState(cards=(JACK, QUEEN))
    mix = agent.action_probabilities(game, state, game.legal_moves(state))
    assert mix[BET] > 0.05, "an equilibrium bluffs a Jack sometimes"


def test_cfr_plays_a_mixed_strategy(rng):
    game = make_game("kuhn")
    agent = make_agent("cfr")
    agent.solve(game, 2_000)
    mixed = 0
    for seat in (0, 1):
        for key, spot in information_sets(game, seat).items():
            mix = agent.average_strategy(key, [game.move_label(m) for m in spot.moves])
            if all(0.02 < p < 0.98 for p in mix.values()):
                mixed += 1
    assert mixed >= 2, "a readable poker strategy is a losing one"


def test_cfr_trains_through_the_ordinary_trainer(rng):
    """CFR learns by walking the tree, but it still has to slot into the same
    trainer as everything else."""
    from gaming_agents.training.trainer import train

    game = make_game("kuhn")
    agent = make_agent("cfr")
    train(game, agent, 200, rng, eval_episodes=5)
    assert agent.iterations == 200


def test_a_cfr_brain_says_why_it_cannot_be_saved(tmp_path, rng):
    from gaming_agents.training.persistence import load_agent, save_agent

    agent = make_agent("cfr")
    agent.solve(make_game("kuhn"), 10)
    path = save_agent(agent, tmp_path / "cfr.json")
    with pytest.raises(NotImplementedError, match="re-solve"):
        load_agent(path)
