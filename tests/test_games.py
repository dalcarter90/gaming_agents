"""Game-specific rules: win detection, scoring, and the odd corner case."""

import random

import pytest

from gaming_agents.games.blackjack import HIT, STICK, Blackjack, _add
from gaming_agents.games.connect_four import ConnectFour, ConnectFourState
from gaming_agents.games.nim import Nim, NimState
from gaming_agents.games.tictactoe import TicTacToe, TicTacToeState
from gaming_agents.games.twenty_forty_eight import LEFT, RIGHT, UP, TwentyFortyEight, _collapse, _slide


# -- Nim ------------------------------------------------------------------

def test_nim_sum_identifies_losing_positions():
    assert Nim.nim_sum((1, 3, 5, 7)) == 0  # first player loses with correct play
    assert Nim.nim_sum((1, 3, 5)) == 7


def test_nim_folds_identical_heaps_into_one_move():
    game = Nim(heaps=(3, 3, 3))
    moves = game.legal_moves(game.initial_state(random.Random(0)))
    assert moves == [(3, 1), (3, 2), (3, 3)]


def test_taking_the_last_object_wins(rng):
    game = Nim(heaps=(2,))
    state = NimState(heaps=(1,), player=0)
    state, rewards = game.step(state, (1, 1), rng)
    assert game.is_terminal(state)
    assert rewards == (1.0, -1.0)
    assert game.outcome(state) == (1.0, -1.0)


def test_optimal_moves_leave_a_zero_nim_sum(rng):
    game = Nim(heaps=(1, 3, 5))
    winning = game.optimal_moves(NimState(heaps=(1, 3, 5), player=0))
    assert winning
    for heap_size, count in winning:
        heaps = list((1, 3, 5))
        heaps[heaps.index(heap_size)] -= count
        assert Nim.nim_sum(tuple(heaps)) == 0


def test_lost_positions_offer_no_optimal_move():
    game = Nim(heaps=(1, 3, 5, 7))
    assert game.optimal_moves(NimState(heaps=(1, 3, 5, 7), player=0)) == []


# -- Tic-Tac-Toe ----------------------------------------------------------

@pytest.mark.parametrize(
    "cells,winner",
    [
        ((1, 1, 1, 0, 0, 0, 0, 0, 0), 0),
        ((0, 0, 0, 2, 2, 2, 0, 0, 0), 1),
        ((1, 0, 0, 1, 0, 0, 1, 0, 0), 0),
        ((1, 0, 0, 0, 1, 0, 0, 0, 1), 0),
        ((0, 0, 2, 0, 2, 0, 2, 0, 0), 1),
        ((1, 2, 1, 2, 1, 2, 2, 1, 2), None),
    ],
)
def test_tictactoe_detects_lines(cells, winner):
    from gaming_agents.games.tictactoe import _winner

    assert _winner(cells) == winner


def test_playing_an_occupied_cell_is_rejected(rng):
    game = TicTacToe()
    state = game.initial_state(rng)
    state, _ = game.step(state, 4, rng)
    with pytest.raises(ValueError):
        game.step(state, 4, rng)


def test_a_full_board_with_no_line_is_a_draw():
    game = TicTacToe()
    state = TicTacToeState(cells=(1, 2, 1, 1, 2, 2, 2, 1, 1), player=0)
    assert game.is_terminal(state)
    assert game.outcome(state) == (0.0, 0.0)


# -- Connect Four ---------------------------------------------------------

def test_vertical_win(rng):
    game = ConnectFour()
    state = game.initial_state(rng)
    for _ in range(3):
        state, _ = game.step(state, 0, rng)  # X
        state, _ = game.step(state, 1, rng)  # O
    state, rewards = game.step(state, 0, rng)
    assert state.winner == 0
    assert rewards == (1.0, -1.0)


def test_horizontal_win(rng):
    game = ConnectFour()
    state = game.initial_state(rng)
    for column in range(3):
        state, _ = game.step(state, column, rng)
        state, _ = game.step(state, column, rng)  # O stacks on top, out of the way
    state, rewards = game.step(state, 3, rng)
    assert state.winner == 0


def test_diagonal_win(rng):
    game = ConnectFour()
    #      col0 col1 col2 col3
    # X on (0,0) (1,1) (2,2) (3,3)
    columns = (
        (1,),
        (2, 1),
        (2, 2, 1),
        (2, 2, 2),
    ) + ((),) * 3
    state = ConnectFourState(columns=columns, player=0, filled=9)
    state, rewards = game.step(state, 3, rng)
    assert state.winner == 0
    assert game.is_terminal(state)


def test_a_full_column_is_not_playable(rng):
    game = ConnectFour(rows=2, cols=2, connect=2, name="tiny")
    state = ConnectFourState(columns=((1, 2), ()), player=0, filled=2)
    assert 0 not in game.legal_moves(state)


def test_mirrored_boards_share_a_key(rng):
    game = ConnectFour()
    left = ConnectFourState(columns=((1,),) + ((),) * 6, player=1, filled=1)
    right = ConnectFourState(columns=((),) * 6 + ((1,),), player=1, filled=1)
    assert game.key(left) == game.key(right)


def test_heuristic_prefers_the_side_with_more_open_lines():
    game = ConnectFour()
    contested = ConnectFourState(columns=((1, 1, 1),) + ((),) * 6, player=1, filled=3)
    assert game.heuristic(contested, 0) > game.heuristic(contested, 1)


# -- Blackjack ------------------------------------------------------------

def test_an_ace_is_demoted_rather_than_busting():
    total, usable = _add(0, False, 1)
    assert (total, usable) == (11, True)
    total, usable = _add(total, usable, 10)
    assert (total, usable) == (21, True)
    total, usable = _add(total, usable, 5)
    assert (total, usable) == (16, False)


def test_hitting_past_21_loses_immediately(rng):
    game = Blackjack()
    state = game.initial_state(rng)
    while not state.done:
        state, rewards = game.step(state, HIT, rng)
    assert game.is_terminal(state)
    assert state.result in (-1.0, 0.0, 1.0)


def test_the_hole_card_stays_out_of_the_state_key(rng):
    """Two hands differing only in the dealer's hidden card must look alike."""
    game = Blackjack()
    from gaming_agents.games.blackjack import BlackjackState

    one = BlackjackState(player_total=18, player_ace=False, dealer_upcard=7, dealer_total=17, dealer_ace=False)
    two = BlackjackState(player_total=18, player_ace=False, dealer_upcard=7, dealer_total=20, dealer_ace=False)
    assert game.key(one) == game.key(two)


def test_sticking_ends_the_hand(rng):
    game = Blackjack()
    state, rewards = game.step(game.initial_state(rng), STICK, rng)
    assert state.done
    assert rewards == (state.result,)


# -- 2048 -----------------------------------------------------------------

@pytest.mark.parametrize(
    "row,expected,gained",
    [
        ([2, 2, 0, 0], [4, 0, 0, 0], 4),
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([4, 4, 8, 0], [8, 8, 0, 0], 8),
        ([2, 0, 2, 4], [4, 4, 0, 0], 4),
        ([2, 4, 2, 4], [2, 4, 2, 4], 0),
    ],
)
def test_a_row_merges_each_pair_once(row, expected, gained):
    out, points = _collapse(row)
    assert out == expected
    assert points == gained


def test_sliding_right_is_a_mirror_of_sliding_left():
    cells = (2, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    left, left_points = _slide(cells, 4, LEFT)
    right, right_points = _slide(cells, 4, RIGHT)
    assert left[:4] == (4, 4, 0, 0)
    assert right[:4] == (0, 0, 4, 4)
    assert left_points == right_points == 4


def test_sliding_up_merges_columns():
    cells = (2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    out, gained = _slide(cells, 4, UP)
    assert out[0] == 4
    assert gained == 4


def test_a_move_that_changes_nothing_is_not_legal(rng):
    from gaming_agents.games.twenty_forty_eight import Game2048State

    game = TwentyFortyEight()
    stuck = Game2048State(cells=(2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2))
    assert game.legal_moves(stuck) == []
    assert game.is_terminal(stuck)


def test_score_accumulates_across_moves(rng):
    game = TwentyFortyEight()
    state = game.initial_state(rng)
    total = 0
    while not game.is_terminal(state):
        state, _ = game.step(state, game.legal_moves(state)[0], rng)
        total = state.score
    assert total > 0
    assert game.outcome(state) == (float(total),)


# -- the solved Blackjack variant -----------------------------------------

def test_the_solver_agrees_with_basic_strategy_where_it_is_obvious():
    """Spot-checks against textbook basic strategy for a no-double, no-split game."""
    from gaming_agents.games.blackjack import BlackjackState

    game = Blackjack()

    def play(total, up, ace=False):
        return game.optimal_move(
            BlackjackState(player_total=total, player_ace=ace, dealer_upcard=up,
                           dealer_total=up, dealer_ace=up == 1)
        )

    assert play(20, 10) == STICK          # never break a good hand
    assert play(17, 7) == STICK           # hard 17 always stands
    assert play(11, 6) == HIT             # cannot bust, so always improve
    assert play(8, 10) == HIT
    assert play(16, 10) == HIT            # stiff hand, strong dealer: take the risk
    assert play(13, 6) == STICK           # stiff hand, weak dealer: let them bust

    # Hard 12 is the distinctive row of the chart, and the one a learner gets
    # wrong longest: stand only against the upcards most likely to bust.
    assert [play(12, up) for up in (2, 3)] == [HIT, HIT]
    assert [play(12, up) for up in (4, 5, 6)] == [STICK, STICK, STICK]
    assert [play(12, up) for up in (7, 8, 9, 10)] == [HIT] * 4

    # Soft 18 stands through a dealer 8 and hits from 9 up.
    assert [play(18, up, ace=True) for up in (2, 7, 8)] == [STICK] * 3
    assert [play(18, up, ace=True) for up in (9, 10, 1)] == [HIT] * 3
    assert play(17, 2, ace=True) == HIT   # soft 17 is never worth standing on


def test_the_house_edge_is_where_theory_says_it_is():
    from gaming_agents.games.blackjack import optimal_value

    assert -0.06 < optimal_value() < -0.03


def test_hitting_a_hand_that_cannot_bust_always_beats_sticking():
    from gaming_agents.games.blackjack import hit_value, stick_value

    for upcard in range(1, 11):
        for total in (9, 10, 11):
            assert hit_value(total, False, upcard) > stick_value(total, upcard)


def test_the_dealer_distribution_is_a_distribution():
    from gaming_agents.games.blackjack import _dealer_finals_from_upcard

    for upcard in range(1, 11):
        finals = dict(_dealer_finals_from_upcard(upcard))
        assert sum(finals.values()) == pytest.approx(1.0)
        assert all(f >= 17 for f in finals), "the dealer never stops below 17"
