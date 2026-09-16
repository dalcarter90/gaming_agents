"""Name -> constructor lookups for games and agents.

Everything the CLI and the curriculum can refer to by string lives here, so
adding a game or an agent is a one-line change in one place.
"""

from __future__ import annotations

from typing import Callable

from .agents.base import Agent
from .agents.human import HumanAgent
from .agents.mcts import MCTSAgent
from .agents.minimax import MinimaxAgent, PerfectNimAgent
from .agents.random_agent import FirstMoveAgent, RandomAgent
from .agents.tabular_q import TabularQAgent
from .games.blackjack import Blackjack
from .games.connect_four import ConnectFour, ConnectFourMini
from .games.nim import Nim
from .games.tictactoe import TicTacToe
from .games.twenty_forty_eight import TwentyFortyEight
from .core.game import Game

GAMES: dict[str, Callable[[], Game]] = {
    "nim": Nim,
    "tictactoe": TicTacToe,
    "connect4-mini": ConnectFourMini,
    "connect4": ConnectFour,
    "blackjack": Blackjack,
    "2048": TwentyFortyEight,
}

AGENTS: dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "first": FirstMoveAgent,
    "qlearner": TabularQAgent,
    "mcts": MCTSAgent,
    "minimax": MinimaxAgent,
    "perfect-nim": PerfectNimAgent,
    "human": HumanAgent,
}


def make_game(name: str, /, **kwargs) -> Game:
    """Build a game by registry name. ``name`` is positional-only so that a
    game taking its own ``name=`` keyword is not shadowed by it."""
    try:
        factory = GAMES[name]
    except KeyError:
        raise ValueError(f"unknown game {name!r}; try one of {sorted(GAMES)}") from None
    return factory(**kwargs)


def make_agent(name: str, /, **kwargs) -> Agent:
    """Build an agent by registry name. Positional-only for the same reason as
    :func:`make_game` -- agents take a ``name=`` of their own."""
    try:
        factory = AGENTS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; try one of {sorted(AGENTS)}") from None
    return factory(**kwargs)


def describe_games() -> list[tuple[str, str, int, int]]:
    """``(name, title, players, difficulty)`` for every registered game."""
    rows = []
    for name in GAMES:
        game = make_game(name)
        rows.append((name, game.title, game.num_players, game.difficulty))
    return sorted(rows, key=lambda row: row[3])
