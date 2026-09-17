"""Name -> constructor lookups for games and agents.

Everything the CLI and the curriculum can refer to by string lives here, so
adding a game or an agent is a one-line change in one place.
"""

from __future__ import annotations

from typing import Callable

from .agents.base import Agent
from .agents.cfr import CFRAgent
from .agents.human import HumanAgent
from .agents.mcts import MCTSAgent
from .agents.minimax import MinimaxAgent
from .agents.solvers import NashKuhnAgent, PerfectBlackjackAgent, PerfectNimAgent
from .agents.random_agent import FirstMoveAgent, RandomAgent
from .agents.tabular_q import TabularQAgent
from .games.blackjack import Blackjack
from .games.connect_four import ConnectFour, ConnectFourMini
from .games.kuhn_poker import KuhnPoker
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
    "kuhn": KuhnPoker,
    "2048": TwentyFortyEight,
}

AGENTS: dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "first": FirstMoveAgent,
    "qlearner": TabularQAgent,
    "cfr": CFRAgent,
    "mcts": MCTSAgent,
    "minimax": MinimaxAgent,
    "perfect-nim": PerfectNimAgent,
    "perfect-blackjack": PerfectBlackjackAgent,
    "nash-kuhn": NashKuhnAgent,
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


#: Agents whose strength is a single number, so ``mcts:800`` can mean something.
STRENGTH_KNOB = {"mcts": "simulations", "minimax": "depth"}


def parse_spec(spec: str) -> tuple[str, dict[str, int]]:
    """Split ``"mcts:800"`` into the agent name and how hard it should think.

    A bare name is left alone. The number means simulations for MCTS and ply
    depth for minimax -- the one knob that decides how strong each of them is.
    """
    name, _, strength = spec.partition(":")
    if not strength:
        return name, {}
    knob = STRENGTH_KNOB.get(name)
    if knob is None:
        raise ValueError(f"{name!r} has no strength setting; drop the ':{strength}'")
    try:
        value = int(strength)
    except ValueError:
        raise ValueError(f"{strength!r} is not a number of {knob}") from None
    if value < 1:
        raise ValueError(f"{knob} must be at least 1")
    return name, {knob: value}


def make_agent(spec: str, /, **kwargs) -> Agent:
    """Build an agent by registry name, or by a strength spec like ``mcts:800``.

    ``spec`` is positional-only for the same reason as :func:`make_game`:
    agents take a ``name=`` of their own.
    """
    name, tuning = parse_spec(spec)
    try:
        factory = AGENTS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; try one of {sorted(AGENTS)}") from None
    return factory(**{**tuning, **kwargs})


def describe_games() -> list[tuple[str, str, int, int]]:
    """``(name, title, players, difficulty)`` for every registered game."""
    rows = []
    for name in GAMES:
        game = make_game(name)
        rows.append((name, game.title, game.num_players, game.difficulty))
    return sorted(rows, key=lambda row: row[3])
