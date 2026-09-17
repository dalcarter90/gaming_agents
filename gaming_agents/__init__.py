"""Agents that learn games by playing them, on a ladder from simple to advanced.

Start at :func:`gaming_agents.training.run_curriculum`, or from the shell::

    python -m gaming_agents curriculum --curriculum classic
"""

from .agents import Agent, CFRAgent, MCTSAgent, MinimaxAgent, RandomAgent, TabularQAgent
from .core import Game, play_episode
from .registry import make_agent, make_game
from .training import (
    evaluate,
    exploitability,
    get_curriculum,
    load_agent,
    run_curriculum,
    save_agent,
    train,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "CFRAgent",
    "Game",
    "MCTSAgent",
    "MinimaxAgent",
    "RandomAgent",
    "TabularQAgent",
    "evaluate",
    "exploitability",
    "get_curriculum",
    "load_agent",
    "make_agent",
    "make_game",
    "play_episode",
    "run_curriculum",
    "save_agent",
    "train",
]
