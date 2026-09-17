"""Monte Carlo tree search (UCT).

Where the table-based learner stops scaling, search takes over: MCTS needs no
training at all, only the ability to simulate the game forward. That makes it
both the strongest opponent in this project and a useful teacher -- the
curriculum uses it as the sparring partner on the harder rungs.

Simulations are *open-loop*: each one re-steps the game from the root, so the
same code handles stochastic games like 2048 (where a child node's board is
resampled every visit) and deterministic ones (where it never changes).

In an imperfect-information game each simulation first draws a world the
acting seat cannot distinguish from the real one, and searches that. Averaging
over those worlds is a real technique -- determinized, or perfect-information
Monte Carlo, search -- but it has a known ceiling: inside any one simulation
the search can see the opponent's hand, so it plans as though its own hand were
public. It therefore never finds a reason to bluff, and never fears being
bluffed. It plays poker honestly, which is not the same as playing it well.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from ..core.game import Game, Move, State
from .base import Agent


class _Node:
    __slots__ = ("state", "player", "moves", "children", "visits", "value", "terminal")

    def __init__(self, game: Game, state: State) -> None:
        self.children: dict[Move, _Node] = {}
        #: Per-edge statistics. These survive a ``refresh``, so a stochastic
        #: game accumulates one set of counts per move across every sampled
        #: board rather than starting over each visit.
        self.visits: dict[Move, int] = {}
        self.value: dict[Move, float] = {}
        self.refresh(game, state)

    def refresh(self, game: Game, state: State) -> None:
        """Point this node at a freshly sampled state, keeping edge stats."""
        self.state = state
        self.terminal = game.is_terminal(state)
        self.player = -1 if self.terminal else game.current_player(state)
        self.moves: list[Move] = [] if self.terminal else game.legal_moves(state)
        for move in self.moves:
            self.visits.setdefault(move, 0)
            self.value.setdefault(move, 0.0)

    @property
    def total_visits(self) -> int:
        return sum(self.visits[m] for m in self.moves)


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(
        self,
        name: str | None = None,
        *,
        simulations: int = 200,
        exploration: float = 1.4,
        rollout_depth: int = 80,
        max_tree_depth: int = 40,
    ) -> None:
        super().__init__(name)
        self.simulations = simulations
        self.exploration = exploration
        """The ``c`` in UCT. Higher explores more; 1.4 is the usual default for
        rewards in [-1, 1]."""
        self.rollout_depth = rollout_depth
        self.max_tree_depth = max_tree_depth
        """How far one simulation may descend before rolling out. Two-player
        games here end long before this; it exists to stop a long game like
        2048 from making every simulation cost a full playthrough."""

    def select_move(self, game: Game, state: State, legal_moves: Sequence[Move], rng: random.Random) -> Move:
        if len(legal_moves) == 1:
            return legal_moves[0]

        seat = game.current_player(state)
        root = _Node(game, state)
        for _ in range(self.simulations):
            if game.imperfect_information:
                # Search a world consistent with what this seat actually knows,
                # resampled every simulation. Descending from the true state
                # would let the search read the opponent's hand.
                root.refresh(game, game.redeal(state, seat, rng))
            self._simulate(game, root, rng)

        # Pick the most-visited move, not the highest-valued one: visit counts
        # are the robust choice because a high mean over two visits is noise.
        best = max(root.moves, key=lambda m: (root.visits[m], root.value[m] / max(root.visits[m], 1)))
        return best

    # -- one simulation ---------------------------------------------------

    def _simulate(self, game: Game, root: _Node, rng: random.Random) -> None:
        node = root
        #: ``(node, move, rewards already banked before that move)``. The
        #: snapshot is what lets each edge be credited with the return that
        #: follows it rather than with the whole simulation's total.
        path: list[tuple[_Node, Move, tuple[float, ...]]] = []
        rewards = [0.0] * game.num_players

        # 1. Descend through the tree while every move has been tried once.
        depth = 0
        while not node.terminal and depth < self.max_tree_depth:
            move = self._tree_policy(node, rng)
            if move is None:
                break
            banked = tuple(rewards)
            state, step_rewards = game.step(node.state, move, rng)
            for i, r in enumerate(step_rewards):
                rewards[i] += r
            path.append((node, move, banked))

            child = node.children.get(move)
            if child is None:
                node.children[move] = _Node(game, state)
                node = node.children[move]
                break  # newly expanded: roll out from here
            if not game.deterministic:
                # Stochastic games resample the child's board every visit --
                # that is what makes this open-loop -- but the edge statistics
                # below it carry over.
                child.refresh(game, state)
            node = child
            depth += 1

        # 2. Roll out to the end of the game from wherever we stopped.
        if not node.terminal:
            for i, r in enumerate(self._rollout(game, node.state, rng)):
                rewards[i] += r

        # 3. Credit each edge with the return that came *after* it, for the
        # player who took it. Crediting the simulation's whole total instead
        # would fold reward earned earlier on the path into every edge below
        # it -- invisible in a game that only pays out at the end, but pure
        # noise in one like 2048 that scores on every move.
        for parent, move, banked in path:
            parent.visits[move] += 1
            parent.value[move] += rewards[parent.player] - banked[parent.player]

    def _tree_policy(self, node: _Node, rng: random.Random) -> Move | None:
        if not node.moves:
            return None
        unexplored = [m for m in node.moves if node.visits[m] == 0]
        if unexplored:
            return rng.choice(unexplored)

        log_total = math.log(node.total_visits)
        return max(
            node.moves,
            key=lambda m: node.value[m] / node.visits[m]
            + self.exploration * math.sqrt(log_total / node.visits[m]),
        )

    def _rollout(self, game: Game, state: State, rng: random.Random) -> tuple[float, ...]:
        totals = [0.0] * game.num_players
        for _ in range(self.rollout_depth):
            if game.is_terminal(state):
                break
            moves = game.legal_moves(state)
            if not moves:
                break
            state, rewards = game.step(state, rng.choice(moves), rng)
            for i, r in enumerate(rewards):
                totals[i] += r
        return tuple(totals)

    def stats(self) -> dict[str, int | float]:
        return {
            "simulations": self.simulations,
            "exploration": self.exploration,
            "max_tree_depth": self.max_tree_depth,
        }
