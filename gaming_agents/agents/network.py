"""A small value network, written out in plain Python.

The project has been stdlib-only throughout and stays that way here. At this
size that is not a sacrifice: one hidden layer over a few dozen inputs is a few
thousand multiply-adds, which is cheap next to the game simulation around it.

What it buys is the thing the linear agent could not do. A weighted sum can
only say "more of this is better"; it cannot say "more of this is better up to
a point, then worse". Blackjack needs exactly that -- hard twelve hits against
a two, stands against a five, hits again against a nine -- and no set of
weights over a single "dealer strength" number can express it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


class ValueNetwork:
    """One hidden layer, tanh inside, a single number out.

    Kept deliberately plain: no momentum, no batching, no adaptive step size.
    Every one of those would be a second explanation for any change in the
    results, and the question being asked is only whether a bend in the
    function helps.
    """

    def __init__(self, inputs: int, hidden: int = 24, outputs: int = 1,
                 rng: random.Random | None = None) -> None:
        source = rng or random.Random(0)
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        # Scaled by fan-in, so the starting output is small whatever the input
        # width -- otherwise an eighty-four-square board saturates tanh before
        # anything has been learned.
        spread = 1.0 / math.sqrt(max(1, inputs))
        self.w1 = [[source.uniform(-spread, spread) for _ in range(inputs)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[source.uniform(-spread, spread) for _ in range(hidden)]
                   for _ in range(outputs)]
        self.b2 = [0.0] * outputs

    def forward(self, x: Sequence[float]) -> tuple[list[float], list[float]]:
        """Return every output and the hidden activations the update will need."""
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            total = bias
            for weight, value in zip(row, x):
                total += weight * value
            hidden.append(math.tanh(total))
        outs = []
        for row, bias in zip(self.w2, self.b2):
            total = bias
            for weight, unit in zip(row, hidden):
                total += weight * unit
            outs.append(total)
        return outs, hidden

    def value(self, x: Sequence[float]) -> float:
        """The single output, for a network used as a value function."""
        return self.forward(x)[0][0]

    def learn(self, x: Sequence[float], hidden: Sequence[float],
              errors: "float | dict[int, float]", rate: float) -> None:
        """Nudge the weights so this input's outputs move by ``errors``.

        A bare number means the only output, which is the value-function case.
        A mapping updates just the outputs named -- in Q-learning only the
        action actually taken has a target, and touching the others would be
        inventing evidence about moves that were never played.

        Backpropagation, spelled out: output weights move in proportion to the
        unit that fed them, and each hidden unit's share is scaled by how much
        room tanh has left to move.
        """
        wanted = {0: errors} if isinstance(errors, (int, float)) else errors
        blame = [0.0] * len(hidden)
        for index, error in wanted.items():
            step = rate * error
            row = self.w2[index]
            for unit_index, unit in enumerate(hidden):
                blame[unit_index] += step * row[unit_index]
                row[unit_index] += step * unit
            self.b2[index] += step

        for unit_index, unit in enumerate(hidden):
            share = blame[unit_index] * (1.0 - unit * unit)
            if share:
                row = self.w1[unit_index]
                for slot, value in enumerate(x):
                    if value:
                        row[slot] += share * value
                self.b1[unit_index] += share

    # -- persistence ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "hidden": self.hidden, "outputs": self.outputs,
                "w1": [list(row) for row in self.w1], "b1": list(self.b1),
                "w2": [list(row) for row in self.w2], "b2": list(self.b2)}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "ValueNetwork":
        net = cls(data["inputs"], data["hidden"], data.get("outputs", 1))
        net.w1 = [list(row) for row in data["w1"]]
        net.b1 = list(data["b1"])
        net.w2 = [list(row) for row in data["w2"]]
        net.b2 = list(data["b2"])
        return net
