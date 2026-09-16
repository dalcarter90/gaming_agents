import random

import pytest

from gaming_agents.registry import GAMES, make_game


@pytest.fixture
def rng() -> random.Random:
    """A seeded RNG, so a failing test fails the same way twice."""
    return random.Random(1234)


@pytest.fixture(params=sorted(GAMES))
def any_game(request):
    """Every registered game in turn, for the shared contract tests."""
    return make_game(request.param)
