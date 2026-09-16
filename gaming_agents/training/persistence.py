"""Saving and reloading what an agent has learned.

Brains are plain JSON on purpose: a Q-table you can open in an editor and read
is worth more while you are debugging than one that loads a few milliseconds
faster.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents.base import Agent
from ..registry import make_agent

FORMAT_VERSION = 1


def save_agent(agent: Agent, path: str | Path, *, kind: str | None = None, extra: dict[str, Any] | None = None) -> Path:
    """Write ``agent``'s learned state to ``path``. Returns the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind or _infer_kind(agent),
        "name": agent.name,
        "stats": agent.stats(),
        "agent": agent.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    # Write through a temporary file so an interrupted save cannot leave a
    # half-written brain where a good one used to be.
    scratch = target.with_suffix(target.suffix + ".tmp")
    scratch.write_text(json.dumps(payload, indent=1, sort_keys=True))
    scratch.replace(target)
    return target


def load_agent(path: str | Path, *, into: Agent | None = None, **agent_kwargs) -> Agent:
    """Rebuild a saved agent, or load a brain ``into`` an existing one."""
    payload = json.loads(Path(path).read_text())
    version = payload.get("format")
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: brain format {version!r}, expected {FORMAT_VERSION}")

    agent = into if into is not None else make_agent(payload.get("kind", "qlearner"), **agent_kwargs)
    agent.load_state_dict(payload.get("agent", {}))
    return agent


def peek(path: str | Path) -> dict[str, Any]:
    """Read a brain's metadata without materialising the agent."""
    payload = json.loads(Path(path).read_text())
    return {
        "format": payload.get("format"),
        "saved_at": payload.get("saved_at"),
        "kind": payload.get("kind"),
        "name": payload.get("name"),
        "stats": payload.get("stats", {}),
        "games": sorted(payload.get("agent", {}).get("q", {})),
    }


def _infer_kind(agent: Agent) -> str:
    from ..registry import AGENTS

    for name, factory in AGENTS.items():
        if type(agent) is factory:
            return name
    return type(agent).__name__
