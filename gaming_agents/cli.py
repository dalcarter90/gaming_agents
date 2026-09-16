"""Command line entry point: ``python -m gaming_agents <command>``."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .agents.base import Agent
from .core.episode import play_episode
from .registry import AGENTS, describe_games, make_agent, make_game
from .training.curriculum import CURRICULA, get_curriculum, run_curriculum
from .training.evaluate import evaluate
from .training.persistence import load_agent, peek, save_agent
from .training.trainer import tournament, train

DEFAULT_BRAIN = "brains/learner.json"


# -- shared helpers -------------------------------------------------------

def _agent_from(spec: str, brain: str | None, *, name: str | None = None, **kwargs) -> Agent:
    """Build an agent, optionally warm-started from a saved brain."""
    agent = make_agent(spec, name=name, **kwargs) if name else make_agent(spec, **kwargs)
    if brain:
        path = Path(brain)
        if path.exists():
            load_agent(path, into=agent)
        else:
            print(f"note: no brain at {path}; starting from scratch", file=sys.stderr)
    return agent


def _maybe_save(agent: Agent, path: str | None) -> None:
    if path:
        saved = save_agent(agent, path)
        print(f"\nsaved brain -> {saved}")


# -- commands -------------------------------------------------------------

def cmd_games(args: argparse.Namespace) -> int:
    print(f"{'name':<14} {'players':>7} {'rung':>5}  title")
    print("-" * 52)
    for name, title, players, difficulty in describe_games():
        print(f"{name:<14} {players:>7} {difficulty:>5}  {title}")
    print(f"\nagents: {', '.join(sorted(AGENTS))}")
    print(f"curricula: {', '.join(sorted(CURRICULA))}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    game = make_game(args.game)
    learner = _agent_from(args.agent, args.load)
    sparring = _agent_from(args.opponent, None) if args.opponent else None
    judge = _agent_from(args.eval_opponent, None) if game.num_players > 1 else None

    print(f"training {learner.name} on {game.title} for {args.episodes:,} episodes")
    print(f"  sparring partner: {args.opponent or 'itself (self-play)'}")

    report = train(
        game,
        learner,
        args.episodes,
        rng,
        opponent=sparring,
        eval_opponent=judge,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    for episodes, evaluation in report.history:
        print(f"  {episodes:>9,} episodes: {evaluation.summary()}")
    print(f"\ndone in {report.seconds:.1f}s; agent stats: {learner.stats()}")
    _maybe_save(learner, args.save)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    game = make_game(args.game)
    left = _agent_from(args.agent, args.load)
    right = _agent_from(args.opponent, None) if game.num_players > 1 else None

    report = evaluate(game, left, right, args.episodes, rng)
    label = f"vs {right.name}" if right else "solo"
    print(f"{left.name} on {game.title} {label}: {report.summary()}")
    print(f"  mean outcome: {report.mean_outcome:,.3f}   mean turns: {report.mean_turns:.1f}")
    return 0


def cmd_curriculum(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    ladder = get_curriculum(args.curriculum)
    learner = _agent_from(args.agent, args.load)

    print(f"curriculum '{ladder.name}': {ladder.description}")
    print(f"agent: {learner.name}   episode scale: {args.scale}x   gating: {'on' if not args.no_gate else 'off'}")

    result = run_curriculum(ladder, learner, rng, gate=not args.no_gate, scale=args.scale)

    print("\n" + result.table())
    print(f"\ntotal episodes played: {result.total_episodes:,}")
    print("curriculum complete" if result.completed else "curriculum incomplete")
    _maybe_save(learner, args.save)
    return 0 if result.completed else 1


def cmd_play(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    game = make_game(args.game)
    learner = _agent_from(args.agent, args.load)

    if game.num_players == 1:
        line_up = [learner]
        you = None
    else:
        human = make_agent("human")
        you = 0 if args.first else 1
        line_up = [human, learner] if you == 0 else [learner, human]

    result = play_episode(game, line_up, rng, learn=False, record=args.show)
    print("\n" + game.render(result.final_state))
    if you is None:
        print(f"\nfinal score: {result.outcome[0]:,.0f} over {result.turns} moves")
    elif result.outcome[you] > 0:
        print("\nyou win")
    elif result.outcome[you] < 0:
        print(f"\n{learner.name} wins")
    else:
        print("\ndraw")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    game = make_game(args.game)
    left = _agent_from(args.agent, args.load, name="left")
    line_up = [left] if game.num_players == 1 else [left, _agent_from(args.opponent, None, name="right")]

    result = play_episode(game, line_up, rng, learn=False, record=True)
    for turn in result.transcript:
        print(f"\n--- {line_up[turn.seat].name} (seat {turn.seat}) plays {turn.move} ---")
        print(turn.board)
    print("\n=== final ===")
    print(game.render(result.final_state))
    print(f"outcome: {result.outcome}  in {result.turns} moves")
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    game = make_game(args.game)
    field = []
    for spec in args.agents:
        brain = args.load if spec == args.agent_with_brain else None
        field.append(_agent_from(spec, brain, name=spec))
    if len({a.name for a in field}) != len(field):
        raise SystemExit("tournament entrants need distinct names")

    print(f"{game.title}: {args.episodes} games per pairing\n")
    print(f"{'agent':<14} {'points':>7} {'W':>5} {'D':>5} {'L':>5}")
    print("-" * 40)
    for name, points, wins, draws, losses in tournament(game, field, args.episodes, rng):
        print(f"{name:<14} {points:>7.1f} {wins:>5} {draws:>5} {losses:>5}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    info = peek(args.brain)
    print(f"brain: {args.brain}")
    for field in ("name", "kind", "saved_at"):
        print(f"  {field}: {info[field]}")
    print(f"  games learned: {', '.join(info['games']) or '(none)'}")
    for field, value in info["stats"].items():
        print(f"  {field}: {value}")
    return 0


# -- wiring ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaming-agents",
        description="Train agents that learn games by playing, from simple games to advanced ones.",
        epilog="Agents may carry a strength: 'mcts:800' searches 800 simulations a move, "
               "'minimax:6' looks six ply ahead.",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    sub = parser.add_subparsers(dest="command", required=True)

    games = sub.add_parser("games", help="list the games, agents and curricula available")
    games.set_defaults(func=cmd_games)

    trainer = sub.add_parser("train", help="train one agent on one game")
    trainer.add_argument("--game", default="tictactoe")
    trainer.add_argument("--agent", default="qlearner")
    trainer.add_argument("--episodes", type=int, default=20_000)
    trainer.add_argument("--opponent", default=None, help="sparring partner (default: self-play)")
    trainer.add_argument("--eval-opponent", default="random")
    trainer.add_argument("--eval-every", type=int, default=5_000, help="0 to evaluate only at the end")
    trainer.add_argument("--eval-episodes", type=int, default=200)
    trainer.add_argument("--load", default=None, help="warm-start from a saved brain")
    trainer.add_argument("--save", default=None, help="where to write the trained brain")
    trainer.set_defaults(func=cmd_train)

    evaluator = sub.add_parser("eval", help="measure an agent without training it")
    evaluator.add_argument("--game", default="tictactoe")
    evaluator.add_argument("--agent", default="qlearner")
    evaluator.add_argument("--opponent", default="random")
    evaluator.add_argument("--episodes", type=int, default=500)
    evaluator.add_argument("--load", default=None)
    evaluator.set_defaults(func=cmd_eval)

    ladder = sub.add_parser("curriculum", help="walk an agent up a ladder of games")
    ladder.add_argument("--curriculum", default="classic", choices=sorted(CURRICULA))
    ladder.add_argument("--agent", default="qlearner")
    ladder.add_argument("--scale", type=float, default=1.0, help="multiply every episode budget")
    ladder.add_argument("--no-gate", action="store_true", help="keep going after a failed stage")
    ladder.add_argument("--load", default=None)
    ladder.add_argument("--save", default=DEFAULT_BRAIN)
    ladder.set_defaults(func=cmd_curriculum)

    human = sub.add_parser("play", help="play a game yourself against an agent")
    human.add_argument("--game", default="tictactoe")
    human.add_argument("--agent", default="mcts:800")
    human.add_argument("--first", action="store_true", help="you move first")
    human.add_argument("--show", action="store_true")
    human.add_argument("--load", default=None)
    human.set_defaults(func=cmd_play)

    viewer = sub.add_parser("watch", help="print a full game between two agents")
    viewer.add_argument("--game", default="tictactoe")
    viewer.add_argument("--agent", default="mcts:400")
    viewer.add_argument("--opponent", default="random")
    viewer.add_argument("--load", default=None)
    viewer.set_defaults(func=cmd_watch)

    arena = sub.add_parser("tournament", help="round-robin several agents")
    arena.add_argument("--game", default="tictactoe")
    arena.add_argument("--agents", nargs="+", default=["random", "mcts", "minimax"])
    arena.add_argument("--episodes", type=int, default=50)
    arena.add_argument("--load", default=None)
    arena.add_argument("--agent-with-brain", default="qlearner", help="which entrant --load applies to")
    arena.set_defaults(func=cmd_tournament)

    inspector = sub.add_parser("inspect", help="summarise a saved brain")
    inspector.add_argument("brain")
    inspector.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
