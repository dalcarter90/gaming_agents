# gaming-agents

Agents that learn games by playing them, on a ladder that runs from trivial to
genuinely hard.

The point of the project is the **curriculum**: one learner starts on Nim,
and only moves up to the next game once it has actually mastered the current
one. Promotion is measured, not scheduled — a stage that does not clear its bar
ends the climb.

```
$ python -m gaming_agents curriculum --curriculum classic

=== stage 2/4: Tic-Tac-Toe ===
    goal: never lose -- correct play here is a draw, so anything less is a mistake
    round 1: 20,000 episodes  non_loss_rate=0.675 (bar 0.95)  0W/135D/65L over 200
    round 2: 40,000 episodes  non_loss_rate=0.980 (bar 0.95)  0W/196D/4L over 200
    benchmark vs random: 182W/17D/1L over 200 (win 91%, non-loss 100%)
    -> mastered Tic-Tac-Toe; promoting
...

stage          rounds  episodes     score     bar  result
--------------------------------------------------------------
nim                 1     3,000     1.000    0.90  MASTERED
tictactoe           2    40,000     0.980    0.95  MASTERED
connect4-mini       3    60,000     0.760    0.75  MASTERED
connect4            2    80,000     0.828    0.80  MASTERED

total episodes played: 183,000
curriculum complete
```

Pure Python, standard library only. No install step needed to try it.

```bash
git clone <this repo> && cd gaming_agents
python -m gaming_agents games          # what is available
python -m gaming_agents curriculum     # watch an agent climb the ladder
python -m gaming_agents play --game tictactoe --first  # play it yourself
```

`pip install -e ".[dev]"` if you want the `gaming-agents` command and pytest.

## The ladder

| rung | game | players | why it is here |
|---|---|---|---|
| 1 | **Nim** | 2 | Solved, tiny, and has a known-perfect opponent. If a learner cannot master this, the bug is in the learner. |
| 2 | **Tic-Tac-Toe** | 2 | Still solvable, but now the reward arrives nine moves after the mistake that caused it. |
| 3 | **Connect Four (4×5)** | 2 | Too big to tabulate exhaustively; the first rung where *how* you generalise starts to matter. |
| 3 | **Blackjack** | 1 | The world itself is random. A bad outcome is no longer evidence of a bad decision. |
| 4 | **Connect Four (6×7)** | 2 | The full board — the same idea, an order of magnitude more of it. |
| 5 | **2048** | 1 | Long episodes, a huge state space, and reward spread thinly across ~120 moves. |

Three curricula ship: `classic` (the two-player rungs), `solo` (Blackjack then
2048), and `full` (all six, interleaved by difficulty).

## The agents

| agent | learns? | what it is |
|---|---|---|
| `random` | no | The floor. Every number below is relative to this. |
| `qlearner` | **yes** | Tabular Q-learning. One table per game, so a single agent carries everything it learned up the ladder. |
| `mcts` | no | Monte Carlo tree search (UCT). Needs no training, only the ability to simulate — so it scales where the table does not. |
| `minimax` | no | Alpha-beta. Exhaustive on the small games, so it plays *perfectly* there. |
| `perfect-nim` | no | The nim-sum rule. Exact and instant. |
| `human` | — | You. |

## What actually happens

Measured on this machine, seed 0. "vs random" is win rate; Blackjack and 2048
are mean score per game.

| game | random | learned (`qlearner`) | search (`mcts`) |
|---|---|---|---|
| Nim vs random | 50% | **100%** after 3k games | — |
| Nim vs perfect play | 0% | **50%** — which *is* optimal¹ | — |
| Tic-Tac-Toe vs minimax (non-loss) | 15% | **98%** after 40k games | 82% |
| Connect Four 6×7 vs random | 50% | **83%** after 80k games | 100% |
| Blackjack (per hand) | −0.39 | **−0.057** after 50k hands | — |
| 2048 (score) | 1,112 | 1,278 after 15k games² | **11,326** |

¹ The (1,3,5,7) opening is a *loss* for whoever moves first, and evaluation
alternates seats. Winning exactly half is what perfect play looks like here;
it is the reason the learner is judged against `random` on this rung.

² Deliberate, and the most interesting number in the table. A table keyed on
exact 4×4 boards will essentially never see the same board twice, so fifteen
thousand games of experience buy almost nothing — 2048 is where tabular
learning runs out of road, and the curriculum says so instead of pretending
otherwise:

```
2048                3    15,000  1278.200 3000.00  not yet
```

Search clears the same rung without training at all.

## How it fits together

```
core/game.py        the Game interface — one abstraction for 1- and 2-player,
                    deterministic and stochastic games
core/episode.py     runs one episode and hands each seat its own transitions
agents/             random, tabular Q-learning, MCTS, alpha-beta, human
training/curriculum.py   the ladder, the mastery gates, the promotion logic
```

The dependency arrow points one way: games know nothing about agents, agents
know nothing about training, and `registry.py` sits above all three.

### The one subtle bit

In a two-player game, a seat's "next state" is **not** the position right after
its own move — the opponent replies first. So the episode runner holds each
seat's decision open until that seat is due to act again, accumulating any
reward earned in the meantime:

```python
Transition(state=..., move=..., reward=<everything earned since>,
           next_state=<the position I face on MY next turn>, done=...)
```

That is what makes an ordinary Q-learning update correct in a turn-based
adversarial game, with no sign-flipping anywhere. In self-play the same agent
holds both seats and learns from both sides of every game.

## Adding a game

Subclass `Game`, implement the nine abstract methods, and register it:

```python
# gaming_agents/registry.py
GAMES["hexapawn"] = Hexapawn
```

`tests/test_game_contract.py` runs against every registered game
automatically — states must be immutable, move labels unique, terminal states
moveless, and random play must always terminate. If it passes that file, every
agent in the project can already play it.

Set `difficulty` and the game slots itself into the right rung of the `full`
curriculum.

## Commands

```bash
python -m gaming_agents games
python -m gaming_agents train --game connect4 --episodes 50000 --save brains/c4.json
python -m gaming_agents eval  --game connect4 --load brains/c4.json --opponent mcts
python -m gaming_agents curriculum --curriculum full --scale 0.5 --no-gate
python -m gaming_agents watch --game connect4 --agent mcts --opponent minimax
python -m gaming_agents tournament --game tictactoe --agents random mcts:400 minimax
python -m gaming_agents play --game connect4 --agent mcts:800 --first
python -m gaming_agents inspect brains/c4.json
```

Any agent can carry a strength: `mcts:800` searches 800 simulations per move,
`minimax:6` looks six ply ahead. `--scale` multiplies every episode budget
(trade wall time for strength), `--no-gate` keeps climbing past a failed stage,
and `--seed` makes any run reproducible. Brains are plain JSON you can open and
read.

## Tests

```bash
python -m pytest        # ~160 tests, about two minutes
```

They check real properties rather than just wiring: that minimax never loses a
game it should draw, that the learner's 50% against perfect Nim is the optimal
50%, that evaluation swaps seats, and that a saved brain plays identically to
the one it came from.

## Where to take it next

- **Function approximation.** The tabular agent's wall on Connect Four and 2048
  is exactly the wall that features or a neural net are for. `Game.key` is the
  seam — swap it for a feature vector.
- **Learning from search.** MCTS is already the strongest thing here. Train the
  learner on its move choices and you have the outline of AlphaZero.
- **An LLM agent.** `Agent` needs one method, and every game renders to text
  with legal moves listed. `agents/llm.py` would drop straight into the same
  ladder and tournaments.
