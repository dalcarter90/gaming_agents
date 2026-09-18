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
| 6 | **Kuhn Poker** | 2 | Hidden information. The first rung where the *state* is not what a player gets to see, and where the best strategy is a mixture rather than a move. |

Four curricula ship: `classic` (the two-player rungs), `solo` (Blackjack then
2048), `poker`, and `full` (all seven, interleaved by difficulty).

## The agents

| agent | learns? | what it is |
|---|---|---|
| `random` | no | The floor. Every number below is relative to this. |
| `qlearner` | **yes** | Tabular Q-learning. One number per exact position — it memorises. |
| `linear` | **yes** | Learns weights over a *description* of a position, so one game teaches it about every position that resembles it — and with `shared=True`, one *game* teaches it about the next one. |
| `mcts` | no | Monte Carlo tree search (UCT). Needs no training, only the ability to simulate — so it scales where the table does not. |
| `minimax` | no | Alpha-beta. Exhaustive on the small games, so it plays *perfectly* there. Refuses poker rather than cheat at it. |
| `cfr` | **yes** | Counterfactual regret minimisation. Learns by walking the game tree, and converges to an unexploitable mixed strategy. |
| `perfect-nim` | no | The nim-sum rule. Exact and instant. |
| `perfect-blackjack` | no | The variant solved by dynamic programming — the ceiling, not an opponent. |
| `nash-kuhn` | no | Kuhn's 1950 equilibrium, in closed form. Unexploitable by construction. |
| `human` | — | You. |

## What actually happens

Measured on this machine. "vs random" is win rate; Blackjack and 2048 are mean
score per game. Blackjack figures are over 200,000 hands (±0.004 at 95%), which
matters — see note 3.

| game | random | learned (`qlearner`) | search (`mcts`) |
|---|---|---|---|
| Nim vs random | 50% | **100%** after 3k games | — |
| Nim vs perfect play | 0% | **50%** — which *is* optimal¹ | — |
| Tic-Tac-Toe vs minimax (non-loss) | 15% | **98%** after 40k games | 82% |
| Connect Four 6×7 vs random | 50% | 83% after 80k games — or **99% after 100**⁵ | 100% |
| Blackjack (per hand) | −0.39 | **−0.045** after 200k hands³ | — |
| 2048 (score) | 1,112 | 1,219 after 15k games² | **11,326** |
| Kuhn poker (exploitability) | 0.469 | 0.250 | 0.164 — and see below⁴ |

¹ The (1,3,5,7) opening is a *loss* for whoever moves first, and evaluation
alternates seats. Winning exactly half is what perfect play looks like here;
it is the reason the learner is judged against `random` on this rung.

² Deliberate, and the most interesting number in the table. A table keyed on
exact 4×4 boards will essentially never see the same board twice, so fifteen
thousand games of experience buy almost nothing — 2048 is where tabular
learning runs out of road, and the curriculum says so instead of pretending
otherwise:

```
    round 3: 15,000 episodes  mean_outcome=1219 (bar 3000)  20 games, mean score 1,219.0
    for reference, mcts:20 scores: 5 games, mean score 12,001.6
    -> did not clear the bar for 2048 (best 1219)
```

Every stage can name a `benchmark` — a reference it reports but never gates on.
For a two-player rung that is a stronger opponent; for a solo rung it is scored
on its own, so the bar stops being an abstract number.

⁵ The difference between memorising positions and describing them. Its own
section is below.

³ Blackjack is solved exactly in `games/blackjack.py` (`perfect-blackjack` plays
it), so this rung has a known ceiling: **−0.0466 per hand**. That is the house edge in a game with no
doubling, splitting or naturals — perfect play still loses, and "success" here
means getting close to −0.047, not close to zero. The learner reaches −0.045,
which is the optimum to within sampling error, and its policy matches the
solved strategy table.

Getting there needed `alpha_mode="visits"`. With a constant step size the
learner plateaus at −0.053 and **more training does not help** — 50k and 500k
hands score the same, because a fixed step size never settles in a stochastic
game. A 1/n schedule converges properly:

| step size | 50k hands | 500k hands |
|---|---|---|
| constant `alpha=0.05` | −0.053 | −0.053 |
| `alpha_mode="visits"` | −0.056 | **−0.045** |

The stage now reports that ceiling next to the result, so the number is
readable on its own:

```
    round 1: 100,000 episodes  mean_outcome=-0.037 (bar -0.055)  20000 games
    for reference, perfect-blackjack scores: 20000 games, mean score -0.037
    -> mastered Blackjack; promoting
```

That distinction is why `alpha_mode` exists rather than being hard-coded: a
constant step is the *right* choice against a self-play opponent that keeps
improving, and the wrong one against a dealer who never changes.

⁴ Poker inverts the scoreboard. The full table is the next section, because
one footnote cannot hold it.

## Describing a position instead of memorising it

The tabular learner keeps one number per exact board. Count how often it ever
gets to reuse one:

| | Tic-Tac-Toe | Connect Four |
|---|---|---|
| positions recorded in 40,000 games | 4,329 | 245,450 |
| seen **exactly once** | 28% | **95%** |
| median times revisited | 3 | **1** |

On Connect Four, 95% of what it writes down it never reads again. The position
never recurs, so the experience is spent the moment it is recorded. Separately,
86% of its Tic-Tac-Toe table is the same position rotated or flipped — it
learns the corner opening four times over.

`Game.features` is the fix. `key` answers *which position is this*; `features`
answers *what is this position like* — threats I have, threats they have, who
holds the centre. `LinearAgent` learns a weight per feature, so every game
adjusts its opinion of every position sharing any of them, including ones it
will never see.

### Does it work?

Win rate against a random opponent on Connect Four, same games, same opponent:

| agent | 100 | 500 | 2,500 | 10,000 | 40,000 |
|---|---|---|---|---|---|
| `qlearner` (memorises) | 0.51 | 0.51 | 0.49 | 0.55 | 0.69 |
| `linear-onehot` (control) | 0.78 | 0.78 | 0.84 | 0.85 | 0.87 |
| `linear` (describes) | **0.97** | **0.99** | **1.00** | 0.99 | 0.99 |

Games needed to clear the curriculum's 80% mastery bar: **100** for the feature
learner, 2,500 for the control, and tens of thousands for the table — the
curriculum run took 80,000.

The same gap shows across the whole ladder. Every rung, same mastery bars:

```
$ python -m gaming_agents curriculum --curriculum classic --agent linear

stage          rounds  episodes     score      bar  result
---------------------------------------------------------------
nim                 1       150  0.993333      0.9  MASTERED
tictactoe           1     1,000     0.965     0.95  MASTERED
connect4-mini       1     1,000  0.796667     0.75  MASTERED
connect4            1     2,000    0.9975      0.8  MASTERED

total episodes played: 4,150
```

**4,150 games against the tabular learner's 183,000**, and every rung cleared
on the first round rather than the third or fifth.

And what it has to remember, after 40,000 games:

```
qlearner         245,855 positions
linear-onehot    156,320 weights
linear                10 weights
```

Ten numbers. You can read them, which a 245,000-row table does not allow:

```
bias            +0.790     my_building     +0.101
their_centre    -0.687     win_available   +0.090
my_centre       +0.240     my_near_wins    +0.049
their_near_wins -0.190     their_building  -0.037
progress        +0.127     must_block      +0.012
```

It worked out that the centre column decides Connect Four, which is the first
thing anyone is taught about the game.

### Two honest caveats

**The lookahead is doing some of the work.** `LinearAgent` picks moves by
playing each one out a single step and valuing the result, which the tabular
agent never did. Untrained, with every weight at zero, that alone wins **77%**
against random. So the jump from `qlearner` to `linear` is not all about
features — which is exactly why `linear-onehot` is in the table. It has the
same lookahead and the same information as `qlearner`, so the gap between the
last two rows, 0.87 against 0.99 and 2,500 games against 100, is what
*describing* a position buys on its own.

**It is not a strong Connect Four player yet**, it is a fast learner. Against
real opposition, after 20,000 games:

| | vs random | vs `mcts:60` | vs `mcts:200` |
|---|---|---|---|
| `linear` (20,000 games) | 98% | **69%** | 27% |
| `qlearner` (80,000 games) | 82% | 16% | — |

Beating a searcher at 60 simulations while the table manages 16% is a real
result, not an artefact of thrashing a random opponent. Losing 73–27 to the
same searcher at 200 simulations is equally real.

What does *not* work is playing the two learners against each other, and the
reason is worth keeping:

```
alternating seats            100W / 0D / 100L
linear always moves first    200W / 0D /   0L
qlearner always moves first  200W / 0D /   0L
```

Whoever moves first wins every single game, whichever agent it is. That is not
one game replayed — 100 matches produced 88 distinct games — it is the
first-move advantage in Connect Four being decisive between two players of
roughly this standard. A head-to-head here measures the seating, not the skill,
which is why both are scored against a common opponent instead.

**The features are hand-written.** Someone who knows Connect Four chose
"threats" and "centre control". That is the honest boundary of this step: the
agent learned *how much* each of those matters, not *that they were the things
to look at*. Learning the description itself is what a neural network does, and
it is the next rung rather than this one.

## Carrying what you learned into a game you have never played

Each game described its positions in its own words, so nothing crossed between
them: the tabular agent keeps a separate table per game, and a Connect Four
position means nothing in Tic-Tac-Toe.

Now the board games share a vocabulary. "One move from winning", "must block",
"holding the centre", "how far along we are" are the *same named quantity* on
both boards, and `LinearAgent(shared=True)` keeps one set of opinions covering
every game instead of one per game. Descriptions peculiar to a board are
prefixed (`tictactoe:my_corners`) so they stay local.

The interesting column is the first one — how it plays a game it has never
played, purely on what a different game taught it.

**Connect Four for 3,000 games, then Tic-Tac-Toe:**

| games of tic-tac-toe played | 0 | 25 | 100 | 500 | 2,000 |
|---|---|---|---|---|---|
| win rate vs random, from scratch | 0.667 | 0.523 | 0.843 | 0.945 | 0.936 |
| win rate vs random, after Connect Four | **0.879** | **0.925** | 0.950 | 0.947 | 0.938 |
| never-lose vs perfect, from scratch | 0.150 | 0.633 | 0.472 | 0.956 | 0.967 |
| never-lose vs perfect, after Connect Four | **0.606** | 0.611 | **0.822** | 0.956 | 0.967 |

**Tic-Tac-Toe for 3,000 games, then Connect Four:**

| games of connect four played | 0 | 25 | 100 | 500 | 2,000 |
|---|---|---|---|---|---|
| win rate vs random, from scratch | 0.767 | 0.841 | 0.954 | 0.993 | 0.995 |
| win rate vs random, after Tic-Tac-Toe | **0.999** | **1.000** | 1.000 | 1.000 | 1.000 |

Three things worth taking from this.

**It transfers, and by a lot.** Tic-Tac-Toe hands Connect Four a 99.9% win rate
before a single game of it has been played. In the other direction Connect Four
takes Tic-Tac-Toe from 0.150 to 0.606 against perfect play with no experience
of the game at all. What crosses over is not tactics but a sense of what a
position is worth: blocking matters, the centre matters, a line you are one
move from completing matters.

**It buys a head start, not a higher ceiling.** By 500 games the two curves
meet exactly — 0.956 and 0.956, then 0.967 and 0.967. Prior experience matters
most where experience is scarce, and stops mattering once the game has been
played enough to teach everything itself.

**The from-scratch learner gets *worse* before it gets better.** 0.667 down to
0.523 at 25 games; 0.633 down to 0.472 at 100. Early on it has opinions built
from almost nothing, and they are worse than having no opinions at all. The
agent arriving with prior experience never dips.

### What actually made this possible

Not the complexity of the games. The tabular learner cannot transfer anything
between them no matter how much it plays, because a position is only ever
itself. What crosses the gap is having a way to describe a position that means
the same thing in both games — and a person wrote that vocabulary. The honest
reading is that transferable experience is a property of the representation,
not a reward for playing harder games.

## Poker, where winning stops being the measure

Every rung up to here could be scored by playing: win rate, or chips per hand.
Kuhn poker breaks that, and the break is worth showing in full.

| agent | vs random | vs the equilibrium | **exploitability** |
|---|---|---|---|
| `random` | −0.000 | −0.166 | 0.469 |
| `qlearner` | +0.086 | −0.005 | 0.250 |
| `mcts:200` | **+0.243** | −0.018 | 0.164 |
| `cfr` | +0.151 | +0.003 | **0.002** |
| `nash-kuhn` | +0.169 | −0.005 | **0.000** |

Chips per hand, seats alternating. Exploitability is what a best response that
knows your strategy takes off you per hand; zero means unexploitable, and it is
computed exactly, not sampled.

Read the first column and MCTS is the best poker player here — it beats a random
opponent harder than the game-theoretic optimum does. Read the last column and
it is losing 0.16 a hand to anyone who studies it. **The ranking by win rate is
close to the reverse of the ranking by exploitability.**

The middle column is the other trap: against the equilibrium, everyone from the
Q-learner to CFR lands within ±0.02 of the game value. An unexploitable opponent
does not punish you, it just declines to lose — so playing against one cannot
tell a badly leaking strategy from a sound one.

### Why Q-learning cannot get there

Its whole method is to find the best action and prefer it more. Here is the
strategy that produces after 300,000 hands of self-play, next to CFR's:

```
                 qlearner          cfr
J, first to act   check 1.00        bet 0.22 / check 0.78
K, first to act   check 1.00        bet 0.66 / check 0.34
Q, facing a bet   fold  1.00        call 0.34 / fold 0.66
```

Every Q-learner line is 1.00. It never bluffs, and it folds a Queen to every
bet — so an opponent can bet every hand and take the pot. That is not a
training-budget problem. In a game whose equilibrium is mixed, any method that
keeps sharpening toward one action per spot is building something readable, and
anything readable is exploitable. Across rounds its exploitability bounces
between 0.33 and 0.17 rather than falling: it is chasing a best response to its
own last strategy, in a circle.

CFR's numbers are not arbitrary either. Kuhn solved this game in 1950: the
first player's equilibria form a family parameterised by α ∈ [0, ⅓], betting a
Jack with probability α, a King with **3α**, and calling with a Queen at
**α + ⅓**. CFR arrived at α = 0.22, 3α = 0.66, α + ⅓ = 0.56 — the family,
rediscovered from regret alone.

### What had to change to make any of this honest

Poker is the first game here where a state holds something the player to move
must not see, and the existing agents did not know that. Measured before the
fix, minimax and MCTS each won about 0.25 chips a hand off a random opponent —
by reading its card.

- `Game.key` is now explicitly the *information set*: what the player to move
  is entitled to know. Kuhn's returns that player's own card and the betting,
  and nothing else.
- `Game.redeal` hands a searcher a world consistent with what it actually
  knows. MCTS now resamples the opponent's card every simulation instead of
  descending from the truth.
- `minimax` refuses imperfect-information games outright. There is no depth
  limit that fixes seeing the opponent's hand.

MCTS still cannot bluff after all that, and it is worth being clear about why:
inside any single sampled world the search can see both hands, so it plans as
though its own were public. It never finds a reason to represent a hand it does
not have. That is a known ceiling of determinized search, not a tuning problem.

## How it fits together

```
core/game.py        the Game interface — one abstraction for 1- and 2-player,
                    deterministic and stochastic, perfect- and
                    imperfect-information games
core/episode.py     runs one episode and hands each seat its own transitions
agents/             random, tabular Q-learning, linear function approximation,
                    MCTS, alpha-beta, CFR, human
training/curriculum.py   the ladder, the mastery gates, the promotion logic
training/exploitability.py  exact best-response analysis, for the poker rung
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
python -m gaming_agents exploit --game kuhn --agent cfr --solve 50000
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

- **Learned features.** `Game.features` closed the gap on Connect Four, but a
  person wrote those features. A network that learns the description from the
  board itself would remove the last hand-built piece — and would give 2048,
  still stuck at 1,219, its first real shot.
- **Learning from search.** MCTS is already the strongest thing here. Train the
  learner on its move choices and you have the outline of AlphaZero.
- **An LLM agent.** `Agent` needs one method, and every game renders to text
  with legal moves listed. `agents/llm.py` would drop straight into the same
  ladder and tournaments.
- **Bigger poker.** Leduc hold'em is the usual next step: two betting rounds and
  a shared card. `Game.initial_outcomes` can carry its community card dealt but
  hidden, so the exploitability machinery would work unchanged — but the
  brute-force best response would not, and would need the real algorithm.
