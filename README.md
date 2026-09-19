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
| 2048 (score) | 1,112 | 1,219 memorising² — **8,110 describing**⁶ | 11,326 |
| Kuhn poker (exploitability) | 0.469 | 0.250 | 0.164 — and see below⁴ |

¹ The (1,3,5,7) opening is a *loss* for whoever moves first, and evaluation
alternates seats. Winning exactly half is what perfect play looks like here;
it is the reason the learner is judged against `random` on this rung.

² A table keyed on exact 4×4 boards will essentially never see the same board
twice, so fifteen thousand games of experience buy almost nothing. 2048 is
where tabular learning ran out of road, and the curriculum said so instead of
pretending otherwise:

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

Now every game speaks a vocabulary defined once, on `Game.UNIVERSAL_FEATURES`:

```
bias            always 1; the baseline value of being in this game
progress        how far through the episode, 0 at the start, 1 at the end
my_strength     how good this position is for me, 0 hopeless to 1 winning
their_strength  the same for the opponent, as far as it is visible to me
win_available   1 when I can finish this right now
must_block      1 when I lose shortly unless I do something about it
```

Narrower ideas are namespaced — `board:centre_mine` for things only board games
have, `tictactoe:corners_mine` for things only one game has — so they cannot
collide once `LinearAgent(shared=True)` pools every game's weights into one set.

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

### The game where memorising cannot work at all

Connect Four showed describing a position beating memorising it. 2048 is the
case where memorising is not merely inefficient but impossible: a 4×4 board
essentially never comes round twice, so a lookup table has nothing to look up.

Same agent, same one-move lookahead, same 300 games. The only difference is
whether the board is named or described.

| | what it stores | mean score |
|---|---|---|
| random play | — | 1,112 |
| untrained, lookahead only | nothing | 2,253 |
| tabular learner, 15,000 games | 245,000 positions | 1,219 |
| **linear, exact board** (control) | **50,754 weights** | **1,970** |
| **linear, described** | **9 weights** | **8,110** |
| MCTS at 20 simulations | nothing, but searches every move | 11,326 |

Memorising did not merely fail here. At 1,970 it finished *below* the 2,253 an
untrained agent scores on the lookahead alone — fifty thousand numbers of
experience, and it would have done better having learned nothing. Nine numbers
took the same agent to 8,110, and to the 1024 tile in ten games out of
twenty-five.

That is the strongest form of the project's central claim. The curriculum rung
that nothing had ever cleared now clears on its first round, and the searching
reference it is measured against is barely ahead:

```
$ python -m gaming_agents curriculum --curriculum solo --agent linear --scale 0.06

=== stage 2/2: 2048 ===
    round 1: 300 episodes  mean_outcome=9167.6 (bar 3000)
    for reference, mcts:20 scores: 5 games, mean score 9,751.2
    -> mastered 2048; promoting
```

Nine weights and 300 games land within 6% of a search that runs twenty
simulations at every one of six hundred moves. Blackjack, in the same run,
still misses its bar at −0.094 — the two solo rungs now fail and pass in
opposite directions for the same reason.

It also fills in the last corner of a grid the earlier rungs mapped out:

| | small state space | huge state space |
|---|---|---|
| **deterministic** | Nim, Tic-Tac-Toe: both work | Connect Four: **describing wins**, 100 games against 80,000 |
| **stochastic** | Blackjack: **memorising wins**, −0.052 against −0.10 | 2048: **describing wins**, 8,110 against 1,970 |

Describing a position pays when there are too many positions to hold and the
value moves roughly in step with the description. It costs when the positions
are few enough to hold exactly and the right answer is not a straight line
through them — Blackjack has 280 decision points and a policy that reverses
direction twice across the dealer's upcard, and no set of weights can say that.

### It did not learn the strategy a person would

The nine weights are worth reading, with one caveat: a constant feature cannot
change which move looks best, so the large `bias` term is an offset that plays
no part in how it acts. What drives the play is the rest.

```
2048:mergeable  +1.367     must_block      -0.354
2048:ordered    +0.525     progress        +0.273
my_strength     -0.626     2048:cornered   -0.116
```

Textbook 2048 says: keep the board empty, and park your biggest tile in a
corner. This agent learned close to the opposite — it puts its strongest weight
on *having merges available* and a negative one on empty space, and it ignores
corners entirely. It plays a packed board full of matching neighbours rather
than a sparse one with an anchored king tile.

It reaches 1024 that way. Two honest caveats: the features are correlated, so
individual weights divide the credit between them somewhat arbitrarily, and the
strategy it found is worse than the best known human one. But it is not a
degenerate policy, and nobody told it to play that way.

### Does it cross between *kinds* of game?

The board games transfer to each other. Blackjack and Kuhn poker were given
descriptions in the same universal vocabulary, so the question could be put
properly: does mastering the simple board games help an agent take up cards?

There is a control in here worth noticing. "After Nim" is the same volume of
prior play in a game described only by lookup key, so it shares no concepts at
all — it separates *prior play helps* from *prior concepts help*.

**Blackjack** (perfect play is −0.0466 a hand, random is −0.39):

| hands of blackjack played | 0 | 100 | 1,000 | 10,000 |
|---|---|---|---|---|
| from scratch | −0.087 | −0.098 | −0.109 | −0.102 |
| after Nim (no shared concepts) | −0.087 | −0.107 | −0.103 | −0.098 |
| after board games | −0.086 | −0.081 | −0.107 | −0.098 |

**Kuhn poker**, chips per hand against a random player:

| hands of poker played | 0 | 100 | 1,000 | 10,000 |
|---|---|---|---|---|
| from scratch | 0.309 | 0.309 | 0.323 | 0.253 |
| after Nim | 0.309 | 0.297 | 0.319 | 0.315 |
| after board games | 0.281 | 0.304 | 0.316 | 0.278 |

**Nothing transfers.** Every row sits inside every other row's noise, and the
board-game prior is no better than the Nim control that shares no concepts
whatsoever.

That is the more interesting half of the transfer story. Board to board moved a
Tic-Tac-Toe agent from 0.150 to 0.606 against perfect play with no experience
of the game. Board to cards moves nothing, *with the vocabulary deliberately
made to match*. Naming two things "my_strength" does not make them the same
thing:

- On a board, value rises roughly in step with how close you are to a line, so
  a weight learned on one board reads correctly on another.
- In Blackjack the right play is not monotonic in the dealer's strength at all —
  hard 12 hits against a 2 or 3, stands against 4 through 6, hits again from 7
  up. No single weight on `their_strength` can say that.
- In poker the right answer is a *mixture*, not a function of hand strength.
  No amount of position-value thinking ever produces a bluff.

A shared vocabulary is necessary for transfer and nowhere near sufficient. What
has to match is the shape of the relationship underneath the words.

### A hole card nobody had looked under

Putting a searching agent on Blackjack for the first time turned up a bug that
had been sitting there since the game was written. `LinearAgent` values a move
by stepping the game forward — and stepping Blackjack forward plays the dealer
out from a hand that includes the **hole card**, which the player is not
allowed to see. One sample is noisy; average enough of them and the average
*is* a measurement of the concealed card:

```
averaging  1 sample  per move: -0.0906
averaging  4 samples per move: -0.0175
averaging 16 samples per move: +0.0102
averaging 64 samples per move: +0.0270   <- better than perfect play
```

Nothing honest beats −0.0466. Blackjack is now marked
`imperfect_information = True` and has a `redeal` that keeps the upcard and the
player's hand and draws a fresh card underneath, so a searcher reasons about a
table it could actually be sitting at. `MCTSAgent` already respected that flag,
so it was quietly fixed by the same change. The regression test is the sharpest
one available: **score better than the house edge and you are cheating.**

The tabular learner was never affected — it only ever sees `key`, which
excludes the hole card and has a test saying so. This is the same class of bug
poker exposed, in a game nobody had thought of as hiding anything.

### What if one agent played everything and kept all of it?

Keeping everything sounds strictly better — more experience, nothing thrown
away. Three ways to spend the same budget across five games, two seeds each:

| | Tic-Tac-Toe | Connect 4 | Blackjack | Poker | 2048 |
|---|---|---|---|---|---|
| **specialist** (one agent per game) | **0.938** | **0.997** | **−0.114** | **0.257** | 8,318 |
| sequential (one pool, games in turn) | 0.819 | 0.900 | −0.342 | 0.244 | 8,931 |
| interleaved (one pool, games rotated) | 0.595 | 0.737 | −0.305 | 0.177 | 7,669 |

**Pooling everything makes it worse at almost everything.** Blackjack falls
from −0.114 to −0.342, which is very nearly the −0.39 of playing at random: all
of it gone. Tic-Tac-Toe drops from 0.94 to 0.60.

And interleaving — rotating the games so none is ever finished, which ought to
protect against forgetting what came earlier — is the *worst* of the three. It
does not preserve each game's lessons; it averages them into a compromise that
serves none of them. Playing the games in turn is better precisely because it
lets one game own the weights at a time.

The exception is 2048, which is unharmed. That turns out to be the clue.

### An agent that is never reset

Every condition above starts a fresh agent. That is itself a protection: a run
that goes badly is discarded and the next one begins clean. So here is one
agent with a continuous life — born once, saved to disk after every stage,
reloaded for the next, carrying everything forward including the stage we
already knew would damage it.

| stage of its life | Tic-Tac-Toe | Connect 4 | Blackjack | Poker | 2048 |
|---|---|---|---|---|---|
| born, never played anything | 0.643 | 0.787 | −0.110 | 0.289 | 2,162 |
| childhood: nim and noughts | 0.943 | 0.993 | −0.115 | 0.246 | 7,705 |
| connect four | **0.957** | 0.993 | −0.111 | 0.303 | 7,648 |
| blackjack | 0.800 | 0.983 | −0.116 | 0.187 | 5,197 |
| poker | 0.597 | 0.987 | −0.116 | **0.312** | 5,920 |
| 2048 | 0.947 | 0.987 | **−0.336** | 0.312 | 9,296 |
| the churn (all five, rotated) | 0.563 | 0.783 | −0.302 | 0.187 | 8,876 |
| back to blackjack | **0.443** | 0.983 | **−0.114** | 0.176 | 5,814 |
| back to the board | **0.940** | 0.983 | −0.114 | 0.217 | **9,806** |
| blackjack once more | 0.513 | **0.997** | −0.122 | 0.176 | 7,856 |
| *(specialists, for reference)* | *0.938* | *0.997* | *−0.114* | *0.257* | *8,318* |

Two things happen, and they are the same thing seen from either end.

**Nothing is permanently lost.** Blackjack collapsed to −0.336, within a
whisker of the −0.39 of playing at random, and one stage of going back to it
restored −0.114 — full specialist level. Tic-Tac-Toe fell to 0.443 and came
back to 0.940. There is no scar tissue; the damage is entirely recoverable.

**Nothing is permanently kept either.** Every recovery is paid for. Going back
to blackjack cost Tic-Tac-Toe 0.563 → 0.443. Restoring the board games cost
2048 nothing that time but cost poker. Returning to blackjack once more knocked
Tic-Tac-Toe down again to 0.513. The agent never accumulates; it reallocates.

Read down the columns and you find something worth sitting with: at some point
in its life it matched or beat the specialist at *every one of the five games*.
It was simply never good at all of them at the same time. (With the caveat that
blackjack's best figure is at birth — the linear agent never learns that game
at all, so its "best" only means least damaged.)

So the answer to "keep everything, never reset" is that you get an agent that
oscillates rather than one that grows. And the reason is not that resetting was
protecting it from failure. It is that ten shared weights have no room for five
games. A person can hold chess and poker and driving at once because there is
capacity for all three; this agent has to choose, every time, and whatever it
learned last wins.

That reframes what the resets were doing. They were not sparing the agent its
mistakes — mistakes turn out to be entirely survivable. They were sparing it a
competition for space it cannot win. The fix for that is not a gentler
curriculum, it is more room.

### The multi-sport childhood

The previous section makes generalists look like a bad idea: one agent holding
five games is worse at almost all of them. But that agent never *stopped* being
a generalist. Children who play several sports do not keep playing all of them
for ever — they branch, and funnel into one.

So: a shared childhood, then a fork. Each branch specialises in one game. Both
conditions get the same total number of training episodes, which means the
multi-sport agent **practises its main game half as much**. That cost is the
whole test.

| main game | early specialist | multi-sport | |
|---|---|---|---|
| Tic-Tac-Toe | 0.940 | 0.938 | level |
| Connect Four | 0.994 | 0.994 | level |
| Blackjack | −0.114 | −0.113 | level |
| Poker | 0.252 | **0.295** | **multi-sport** |
| 2048 | 8,325 | **9,364** | **multi-sport** |

**The multi-sport agent never loses.** On half the practice at its own game it
draws level everywhere and wins outright on the two games with room to win in.
Poker by 17%, 2048 by 12%, both on seeds that agree.

The three level results are ceilings and floors rather than real ties: the
board games are both at the top of what this agent can do, and Blackjack is at
the bottom — the linear agent never learns that game at all, so both conditions
sit on the lookahead floor. Where there was headroom, the varied childhood won;
where there was none, it cost nothing.

### Why the branch is the whole difference

Set the two experiments side by side and they look contradictory. Pooling five
games makes an agent worse at almost all of them; a pooled childhood makes it
better. The difference is entirely in what happens next.

The continuous-life run showed that damage from another game is cheap to undo —
one stage of going back restored full specialist level from a position that was
nearly random. What it did *not* show was any lasting benefit, because that
agent never stopped being pulled in five directions.

Branching resolves it. The childhood leaves the weights somewhere sensible
rather than at zero: a rough sense of what a strong position is, what an
immediate threat looks like, that being one move from losing is bad. The
specialisation that follows is then free to move those weights wherever that
one game needs them, and it gets there faster than it would have from nothing.
The detour costs nothing because the interference washes out; the structure
survives because it was roughly right to begin with.

Jack of all trades and master of none is what you get if you never stop.
Stop at the right time and the trades were free.

*(Two seeds each. The 2048 column swings by thousands of points between seeds,
so read the margin rather than the absolute figures.)*

### Why: the same word, opposite meanings

The weight each game arrives at for the *same named concept*, trained
separately:

| | Tic-Tac-Toe | Connect 4 | Blackjack | 2048 |
|---|---|---|---|---|
| `my_strength` | +0.225 | +0.028 | **+0.852** | +0.012 |
| `their_strength` | −0.231 | −0.166 | −0.352 | **+0.379** |
| `must_block` | +0.071 | −0.028 | **+0.402** | **−0.287** |
| `progress` | −0.248 | +0.190 | +0.306 | **+1.131** |

Three of the four change sign between games, and the vocabulary is exactly
where the conflict lives.

`progress` means "the board is filling and a draw is coming" in Tic-Tac-Toe and
"my biggest tile is growing" in 2048 — the same word, worth −0.25 in one game
and +1.13 in the other. `must_block` is a warning worth heeding in Blackjack
(+0.40) and, in 2048, a state the agent has learned to shrug at (−0.29).

Put those in one place and they fight. 2048 wins, because its weights are
several times larger than anyone else's, which is exactly why 2048 is the one
game pooling does not damage. Blackjack loses hardest because its one strong
opinion, `my_strength` at +0.852, is outvoted by three games that barely care.

So the honest answer to "keep everything, no matter what" is that the agent
already keeps everything *within* a game — nothing is discarded, and a weight
only moves when something surprising happens. Keeping it across games is a
different proposition, and it costs more than it pays unless the games agree
about what their shared words mean. `shared=False` is the default for that
reason, and this is the measurement behind the default.

*(Two seeds, so treat the 2048 column as noise — its twelve evaluation games
per condition span thousands of points. The board and card columns are large
and consistent across both seeds.)*

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

- **Learned features.** `Game.features` closed the gap on Connect Four and on
  2048, but a person wrote those features every time, and the one game still
  unaddressed — chess, or anything with a screen — would need a new set again.
  A network that learns the description from the board itself is the only way
  off that treadmill, and it is now the last hand-built piece left.
- **Learning from search.** MCTS is already the strongest thing here. Train the
  learner on its move choices and you have the outline of AlphaZero.
- **An LLM agent.** `Agent` needs one method, and every game renders to text
  with legal moves listed. `agents/llm.py` would drop straight into the same
  ladder and tournaments.
- **Bigger poker.** Leduc hold'em is the usual next step: two betting rounds and
  a shared card. `Game.initial_outcomes` can carry its community card dealt but
  hidden, so the exploitability machinery would work unchanged — but the
  brute-force best response would not, and would need the real algorithm.
