# NuCS vs Choco: a pure-Python solver meets a JVM veteran

## TL;DR

[NuCS](https://github.com/yangeorget/nucs) is a constraint solver written 100% in Python, accelerated by
[NumPy](https://numpy.org/) and [Numba](https://numba.pydata.org/).
[Choco](https://github.com/chocoteam/choco-solver) is one of the reference open-source constraint solvers,
written in Java and developed for more than two decades.

Comparing them looks lopsided: an interpreted language against a heavily optimized JVM solver with a rich catalog of
arc-consistent global constraints. The reality is more interesting. When both solvers run the *same* model they are, for
all practical purposes, **the same speed** — and on the largest instances NuCS actually pulls *ahead*, because once
Numba has compiled the inner loops the Python tax is gone and only the cost-per-node remains. When the models differ the
result is a genuine trade rather than a rout: on some problems Choco's arc consistency is the right, fast tool; on others
NuCS's cheap bound consistency plus a little remodeling wins outright; and on at least one problem NuCS's modeling
freedom lets it solve instances that *neither* plain solver can.

This article walks through five benchmark problems, gives the exact NuCS command line for each, details the constraints
and search strategy, draws the performance curves, and ends on the design decision that explains the whole picture:
**NuCS represents domains as `min..max` intervals and is therefore limited to bound consistency, while Choco can also
represent domains with holes and run full arc consistency.**

All NuCS code shown here lives in the [NuCS repository](https://github.com/yangeorget/nucs) under the
[MIT license](https://github.com/yangeorget/nucs/blob/main/LICENSE.md).

---

## NuCS

### History

NuCS is a young project. Its first public releases date from 2024, and it has iterated very quickly since — the version
benchmarked here is **11.2.0**. It was built around a deliberately unusual bet: write a competitive finite-domain
constraint solver **entirely in Python**, and recover the performance normally lost to interpretation through
just-in-time compilation rather than a C or C++ core. It is distributed on PyPI (`pip install nucs`), which keeps
installation as simple as any other Python package — no native toolchain, no JVM.

### Architecture

A `Problem` carries a NumPy array of variable domains — one `(min, max)` pair per variable, a variable being *bound*
when `min == max` — together with a list of *propagators*. A propagator is a constraint's filtering algorithm,
registered under a numeric `ALG_*` identifier. Each propagator is three small functions:

- `compute_domains_*` — the actual filtering, returning *inconsistency*, *consistency*, or *entailment*;
- `get_triggers_*` — which domain events should re-wake the propagator;
- `get_complexity_*` — a cost estimate used to order the propagation queue.

The `BacktrackSolver` interleaves propagation-to-a-fixpoint with a branching decision, chosen by a *variable heuristic*
(which unbound variable to branch on) and a *domain heuristic* (which value to try). A `MultiprocessingSolver` fans
several backtracking searches out over a split of the search space.

What makes this fast despite being Python is that essentially every hot function is decorated with
`@njit(cache=True, fastmath=True)`. Numba compiles these to native code on first use and caches the result on disk, so a
*warm* process runs compiled machine code, not interpreted bytecode. Domains are plain NumPy arrays mutated in place,
and propagators exchange typed `NDArray`s and integers — never Python objects. The price is a cold-start compilation
(mitigated by the cache) and a coding discipline inside jitted code: no dictionaries, no exceptions, no `isinstance`, no
strings. The payoff is twofold: C-like throughput, and **cheap, fast modeling** — adding a redundant constraint,
swapping a heuristic, or writing a problem-specific consistency algorithm is a handful of lines of ordinary Python. As
we will see, that second property is where several of NuCS's wins come from.

The single most consequential design choice is that a domain is *always an interval*. There is no representation for a
hole in the middle of a domain, which is exactly what makes the whole engine expressible as operations on two NumPy
arrays — and what limits NuCS to **bound consistency**. We will return to this at the end; it is the thread running
through every result below.

---

## Choco

### History

Choco is a veteran of the constraint-programming world. It first appeared around the turn of the 2000s and has been
through several full rewrites. The modern lineage — **Choco 3, then 4, and now 6** — is a ground-up redesign developed
largely at IMT Atlantique (the TASC / LS2N research group in Nantes) by Charles Prud'homme, Jean-Guillaume Fages and
contributors. It is an open-source Java library, distributed under a BSD license, and is widely used in both research
and industry. The version benchmarked here is **6.0.1**, a current release.

### Architecture

Choco is an event-based constraint solver on the JVM. A `Model` holds integer, boolean, set and real variables together
with `Constraint`s; each constraint delegates filtering to one or more *propagators* driven by a fine-grained
event/propagation engine. On top sits a configurable search loop with a large library of branching strategies, restart
policies, and — historically a distinctive feature — **explanations** (conflict-based learning).

Where Choco fundamentally differs from NuCS is in its **domain representation**: it supports both *bounded* domains
(intervals, like NuCS) and *enumerated* domains (bitsets), which can represent arbitrary subsets — domains *with holes*.
That richer representation is what unlocks the **breadth and strength of its global-constraint catalog**: `allDifferent`
at several consistency levels (including Régin's arc-consistent algorithm), cardinality and counting constraints,
`cumulative`, `circuit`, automaton/regular constraints, and many more, most backed by carefully engineered, often
**arc-consistent**, filtering.

A mature JVM solver brings two things a young Python project cannot match today: a JIT (HotSpot) tuned over twenty
years, and a deep library of strong global constraints you can drop into a model and trust. The flip side is that those
strong filters are also where Choco can become *expensive*: arc-consistent global constraints do a lot of work per node,
and that work does not always pay for itself.

---

## The comparison setup

| Component | NuCS side                  | Choco side                    |
|-----------|----------------------------|-------------------------------|
| Solver    | NuCS **11.2.0**            | choco-solver **6.0.1**        |
| Runtime   | CPython + Numba **0.65.1** | Java **26.0.1** (HotSpot JVM) |
| Numerics  | NumPy **2.4.6**            | —                             |

A few honest caveats before reading any number:

- All timings are in **milliseconds**, measured on the same machine. Cross-language micro-benchmarks are always
  approximate; treat *ratios and curve shapes*, not absolute values, as the signal.
- NuCS timings are taken **after** the Numba JIT cache is warm — the one-off compilation cost is not what we want to
  measure. Even warm, NuCS pays a small **fixed startup cost** of a few hundred milliseconds (reading the cache, warming
  NumPy) that you can see as a floor on the smallest instances. It is paid once per process, not per search node, so it
  is noise for a long solve and dominant for a tiny one.
- Every NuCS run uses the cache directory and the same options pattern; the leading
  `NUMBA_CACHE_DIR=.numba/cache` and the trailing
  `--no-display-solutions --no-display-stats --log-level ERROR` (which just silence output) are elided from the command
  lines below for readability.
- Each problem shows the **NuCS command line**, the **constraints and search strategy**, a **curve diagram**
  (lower is better), the raw table, and a short analysis. In any "speedup" column, a value greater than 1 means
  **NuCS is faster** by that factor.

The five problems split naturally into two groups: those where **both solvers run essentially the same model** (so we
are comparing engines) and those where **the models differ** (so we are comparing modeling strategies as much as
engines).

---

## Group 1 — same model, comparing engines

Here both solvers post the same constraints with the same consistency level and a comparable search strategy. The only
thing under test is the engine.

### All-interval series (find one solution)

This is CSPLIB #7: find a permutation of `0..n-1` whose `n-1` consecutive absolute differences are themselves all
different. Both solvers use bound consistency and a first-fail (smallest-domain) heuristic on the same formulation.

**NuCS command line:**

```bash
python -m nucs.examples.all_interval_series \
  -n 500 --var-heuristic 3 --symmetry-breaking --cp-max-height 10000
```

**Constraints and strategy.** For each `i`, a `SUM_EQ` propagator defines `diff_i = x_{i+1} - x_i` and an `ABS_EQ`
propagator defines `|diff_i|`; two `ALLDIFFERENT`s enforce that the values and the absolute differences are each a
permutation; two `LEQ_C` propagators break the obvious reversal/complement symmetries. `--var-heuristic 3` selects the
**smallest-domain** variable heuristic (first-fail), branching only on the `n` series variables (`decision_variables`).

```python
for i in range(n - 1):
    self.add_propagator(ALG_SUM_EQ, [n + i, i, i + 1])       # diff_i = x_{i+1} - x_i
    self.add_propagator(ALG_ABS_EQ, [n + i, 2 * n - 1 + i])  # |diff_i|
self.add_propagator(ALG_ALLDIFFERENT, range(n))
self.add_propagator(ALG_ALLDIFFERENT, range(2 * n - 1, 3 * n - 2))
if symmetry_breaking:
    self.add_propagator(ALG_LEQ_C, [0, 1], [-1])
    self.add_propagator(ALG_LEQ_C, [3 * n - 3, 2 * n - 1], [-1])
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#1f77b4, #ff7f0e"}}}}%%
xychart-beta
    title "All-interval series — time (ms) vs size"
    x-axis "series size n" [500, 1000, 2000, 4000, 8000, 16000]
    y-axis "time (ms)" 0 --> 121000
    line [240, 392, 950, 3261, 16595, 120340]
    line [225, 398, 972, 3352, 15153, 85236]
```

_Series order: **Choco** (blue), **NuCS** (orange). Lower is better._

| size  | Choco (BC, ff) | NuCS (BC, ff) | speedup |
|-------|---------------:|--------------:|--------:|
| 500   |            240 |           225 |   1.07× |
| 1000  |            392 |           398 |   0.98× |
| 2000  |            950 |           972 |   0.98× |
| 4000  |           3261 |          3352 |   0.97× |
| 8000  |          16595 |         15153 |   1.10× |
| 16000 |         120340 |         85236 |   1.41× |

This is the most telling chart in the whole benchmark *because nothing differs except the engine* — and for the first
five sizes the two curves sit on top of each other (within ~3%). Then at n=16000 they separate: Choco takes 120 s, NuCS
85 s — **NuCS is 1.41× faster**. A pure-Python solver not merely matching but *beating* Choco on an identical model is
the headline result: once Numba has compiled the propagators, NuCS's vectorized per-node work is genuinely competitive,
and on the biggest instance its lower constant factor wins.

### Schur's lemma (prove no solution)

CSPLIB #15: partition `1..n` into 3 sum-free sets — here proven infeasible, so the *entire* search tree must be
explored, which makes this a clean test of raw propagation throughput. Both solvers use bound consistency, and to keep
the models identical the run is done **without symmetry breaking**.

**NuCS command line:**

```bash
python -m nucs.examples.schur_lemma \
  -n 100 --no-symmetry-breaking
```

**Constraints and strategy.** Each of the `n` numbers gets three 0/1 membership variables; a `SUM_EQ_C` forces it into
exactly one set; for every valid triple `(x, y, z = x+y+1)` a `SUM_LEQ_C` forbids all three being in the same set
(the sum-free condition). With `--no-symmetry-breaking` the `LEXICOGRAPHIC_LEQ` ordering constraint is dropped so the
model matches Choco's. The default variable heuristic drives the search.

```python
for x in range(n):
    self.add_propagator(ALG_SUM_EQ_C, [x * 3, x * 3 + 1, x * 3 + 2], [1])
for k in range(3):
    for x in range(n):
        for y in range(n):
            z = x + y + 1
            if 0 <= z < n:
                self.add_propagator(ALG_SUM_LEQ_C, [3 * x + k, 3 * y + k, 3 * z + k], [2])
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#1f77b4, #ff7f0e"}}}}%%
xychart-beta
    title "Schur's lemma — time (ms) vs size"
    x-axis "problem size n" [100, 200, 400, 800, 1600]
    y-axis "time (ms)" 0 --> 18000
    line [197, 333, 702, 2701, 17864]
    line [418, 542, 1050, 3118, 14592]
```

_Series order: **Choco** (blue), **NuCS** (orange). Lower is better._

| size | Choco (BC) | NuCS (BC) | speedup |
|------|-----------:|----------:|--------:|
| 100  |        197 |       418 |   0.47× |
| 200  |        333 |       542 |   0.61× |
| 400  |        702 |      1050 |   0.67× |
| 800  |       2701 |      3118 |   0.87× |
| 1600 |      17864 |     14592 |   1.22× |

At the smallest size Choco is genuinely faster — about 2×. But watch the gap close monotonically as the tree grows:
0.47× → 0.61× → 0.67× → 0.87× → **1.22×**. That shrinking deficit is NuCS's roughly constant per-process overhead being
amortized against a growing absolute runtime; by n=1600 it has not just vanished, it has *reversed* — NuCS proves
infeasibility in 14.6 s against Choco's 17.9 s. NuCS pays a setup cost here, not an algorithmic one, and once the search
is large enough its cheaper per-node propagation comes out ahead.

**Takeaway for Group 1.** When the model is fixed, NuCS and Choco run at essentially the same asymptotic speed — and at
the largest sizes NuCS edges in front on *both* problems. The only visible disadvantage is a small, constant NuCS startup
cost that dominates tiny instances and is irrelevant on large ones. For a pure-Python solver against a two-decade-old
JVM engine, that is a remarkable result.

---

## Group 2 — different models, comparing strategies

On these three problems the two solvers do not run the same algorithm, so we are comparing *modeling strategies* as much
as engines. Choco reaches for strong, arc-consistent global constraints or enumerated domains; NuCS reaches for cheap
bound consistency plus redundant constraints, channeling, or a hand-written filter. The results are mixed — and that is
the honest, interesting part.

### Latin square (find one solution)

A Latin square is an `n×n` grid where every row and every column is a permutation of `0..n-1`. The straightforward
model, used by both solvers, posts a bound-consistent `allDifferent` per row and per column.

**NuCS command lines:**

```bash
# plain bound-consistent model
python -m nucs.examples.latin_square \
  -n 20 --cp-max-height 10000

# redundant (row/column) model
python -m nucs.examples.latin_square \
  -n 20 --model-rc --cp-max-height 10000
```

**Constraints and strategy.** The plain model is just `2n` `ALLDIFFERENT` propagators (one per row, one per column) over
the `n²` cell variables, with the default first-not-instantiated branching:

```python
for i in range(self.n):
    self.add_propagator(ALG_ALLDIFFERENT, self.row(i))
    self.add_propagator(ALG_ALLDIFFERENT, self.column(i))
```

The redundant model (`--model-rc`, class `LatinSquareRCProblem`) adds two extra *views* of the same grid — one indexed
by *(color, column) → row*, one by *(row, color) → column* — each with its own row/column `ALLDIFFERENT`s, and links the
three views with channeling (`PERMUTATION_AUX`) constraints, so that pruning found in one view immediately propagates to
the others:

```python
# row[c,j]=i  <=>  color[i,j]=c
self.add_propagator(ALG_PERMUTATION_AUX, [*self.column(j), *self.column(j, M_ROW)])
# row[c,j]=i  <=>  column[i,c]=j
self.add_propagator(ALG_PERMUTATION_AUX, [*self.row(c, M_ROW), *self.column(c, M_COLUMN)])
# color[i,j]=c  <=>  column[i,c]=j
self.add_propagator(ALG_PERMUTATION_AUX, [*self.row(i), *self.row(i, M_COLUMN)])
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#1f77b4, #ff7f0e, #2ca02c"}}}}%%
xychart-beta
    title "Latin square — time (ms) vs size"
    x-axis "square size n" [20, 30, 40, 50]
    y-axis "time (ms)" 0 --> 900
    line [105, 126, 180, 900]
    line [376, 380, 397, 900]
    line [412, 453, 547, 728]
```

_Series order: **Choco (BC)** (blue), **NuCS (BC)** (orange), **NuCS (BC + redundant)** (green). At n=50 the two
plain-BC lines leap to the top of the chart: this marks **"did not finish"**, not a measured time. NuCS with redundant
constraints is the only model that completes n=50 — in 728 ms. Lower is better._

| size | Choco (BC)       | NuCS (BC)        | NuCS (BC + redundant) |
|------|-----------------:|-----------------:|----------------------:|
| 20   |              105 |              376 |                   412 |
| 30   |              126 |              380 |                   453 |
| 40   |              180 |              397 |                   547 |
| 50   | ✗ did not finish | ✗ did not finish |                   728 |

Two things to read here. First, on the *same* plain-BC model, Choco is faster than NuCS on the small solvable
instances — the familiar constant-overhead gap, with NuCS sitting on its ~376 ms floor while Choco starts near 105 ms.
Second, and more important: **both plain-BC models fall off a cliff at order 50** and fail to finish. The plain
formulation simply does not prune enough, and that is an algorithmic wall, not an engine one — Choco hits it too.

The NuCS redundant model walks straight past that wall. Its curve is almost flat (412 → 728 ms while the order grows
from 20 to 50) because the channeled views prune so much that search barely expands. The win here is not a tidy speedup
ratio; it is **qualitative** — NuCS solves, in well under a second, an instance that neither plain-BC solver can finish
at all. And the thing that made it possible is not the engine but the *ease of remodeling*: a couple of extra propagator
loops in ordinary Python.

### Magic sequence (find one solution)

A magic sequence (CSPLIB #19) is a sequence `x_0, …, x_{n-1}` where each `x_i` equals the number of occurrences of the
value `i` in the sequence. Choco's natural model uses a **strong arc-consistent global constraint**. NuCS uses only
simple `count` constraints plus redundant linear constraints.

**NuCS command lines:**

```bash
# count + one redundant linear constraint
python -m nucs.examples.magic_sequence \
  -n 100 --var-heuristic 3 --model-r1

# adds a second, weighted-sum redundant constraint
python -m nucs.examples.magic_sequence \
  -n 100 --var-heuristic 3 --model-r1 --model-r2
```

**Constraints and strategy.** For each `i`, a `COUNT_EQ` propagator ties `x_i` to the number of cells equal to `i`. The
first redundant constraint (`--model-r1`, a `SUM_EQ_C`) states that the values must sum to `n`; the second
(`--model-r2`, an `AFFINE_EQ`) is a weighted-sum identity `Σ i·x_i = n`. Both runs use the smallest-domain (first-fail)
heuristic via `--var-heuristic 3`.

```python
for i in range(n):
    self.add_propagator(ALG_COUNT_EQ, list(range(n)) + [i], [i])
if model_r1:
    self.add_propagator(ALG_SUM_EQ_C, range(n), [n])            # redundant: values sum to n
if model_r2:
    self.add_propagator(ALG_AFFINE_EQ, range(n), range(n + 1))  # redundant: weighted-sum identity
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#1f77b4, #ff7f0e, #2ca02c"}}}}%%
xychart-beta
    title "Magic sequence — time (ms) vs size"
    x-axis "sequence size n" [100, 200, 300, 400]
    y-axis "time (ms)" 0 --> 3000
    line [149, 307, 779, 1783]
    line [416, 710, 1451, 2913]
    line [387, 455, 624, 942]
```

_Series order: **Choco (AC)** (blue), **NuCS (BC, r1)** (orange), **NuCS (BC, r1 + r2)** (green). Lower is better._

| size | Choco (AC) | NuCS (BC, r1) | NuCS (BC, r1 + r2) | speedup vs Choco |
|------|-----------:|--------------:|-------------------:|-----------------:|
| 100  |        149 |           416 |                387 |            0.38× |
| 200  |        307 |           710 |                455 |            0.67× |
| 300  |        779 |          1451 |                624 |            1.25× |
| 400  |       1783 |          2913 |                942 |            1.89× |

This is the most dramatic crossover of the set. Choco's arc-consistent filtering is excellent on small instances: at
n=100 it is ~2.5× faster than the best NuCS model. But its curve climbs *much* faster — Choco grows ~12× from n=100 to
n=400 (149 → 1783 ms), while the two-redundant NuCS model grows only ~2.4× (387 → 942 ms). The lines **cross around
n=250**, and by n=400 NuCS is **1.89× faster** — 942 ms against Choco's 1783 ms. The second redundant constraint is
what does it: comparing the orange and green curves, adding `Σ i·x_i = n` cuts the n=400 time from 2913 ms to 942 ms.
The reading is not "AC is wasteful" — it clearly pays on small instances — but "cheap BC plus the right pair of
redundant constraints has a far flatter cost-per-node, and overtakes decisively as the instance grows."

### Golomb ruler (minimize)

Golomb (CSPLIB #6) is the most interesting comparison because it leans hardest on domain representation: find `n` marks
`0 = m_0 < m_1 < … < m_{n-1}` whose pairwise distances are all different, minimizing `m_{n-1}`. Choco's sample uses
**enumerated domains** — effectively stronger consistency — which lets it prune values in the *middle* of the distance
intervals. Plain NuCS bound consistency cannot do that, and it shows. So NuCS offers a **problem-specific consistency
algorithm**: a few dozen lines of jitted Python (`golomb_consistency_algorithm`) that pre-prune the distance variables
with a minimal-sum-of-distinct-integers argument before running standard bound consistency.

**NuCS command lines:**

```bash
# plain bound consistency (--consistency-algorithm 0 forces BC)
python -m nucs.examples.golomb \
  -n 9 --symmetry-breaking --consistency-algorithm 0

# custom Golomb consistency algorithm (the example's default)
python -m nucs.examples.golomb \
  -n 9 --symmetry-breaking
```

**Constraints and strategy.** Distance variables `dist_ij` are tied by `SUM_EQ` (`dist_ij = m_j − m_i`), a single
`ALLDIFFERENT` enforces all distances distinct, redundant `LEQ_C` bounds relate each distance to the ruler length, and a
`LEQ_C` breaks the mirror symmetry. The objective `m_{n-1}` is minimized with `solver.minimize(...)`. The custom variant
swaps the plain bound-consistency algorithm for `golomb_consistency_algorithm`, which tightens the distance lower bounds
before delegating to `bound_consistency_algorithm`; `--consistency-algorithm 0` overrides that back to plain BC for the
comparison.

```python
for i in range(1, mark_nb - 1):
    for j in range(i + 1, mark_nb):
        self.add_propagator(ALG_SUM_EQ, [index(mark_nb, 0, i), index(mark_nb, i, j), index(mark_nb, 0, j)])
self.add_propagator(ALG_ALLDIFFERENT, range(self.domain_nb))
# redundant length bounds + symmetry breaking omitted for brevity
```

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#1f77b4, #ff7f0e, #2ca02c"}}}}%%
xychart-beta
    title "Golomb ruler — time (ms) vs number of marks"
    x-axis "marks" [9, 10, 11, 12]
    y-axis "time (ms)" 0 --> 130000
    line [202, 481, 6800, 65705]
    line [438, 883, 12428, 128040]
    line [414, 641, 6791, 67319]
```

_Series order: **Choco (enumerated domains)** (blue), **NuCS BC** (orange), **NuCS custom consistency** (green).
Lower is better._

| marks | Choco (enum. domains) | NuCS (BC) | NuCS (custom consistency) |
|-------|----------------------:|----------:|--------------------------:|
| 9     |                   202 |       438 |                       414 |
| 10    |                   481 |       883 |                       641 |
| 11    |                  6800 |     12428 |                      6791 |
| 12    |                 65705 |    128040 |                     67319 |

Plain NuCS BC (orange) is about **2× slower** than Choco across the board — exactly the penalty you would expect when
the opponent can prune holes you cannot. But the custom-consistency variant (green) closes the gap almost perfectly: at
11 marks, 6791 ms vs Choco's 6800 ms; at 12 marks, 67319 ms vs 65705 ms — within ~2%. NuCS does not need enumerated
domains to recover the missing pruning; it needs a **dedicated propagator** that encodes the same logic. The cost is
developer effort, not solver capability.

---

## What the numbers say

| Problem             | Models    | Outcome at scale                            | Margin                       |
|---------------------|-----------|---------------------------------------------|------------------------------|
| All-interval series | same (BC) | NuCS, increasingly                          | 1.41× at n=16000             |
| Schur's lemma       | same (BC) | NuCS overtakes                              | 1.22× at n=1600              |
| Magic sequence      | different | Choco small, NuCS wins big at scale         | 1.89× at n=400               |
| Golomb ruler        | different | Choco; tie with NuCS custom propagator      | ~2× plain, ~1× with custom   |
| Latin square        | different | NuCS solves what plain BC cannot            | qualitative (feasibility)    |

- **Equal models, equal speed — then NuCS edges ahead.** When both solvers run the same algorithm, NuCS matches Choco
  through the mid sizes and *wins on the largest instances of both problems* (all-interval series and Schur's lemma). The
  only visible gap is a small, constant NuCS startup overhead that vanishes as the search grows.
- **Stronger consistency often pays — but its cost-per-node grows.** On magic sequence Choco's arc-consistent global
  constraint is faster on small instances, yet NuCS's lightweight BC plus two redundant constraints has a far flatter
  slope and is nearly 2× faster by n=400.
- **Modeling freedom can change feasibility, not just speed.** On Latin square the plain BC model walls out at order 50
  for *both* solvers; NuCS's redundant channeling model solves it in 728 ms — because remodeling in Python is cheap.
- **Sometimes AC really is decisive.** On Golomb ruler, enumerated domains give Choco pruning that plain BC simply
  cannot reproduce — and NuCS only catches up by writing a custom propagator.

---

## The root cause: domains and consistency

Step back from the individual problems and one design decision explains the whole picture.

**NuCS represents each variable domain as a single `min..max` interval.** A domain is two integers; binding a variable
means `min == max`. This is compact, cache-friendly, and trivially vectorizable with NumPy — which is precisely what
lets Numba make the propagators so fast. But it has a hard consequence: NuCS can only *shrink the bounds* of a domain. It
cannot punch a hole in the middle. The strongest filtering it can express is therefore **bound consistency (BC)** — it
guarantees the endpoints of each domain are supported, nothing more.

**Choco represents domains with holes** (bitset / enumerated representations alongside bounded ones). It can remove an
arbitrary value from the middle of a domain. That is what makes **arc consistency (AC)** possible: a filtering algorithm
can delete *every* unsupported value, not just trim the extremes. Choco's catalog leans into this — `allDifferent` (AC),
arc-consistent cardinality and counting constraints, enumerated-domain propagation — and these are exactly the
constraints behind its results on magic sequence and Golomb above.

So the two solvers occupy genuinely different points in the design space:

|                       | NuCS                                                           | Choco                                          |
|-----------------------|----------------------------------------------------------------|------------------------------------------------|
| Domain representation | `min..max` interval                                            | values with holes (bitset/enumerated)          |
| Strongest consistency | bound consistency (BC)                                         | arc consistency (AC)                           |
| Best at               | tight bounded models, vectorized propagation, cheap remodeling | strong global constraints, problems needing AC |
| Cost model            | very low per-node cost                                         | higher per-node cost, fewer nodes              |

And here is the nuance the benchmarks reveal. NuCS's "weaker" consistency is rarely the disadvantage it looks like on
paper. Because BC propagators are so cheap, and because remodeling in Python is so easy, the productive move in NuCS is
to **recover the missing pruning with redundant constraints, channeling, or a custom filter** rather than with an
expensive AC algorithm. On magic sequence that lower per-node cost lets BC pass Choco's arc-consistent global constraint
decisively once the instance is large; on Latin square it is the difference between solving order 50 and not solving it
at all; and even on the *identical* models of Group 1, the lower constant factor is what puts NuCS ahead at the top end.

The flip side is just as real. When a problem genuinely needs values removed from the middle of domains, Choco's
arc-consistent globals and enumerated domains are the right tool out of the box — fast on small magic-sequence instances
and ahead on Golomb until NuCS answers with hand-written code. Choco gives you that strength for free; NuCS makes you
build it, but lets you build it in a few lines.

---

## Conclusion

Reading the benchmark as "which solver is faster" misses the point. Two different things are being measured:

1. **Same model, comparing engines (all-interval series, Schur's lemma).** The two are essentially tied through the mid
   range, separated only by a small constant NuCS startup cost — and on the *largest* instance of each problem NuCS
   pulls ahead (1.41× on all-interval at n=16000, 1.22× on Schur at n=1600). That a *pure-Python* solver runs within a
   hair of a mature, two-decade-old JVM solver and then *overtakes* it at scale is the most pleasantly surprising
   outcome. Numba does not just close the Python penalty; the resulting per-node cost is genuinely low.

2. **Different models, comparing strategies (magic sequence, Golomb ruler, Latin square).** Here it is a genuine trade.
   Choco's arc consistency is fast and decisive on some instances (small magic sequences, Golomb); NuCS's cheap BC plus
   redundant constraints has a flatter cost curve and wins big at scale on others (large magic sequences, ~1.9×),
   recovers parity where AC led (Golomb, with a custom propagator), and — thanks to how cheap remodeling is — solves a
   Latin square that *neither* plain-BC model can.

Underneath both observations lies one architectural fact. **NuCS stores domains as `min..max` intervals and is therefore
limited to bound consistency; Choco stores domains with holes and can run arc consistency.** That is the honest, lasting
difference between the two solvers. It is why Choco ships strong AC global constraints that NuCS cannot replicate
directly, and it is *also* why NuCS's cheap-BC-plus-redundant-constraints style can compete — and increasingly win —
when a good redundant model exists. Neither approach dominates; they are different bets.

For someone who wants Choco's breadth of battle-tested, arc-consistent global constraints, mature search infrastructure,
and explanations, Choco remains the reference. For someone who wants to **experiment with models in Python and still get
native-code speed**, NuCS makes a strong case: the language is no longer the bottleneck, and the freedom to reshape the
model — adding redundancy, channeling, or a custom BC filter in a few lines — often matters as much as raw engine speed
or raw consistency strength.

---

Some useful links to go further with NuCS:

* the source code: https://github.com/yangeorget/nucs
* the documentation: https://nucs.readthedocs.io/en/latest/index.html
* the Pip package: https://pypi.org/project/NUCS/
