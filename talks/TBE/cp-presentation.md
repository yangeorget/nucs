---
marp: true
theme: default
paginate: true
size: 16:9
title: Constraint Programming on Integers
author: Yan Georget
style: |
  section {
    font-family: 'Helvetica Neue', 'Arial', sans-serif;
    font-size: 24px;
    color: #1e293b;
    background: #fafafa;
    padding: 56px 72px;
  }
  section h1 {
    color: #0f172a;
    font-size: 40px;
    border-bottom: 3px solid #3b82f6;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }
  section h2 {
    color: #1e40af;
    font-size: 28px;
    margin-top: 0;
  }
  section.lead {
    text-align: center;
    justify-content: center;
    color: white;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
  }
  section.lead h1 {
    color: white;
    border-bottom: none;
    font-size: 60px;
  }
  section.lead h2 {
    color: #cbd5e1;
    font-weight: normal;
    font-size: 32px;
  }
  section.part1 { border-top: 8px solid #ef4444; }
  section.part2 { border-top: 8px solid #3b82f6; }
  section.part3 { border-top: 8px solid #10b981; }
  section.part4 { border-top: 8px solid #8b5cf6; }
  section.lead.part1 { background: linear-gradient(135deg, #7f1d1d 0%, #ef4444 100%); border-top: none; }
  section.lead.part2 { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-top: none; }
  section.lead.part3 { background: linear-gradient(135deg, #064e3b 0%, #10b981 100%); border-top: none; }
  section.lead.part4 { background: linear-gradient(135deg, #4c1d95 0%, #8b5cf6 100%); border-top: none; }
  code {
    background: #f1f5f9;
    color: #be185d;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.92em;
  }
  pre {
    background: #1e293b;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 18px;
    line-height: 1.45;
    box-shadow: 0 4px 8px rgba(0,0,0,0.12);
  }
  pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
  }
  blockquote {
    border-left: 5px solid #3b82f6;
    background: #eff6ff;
    padding: 10px 18px;
    margin: 14px 0;
    font-style: normal;
    color: #1e3a8a;
    border-radius: 0 6px 6px 0;
  }
  table {
    font-size: 22px;
    border-collapse: collapse;
    margin: 14px auto;
    width: 100%;
  }
  th {
    background: #1e3a8a;
    color: white;
    padding: 10px 16px;
    text-align: left;
  }
  td {
    padding: 8px 14px;
    border-bottom: 1px solid #e2e8f0;
    background: white;
  }
  tr:nth-child(even) td { background: #f8fafc; }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
  }
  .columns-1-2 {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 28px;
  }
  .takeaway {
    background: #fef3c7;
    border-left: 5px solid #f59e0b;
    padding: 10px 18px;
    margin-top: 14px;
    border-radius: 0 6px 6px 0;
    color: #78350f;
    font-weight: 500;
  }
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    background: #dbeafe;
    color: #1e3a8a;
    font-size: 0.8em;
    margin-right: 6px;
  }
  ul, ol { line-height: 1.55; }
  strong { color: #1e40af; }
  section.lead strong { color: #fde68a; }
  header, footer {
    color: #64748b;
    font-size: 14px;
  }
---

<!-- _class: lead -->

# Constraint Programming
## on Integers — a practical tour with NuCS

<br>

**Yan Georget** &nbsp;·&nbsp; 60 minutes

---

# What you'll get out of this hour

<div class="columns">

<div>

### <span class="pill">Part 1</span> Motivation
Why we need CP — NP-hard problems and the search-space explosion.

### <span class="pill">Part 2</span> Theory
CSP, propagation, consistency, search.

</div>

<div>

### <span class="pill">Part 3</span> Real problems
Queens, Sudoku, Crypto, Golomb, TSP — in NuCS.

### <span class="pill">Part 4</span> Inside a solver
Propagators, propagation loop, backtracking, JIT.

</div>

</div>

<div class="takeaway">No prior CP knowledge assumed. We stay on <strong>integer variables</strong> throughout.</div>

---

<!-- _class: lead part1 -->

# Part 1
## Why constraint programming?

---

<!-- _class: part1 -->

# A familiar pain: NP-completeness

A problem is in **NP** if a candidate solution can be checked in polynomial time.
A problem is **NP-hard** if every NP problem reduces to it.
**NP-complete** = NP ∩ NP-hard.

Classic NP-complete decision problems you've likely met:

- **SAT** — is this Boolean formula satisfiable?
- **3-coloring** — can this graph be 3-colored?
- **Hamiltonian cycle** — does this graph have a tour visiting every node once?
- **Subset sum**, **bin packing**, **scheduling with precedences**, …

<div class="takeaway">No known polynomial-time algorithm. But we still need answers — every day.</div>

---

<!-- _class: part1 -->

# The naive approach blows up fast

8-queens — place 8 queens on an 8×8 board, no two attacking.

| Encoding | Search space |
| -------- | -----------: |
| Naive: pick any 8 squares | $\binom{64}{8} \approx 4.4 \cdot 10^9$ |
| One queen per column | $8^8 \approx 1.7 \cdot 10^7$ |
| Permutations (one per column **and** row) | $8! = 40\,320$ |
| **A good CP model** | **~100 nodes explored** |

<div class="takeaway">The art is <strong>encoding</strong> the problem so the solver can prune aggressively.</div>

---

<!-- _class: part1 -->

# Where CP fits among solvers

| Paradigm | Variables | Strength |
| -------- | --------- | -------- |
| **SAT** | Boolean | Huge industrial formulas, conflict learning |
| **MIP** | Continuous + integer | Linear constraints, strong LP relaxation |
| **SMT** | Boolean + theories | Verification, symbolic execution |
| **CP** | Integer (mostly), finite | **Expressive global constraints**, puzzles, scheduling |

<br>

CP shines when constraints are **structured and non-linear**:
`all_different`, `circuit`, `cumulative`, `element`, …

---

<!-- _class: part1 -->

# A CP problem in one breath

A **Constraint Satisfaction Problem (CSP)** is a triple $\langle X, D, C \rangle$:

- $X = \{x_1, \dots, x_n\}$ — **variables**
- $D = \{D_1, \dots, D_n\}$ — **domain** for each variable (here: finite sets of integers)
- $C = \{c_1, \dots, c_m\}$ — **constraints**, each restricting a subset of variables

A **solution** assigns each $x_i$ a value $v_i \in D_i$ such that every $c \in C$ holds.

<div class="takeaway">If we also want <strong>min</strong> or <strong>max</strong> of an objective, it becomes a <strong>COP</strong> (Constraint Optimization Problem).</div>

---

<!-- _class: part1 -->

# Teaser: N-Queens in NuCS

```python
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver

class QueensProblem(Problem):
    def __init__(self, n: int):
        super().__init__([(0, n - 1)] * n)               # n variables, domain 0..n-1
        self.add_propagator(ALG_ALLDIFFERENT, range(n))                   # rows
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(n))         # diag ↘
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(0, -n, -1)) # diag ↙

solver = BacktrackSolver(QueensProblem(8))
print(next(solver.solve()))
```

<div class="takeaway">Three lines of model. The rest is the solver's job.</div>

---

<!-- _class: lead part2 -->

# Part 2
## How a CP solver thinks

---

<!-- _class: part2 -->

# Two ingredients: propagation and search

Solving a CSP is the interleaving of:

1. **Propagation** — for each constraint, remove from each variable's domain values that cannot participate in any solution of that constraint. Repeat until nothing changes.

2. **Search** — when propagation can't decide, **branch**: pick a variable, split its domain, recurse. On failure, **backtrack**.

<br>

<div class="columns">

<div>

**Propagation alone**
rarely solves anything.

</div>

<div>

**Search alone**
explodes.

</div>

</div>

<div class="takeaway">Together they're powerful.</div>

---

<!-- _class: part2 -->

# Domains, in NuCS specifically

NuCS uses **bound consistency** — each domain is represented as an **interval** $[\min, \max]$, not a full set.

```python
super().__init__([(0, n - 1)] * n)   # n domains, each = interval [0, n-1]
```

<div class="columns">

<div>

**Cheap**
2 integers per variable.

**Ground**
when $\min = \max$.

</div>

<div>

**Trade-off**
no holes inside an interval — but much faster.

**Choice**
bound consistency over arc consistency.

</div>

</div>

---

<!-- _class: part2 -->

# Consistency levels (informally)

For a constraint $c$ on variables $x_1, \dots, x_k$:

- **Node consistency** — values in $D_i$ satisfy any unary constraint on $x_i$.
- **Arc consistency (AC)** — every value in $D_i$ has a **support** in every other $D_j$ such that $c$ holds.
- **Bound consistency (BC)** — the same, but only the endpoints $\min(D_i), \max(D_i)$ need a support, and the supporting values can be any real in $[\min, \max]$.
- **Path / k-consistency** — stronger, more expensive, rarely worth it.

<div class="takeaway">Enforcing AC removes more values than BC, but costs more per call. Most modern solvers (including NuCS) lean on <strong>BC + smart global constraints</strong>.</div>

---

<!-- _class: part2 -->

# AC-3, the canonical propagation algorithm

```
Q ← all (variable, constraint) pairs
while Q not empty:
    (x, c) ← Q.pop()
    if revise(x, c) reduced D_x:
        if D_x is empty: return INCONSISTENT
        Q ← Q ∪ { (y, c') : y is a neighbor of x in c' ≠ c }
return CONSISTENT
```

Key idea: **only re-wake constraints whose variables changed**.

<div class="takeaway">In NuCS this becomes the <code>triggered_propagators</code> queue + per-variable <strong>event masks</strong> (MIN, MAX, GROUND).</div>

---

<!-- _class: part2 -->

# Global constraints: more than syntactic sugar

`all_different(x_1, …, x_n)` is logically equivalent to a clique of $\binom{n}{2}$ pairwise `≠` constraints.

<div class="columns">

<div>

**Clique form** propagates **weakly**:
each `≠` only fires when one side is ground.

</div>

<div>

**Dedicated** `all_different` reasons about the **whole set**:
Régin's AC uses bipartite matching;
BC versions use Hall intervals.

</div>

</div>

<div class="takeaway">Same logic, <strong>more reasoning per call</strong> → much smaller search tree. This is why CP scales on combinatorial problems.</div>

---

<!-- _class: part2 -->

# Search: backtracking + branching heuristics

```python
def search(state):
    propagate(state)
    if inconsistent: return FAIL
    if all variables ground: yield SOLUTION
    x = choose_variable(state)         # variable heuristic
    for v in choose_values(D_x):       # value heuristic
        push(state); assign(x, v)
        search(state)
        pop(state)
```

<div class="columns">

<div>

**Variable heuristics**
smallest domain first (*first-fail*),
most-constrained, impact-based.

</div>

<div>

**Value heuristics**
smallest first, min-conflicts,
promise-based.

</div>

</div>

<div class="takeaway">Heuristic quality is <strong>everything</strong>.</div>

---

<!-- _class: part2 -->

# Optimization = repeated satisfaction

To minimize an objective variable $z$:

```
best ← +∞
while True:
    add temporary constraint  z < best
    sol ← solve()                       # one solution
    if sol is None: return best
    best ← sol[z]
```

Each new solution tightens the upper bound on $z$. The added bound prunes the rest of the tree.

This is **branch and bound**. &nbsp; In NuCS: `solver.minimize(var)` / `solver.maximize(var)`.

---

<!-- _class: lead part3 -->

# Part 3
## Real problems, in NuCS

---

<!-- _class: part3 -->

# Problem 1 — N-Queens (revisited)

Three lines, again:

```python
super().__init__([(0, n - 1)] * n)
self.add_propagator(ALG_ALLDIFFERENT, range(n))
self.add_propagator(ALG_ALLDIFFERENT, range(n), range(n))          # x_i + i  all different
self.add_propagator(ALG_ALLDIFFERENT, range(n), range(0, -n, -1))  # x_i - i  all different
```

The clever bit — the second and third calls pass an **offset** per variable.
`all_different` is applied to $x_i + i$ (resp. $x_i - i$) **without materializing** those expressions: the offset is baked into the propagator.

<div class="takeaway">Three "reasoning hubs", not $3 \cdot \binom{n}{2}$ disequalities.</div>

---

<!-- _class: part3 -->

# Problem 2 — Sudoku

```python
class SudokuProblem(LatinSquareProblem):
    def __init__(self, givens):
        super().__init__(list(range(1, 10)), givens)        # rows + columns
        for i in range(3):
            for j in range(3):
                o = i * 27 + j * 3
                self.add_propagator(ALG_ALLDIFFERENT,
                    [0+o, 1+o, 2+o, 9+o, 10+o, 11+o, 18+o, 19+o, 20+o])
```

- 81 variables, each with domain $\{1, \dots, 9\}$.
- `LatinSquareProblem` provides 9 row + 9 column `all_different`s.
- Add 9 `all_different`s for the 3×3 boxes.

<div class="takeaway">Hardest Sudokus → <strong>milliseconds</strong>.</div>

---

<!-- _class: part3 -->

# Problem 3 — Cryptarithmetic

> `SEND + MORE = MONEY`

Variables: one per distinct letter, domain $\{0, \dots, 9\}$.

Constraints:

- `all_different(S, E, N, D, M, O, R, Y)`
- $S \neq 0$, $M \neq 0$ &nbsp; (no leading zero)
- A linear equation: &nbsp; $1000\,S + 100\,E + 10\,N + D + \dots = \dots$

NuCS expresses this with `ALG_AFFINE_EQ` (one linear-equality propagator) + one `ALG_ALLDIFFERENT`.

<div class="takeaway">The pattern: <strong>few constraints, each doing a lot</strong>.</div>

---

<!-- _class: part3 -->

# Problem 4 — Golomb ruler

> Find $n$ integer marks $0 = m_0 < m_1 < \dots < m_{n-1}$ such that all pairwise distances $m_j - m_i$ are **distinct**, and $m_{n-1}$ is **minimal**.

NuCS uses a distance-variable model:

```python
# one variable per pair (i, j)  →  dist_ij = m_j - m_i
self.add_propagator(ALG_SUM_EQ, [d_0i, d_ij, d_0j])    # for each i < j
self.add_propagator(ALG_ALLDIFFERENT, range(dist_nb))
```

Plus **redundant constraints** that the solver could deduce in principle, but not fast enough:
$\text{dist}_{i,j} \le m_{n-1} - \text{sum of first } (n-1-(j-i)) \text{ integers}$.

<div class="takeaway">Redundant constraints are a <strong>major</strong> practical lever.</div>

---

<!-- _class: part3 -->

# Golomb: symmetry breaking

The reversed ruler is also a Golomb ruler. To avoid exploring it:

```python
# first gap < last gap
self.add_propagator(ALG_LEQ_C, [d_{0,1}, d_{n-2,n-1}], [-1])
```

<div class="takeaway">One linear constraint = the search tree halved.</div>

<br>

**Symmetry breaking** is a recurring CP technique — exploit problem structure to forbid equivalent solutions.

---

<!-- _class: part3 -->

# Problem 5 — TSP

> $n$ cities, costs $c_{ij}$. Find a Hamiltonian cycle of minimum total cost.

**Modeling idea — successor representation:**

- $\text{succ}_i \in \{0, \dots, n-1\} \setminus \{i\}$ — the next city after $i$.
- A global `circuit` constraint enforces that `succ` forms one Hamiltonian cycle (**no sub-tours**).
- For each $i$, the cost contribution &nbsp; $\text{cost}_i = c_{i, \text{succ}_i}$ &nbsp; — modeled with **`element`**: index into the row $c_i$ by the variable $\text{succ}_i$.

---

<!-- _class: part3 -->

# TSP in NuCS

```python
class TSPProblem(CircuitProblem):
    def __init__(self, costs):
        n = len(costs)
        super().__init__(n)                                       # adds circuit over succ/pred
        self.succ_costs = self.add_variables([...])               # one cost variable per city
        self.total_cost = self.add_variable((sum_min, sum_max))
        for i in range(n):
            self.add_propagator(ALG_ELEMENT_EQ,
                [i, self.succ_costs + i], costs[i])               # cost_i = costs[i][succ_i]
        self.add_propagator(ALG_SUM_EQ,
            list(range(self.succ_costs, self.succ_costs + n)) + [self.total_cost])
```

Then: &nbsp; `solver.minimize(problem.total_cost)`.

---

<!-- _class: part3 -->

# Lessons from the five examples

1. **Few high-level constraints** beat many primitive ones.
2. **Reformulation matters** — successor variables for TSP, distance variables for Golomb, dual models for Queens.
3. **Redundant constraints** strengthen propagation without changing the solution set.
4. **Symmetry breaking** divides the search tree.
5. **The objective variable participates in propagation** — that's the engine of branch-and-bound.

---

<!-- _class: lead part4 -->

# Part 4
## Inside the solver

---

<!-- _class: part4 -->

# Anatomy of a CP solver

```
┌─────────────────────────────────────────────────────────────┐
│  Domains (state)        — [min, max] per variable           │
│  Propagators            — constraint implementations        │
│  Trigger table          — variable × event → propagator set │
│  Propagation queue      — propagators waiting to re-fire    │
│  Search engine          — branching + backtracking          │
│  Heuristics             — choose variable, choose value     │
│  Statistics             — backtracks, props, solutions, …   │
└─────────────────────────────────────────────────────────────┘
```

<div class="takeaway">NuCS specifics: everything is a <strong>NumPy array</strong> so it can be JIT-compiled with <strong>Numba</strong>.</div>

---

<!-- _class: part4 -->

# Propagator interface (NuCS)

Each propagator is **three functions** + an entry in a registry:

```python
ALG_X = register_propagator(get_triggers_x, get_complexity_x, compute_domains_x)
```

- `get_triggers_x(n, variable, parameters) -> EVENT_MASK`
   which domain events of *this* variable should re-wake the propagator?
- `get_complexity_x(n, parameters) -> int`
   used to order the propagation queue (cheap first).
- `compute_domains_x(domains, parameters) -> {INCONSISTENCY, CONSISTENCY, ENTAILMENT}`
   the actual filtering. Returns whether we hit a contradiction, made progress, or the constraint is now permanently satisfied.

---

<!-- _class: part4 -->

# A real propagator: `abs(y) = x`

```python
@njit(cache=True, fastmath=True)
def compute_domains_abs_eq(domains, parameters):
    y, x = domains[0], domains[1]
    if y[MIN] > 0:                        # y already positive → abs(y) = y
        if y[MIN] > x[MIN]: x[MIN] = y[MIN]
        if y[MAX] < x[MAX]: x[MAX] = y[MAX]
        ...
    elif y[MAX] < 0:                      # y already negative → abs(y) = -y
        ...
    else:                                 # 0 ∈ [y[MIN], y[MAX]]
        if x[MIN] < 0: x[MIN] = 0
        max_y = max(-y[MIN], y[MAX])
        if max_y < x[MAX]: x[MAX] = max_y
        ...
    return PROP_CONSISTENCY
```

<div class="takeaway">Pure mutation of <code>min</code>/<code>max</code>. No allocation. No Python objects. JIT-compiled.</div>

---

<!-- _class: part4 -->

# Events and triggers — only re-wake what matters

```python
def get_triggers_abs_eq(n, variable, parameters):
    if variable == 0:                # y: changing either bound affects x
        return EVENT_MASK_MIN_MAX
    return EVENT_MASK_MAX            # x: only MAX shrinking can prune y
```

Event masks: `MIN` changed, `MAX` changed, `GROUND` reached.
A propagator subscribes to a precise subset.

<div class="takeaway"><strong>Wakeup discipline is half the performance story</strong> — the other half is keeping the work itself fast.</div>

---

<!-- _class: part4 -->

# The propagation loop (pseudocode)

```python
while queue not empty:
    p = queue.pop_lowest_complexity()
    status = p.compute_domains(domains[p.vars], p.params)
    if status == INCONSISTENCY:
        return FAIL
    if status == ENTAILMENT:
        mark p as entailed (skip in this subtree)
    for v in p.vars where domain changed:
        for (event, q) in triggers[v]:
            if event in changes[v]:
                queue.add(q)
```

<div class="takeaway">Fixed-point. Terminates because domains are <strong>finite</strong> and only <strong>shrink</strong>.</div>

---

<!-- _class: part4 -->

# Search + propagation, together

```python
def search(state, depth):
    propagate(state)                       # fixed-point
    if FAIL:        backtrack(); return
    if all ground:  yield solution; return
    x = choose_variable(state)
    for branch in choose_values(D_x):      # e.g. x = v  |  x ≠ v
        push_state(state)                  # save min/max + entailed flags
        assert branch                      # adds to the queue
        search(state, depth + 1)
        pop_state(state)
```

<div class="takeaway">Two engines, one loop. The <strong>trail</strong> (saved domain snapshots) keeps backtracking <em>O(vars)</em>.</div>

---

<!-- _class: part4 -->

# Why this is hard to make fast in Python

- Per-variable updates are **tight integer math** — comparisons, mins, maxes.
- A solve can do **millions** of propagator calls per second.
- Pure Python's interpreter overhead would dominate by 50–100×.

**NuCS strategy:**

1. **Numba JIT** every hot function — `@njit(cache=True, fastmath=True)`.
2. **No Python objects** in jitted code — only typed NumPy arrays and ints.
3. **Mutate in place**, never reallocate.
4. **Function pointers** via `_get_wrapper_address` so propagators can be dispatched from jitted code.
5. **Multiprocessing** for embarrassingly parallel splits (`MultiprocessingSolver` over `problem.split(...)`).

---

<!-- _class: part4 -->

# What you give up vs. what you get

<div class="columns">

<div>

### ⚠️ You give up

- Sparse domains with **holes** (NuCS is bound-consistent).
- Arbitrary Python in hot paths.
- Some **expressiveness** — model with the propagators that exist (or write a new one).

</div>

<div>

### ✅ You get

- A **pure-Python** package that's competitive with C++ solvers on many benchmarks.
- A **small, readable** codebase you can extend.
- **Warm-cache** startup (Numba) + trivial multiprocessing parallelism.

</div>

</div>

---

<!-- _class: part4 -->

# When should you reach for CP?

<div class="columns">

<div>

### ✅ Good fit

- Combinatorial puzzles, **scheduling**, **rostering**, configuration, routing.
- Problems with rich logical structure — `all_different`, precedences, packing.
- Optimization where the objective is **integer** and the model is highly constrained.

</div>

<div>

### ⚠️ Less good fit

- Pure linear / continuous → use **MIP**.
- Boolean-heavy with no structure → **SAT** may be faster.
- Very-large-scale industrial OR → **CP-SAT** (OR-Tools) combines ideas from CP + SAT.

</div>

</div>

---

# What we covered

1. **NP-completeness** motivates smart search, not just enumeration.
2. **CP** = $\langle X, D, C \rangle$ + propagation + search.
3. **Global constraints** (`all_different`, `circuit`, `element`, …) carry most of the reasoning.
4. **NuCS** models problems in a few `add_propagator` calls.
5. A solver is a **propagation queue** + a **backtracking loop**. NuCS makes it fast with **Numba**.

---

# Resources

- 📚 **CSPLIB** — [csplib.org](https://www.csplib.org) — reference catalogue (Queens = #54, Golomb = #6).
- 📕 **Handbook of Constraint Programming** — Rossi, van Beek, Walsh (Elsevier, 2006).
- 🧰 **MiniZinc Handbook** — best on-ramp to modeling.
- ⚙️ **NuCS** — [github.com/yangeorget/nucs](https://github.com/yangeorget/nucs)
- 🏭 **OR-Tools CP-SAT** — Google's industrial-grade solver.

---

<!-- _class: lead -->

# Questions?

<br>

*Bonus demo material if there's time:*
&nbsp;&nbsp;live-run Queens, Sudoku, Golomb-10, TSP-15.
