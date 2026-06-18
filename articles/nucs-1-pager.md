# NuCS — a fast constraint solver written 100% in Python

## What it is

NuCS is an open-source Python library (MIT license, `pip install nucs`) for solving
**Constraint Satisfaction Problems (CSP)** and **Constraint Optimization Problems** over finite domains.
You declare variables, their domains, and the constraints that link them;
the solver searches for assignments that satisfy every constraint — or, for optimization, the one that minimizes or
maximizes a target.

Its distinguishing bet is that it is written *entirely* in Python yet aims to be competitive with mature compiled
solvers.
It recovers the performance normally lost to interpretation not through a C/C++ core but through **just-in-time
compilation**:
the engine is built on [NumPy](https://numpy.org/) and [Numba](https://numba.pydata.org/), so the hot paths run as
cached native code.
The practical payoff is that installation is as simple as any Python package (no native toolchain, no JVM), modeling
takes only a few lines, and a warm process still runs at near-C throughput.

## How it works

A `Problem` holds a NumPy array of variable domains — one `(min, max)` pair per variable, a variable being *bound* when
`min == max` — plus a list of **propagators**.
A propagator is one constraint's filtering algorithm, registered under a numeric `ALG_*` identifier (e.g.
`ALG_ALLDIFFERENT`, `ALG_AFFINE_EQ`, `ALG_ELEMENT_IV`).
Internally each propagator is three small functions: one that does the actual filtering (reporting *inconsistency*,
*consistency*, or *entailment*),
one that declares which domain events should re-wake it, and one that estimates its cost so the propagation queue can be
ordered.

The `BacktrackSolver` alternates between propagating to a fixpoint and making a branching decision.
Two pluggable heuristics drive that decision:
a **variable heuristic** (which unbound variable to branch on) and a **domain heuristic** (which value to try).
For optimization, NuCS uses branch-and-bound — every time an intermediate solution `s` is found while minimizing a
variable `t`, it adds the constraint `t < s` to prune the rest of the search;
it offers a `RESET` mode (restart with tightened bounds) and a more efficient `PRUNE` mode (rewrite choice points in
place).
A `MultiprocessingSolver` fans several searches across a split of the search space to use multiple cores.

**The one design choice that shapes everything:** a domain is *always a single `min..max` interval*.
NuCS can shrink the bounds of a domain but cannot punch a hole in the middle.
This is what makes the whole engine expressible as operations on a couple of NumPy arrays (and therefore fast and
vectorizable),
but it limits NuCS to **bound consistency (BC)** rather than the stronger arc consistency (AC).
The intended workaround is cheap remodeling — adding redundant constraints, channeling between alternate views, or
writing a small custom filter — rather than reaching for an expensive AC algorithm.

## Bottom line

NuCS is a strong fit when you want to **prototype and reshape constraint models in Python and still get native-code
speed** —
small-to-medium CSPs and optimization problems, research, teaching, and combinatorial proofs.
Its cheap remodeling often matters as much as raw engine speed: adding redundancy or a custom filter is a handful of
lines.

---

**Links:** Source: <https://github.com/yangeorget/nucs> · Docs: <https://nucs.readthedocs.io/> ·
PyPI: <https://pypi.org/project/NuCS/>
