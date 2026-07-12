# CLAUDE.md

Guidance for Claude Code working in this repository.
For repository layout and core concepts: see `ARCHITECTURE.md`.
For style checking (ruff & mypy): see `.claude/rules/check-style.md`.
For running tests (with and without JIT): see `.claude/rules/run-tests.md`.

## FlatZinc adapter

Register NuCS as a MiniZinc solver, then solve a `.mzn` model:

```bash
fzn-nucs --register
minizinc --solver nucs model.mzn
```

## Example

```python
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_ALLDIFFERENT
from nucs.solvers.backtrack_solver import BacktrackSolver


class QueensProblem(Problem):
    def __init__(self, n: int):
        super().__init__([(0, n - 1)] * n)
        self.add_propagator(ALG_ALLDIFFERENT, range(n))
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(n))
        self.add_propagator(ALG_ALLDIFFERENT, range(n), range(0, -n, -1))


solver = BacktrackSolver(QueensProblem(8))
solution = next(solver.solve(), None)
```
