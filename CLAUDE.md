# CLAUDE.md

Guidance for Claude Code working in this repository.
For repository layout and core concepts: see `ARCHITECTURE.md`.

## Conventions

Each convention lives in a skill under `.claude/skills/`, loaded on demand — invoke it at the moment below:

- After editing any Python file under `nucs/`, `tests/` or `scripts/`: `check-style`. It must pass before committing.
- Running or debugging tests, with or without the JIT: `run-tests`.
- Creating a Python file: `add-header`, since every one starts with the `header.txt` banner.
- Writing or editing a docstring: `write-docstring`.

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
