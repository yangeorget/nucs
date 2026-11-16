---
name: write-python-code
description: How NuCS code is kept simple — module layout, the naming vocabulary, when to share code and when a copy is correct, and the design choices that are settled. Use when writing, editing or refactoring any Python file under nucs/, tests/ or scripts/, when choosing a name, or before proposing a refactoring.
---

# Write Python code

These rules are the decisions of this project, not general advice. For jitted code, also apply
`write-numba-friendly-python-code`; when the two skills disagree, the Numba skill wins.

## Module layout

- Order: the imports, the logger and the constants, the classes, then the module functions.
- Put the module functions directly after the last class, not at the end of the file: a module statement after the
  classes (such as `RESTART_POLICY_NEVER = RestartPolicy()`) can call them at import time.
- A function that import-time code needs stays above the classes: a `@pytest.mark.parametrize` list, or a function
  that a module statement calls (`restart_limits`). A call inside a lambda is not an import-time use.
- A comment directly above a function moves with the function.

## Names

- Use the words that the code already uses. Before you make a new name, search for the concept (`git grep`). Do not
  make a synonym for a concept that has a name.
- A count ends with `_nb` (`domain_nb`, `trailed_nb`). A size ends with `_sz` (`utrail_sz`, `table_sz`). Names
  that come from outside NuCS (the standard library, Numba, a dataset key) keep their spelling.
- A propagator `<name>` lives in `nucs/propagators/<name>_propagator.py` and has the functions
  `get_complexity_<name>`, `get_triggers_<name>`, `compute_domains_<name>` and, when it has a state,
  `get_state_<name>`. A private variant of one of them starts with `_` and ends with what makes it different
  (`_compute_domains_count_eq_plain`). The `add-propagator` skill has the full contract.
- A constant is in capitals and starts with the name of its group: `DOMAIN_MIN`, `OFFSETS_STATE`, `ALG_*`,
  `PROP_FLAG_*`, `STATS_IDX_*`. A column index of an array is a constant, never a literal number.
- A rename applies to all the code, the tests and the open documents. Closed proposals and the sections of
  released versions in the changelog keep the old name.

## Duplication

- Put a rule in one place. Examples: one write barrier (`tighten_at`), one walk over the triggers
  (`update_propagators`). When two functions do the same steps, make a helper.
- In jitted code, the helper must not cost speed. A helper that the propagation loop calls must be
  `inline="always"`; a new `@njit(cache=True)` helper that is not inlined adds about 0.2 s to the start of each
  process (see `write-numba-friendly-python-code`).
- A copy is correct only when a measurement shows that the shared form is slower, or when a specialized variant has
  a different cost (the plain scan and the live-set scan of `count_eq`). Write a comment at the copy that tells why
  it is a copy.
- In tests, make a parametrized case table, not a copy of a test method (see `write-tests`).
- To make a long function easier to read, split it into named steps that are `inline="always"` (`gather_domains`,
  `record_entailment` in `bc_algorithm`). A named step costs nothing and tells what the block does.

## Settled choices: do not propose them again

- The search loop takes its arrays as positional parameters. Do not group them into namedtuples or records.
  ARCHITECTURE.md, "Data-oriented state", gives the reason.
- `nucs/propagators/propagators.py` imports and registers each propagator explicitly. Do not register by a string
  name: the explicit imports keep mypy, ruff and the IDE navigation working.
- The solver settings (`searches`, `weight_decay`, `restarts`, `last_conflict`) are parameters of the constructor
  only, not of `solve()` or `optimize()`.

## Other rules

- Each construct must also run under `NUMBA_DISABLE_JIT=1` (ARCHITECTURE.md, "The pure-Python escape hatch").
- Give each function its type hints: `mypy` runs in `check-style`.
- A comment tells why, not what. Write it in ASD-STE100 Simplified Technical English. A docstring follows
  `write-docstring`.
- Remove dead code and the parameters that no caller uses. Do not keep an old name as an alias. When the change
  removes a documented extension point, add a `Removed` entry to the changelog: the next release is a major
  (see `bump-version`).
