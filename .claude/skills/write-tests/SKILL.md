---
name: write-tests
description: How NuCS tests are laid out and written — where a test file goes, class and method naming, parametrized case tables, brute-force oracle tests, solver-level tests, and keeping the suite fast with and without the JIT. Use when adding or editing a test under tests/, when a bug fix needs a regression test, or when deciding how to test a propagator, heuristic, solver or FlatZinc builtin.
---

# Write tests

## Where a test goes

`tests/` mirrors `nucs/`, one test file per module:

| module | test file |
|---|---|
| `nucs/buckets.py` | `tests/test_buckets.py` |
| `nucs/heuristics/min_value_dom_heuristic.py` | `tests/heuristics/test_min_value_dom_heuristic.py` |
| `nucs/propagators/abs_eq_propagator.py` | `tests/propagators/test_abs_eq.py` (no `_propagator` suffix) |
| `nucs/examples/queens/queens_problem.py` | `tests/examples/test_queens.py` |
| `nucs/fzn/builtins.py` | `tests/fzn/test_builtins.py` |

Add to the existing file when there is one. A new file starts with the `header.txt` banner (`add-header`).

## Shape

```python
class TestAbsEq(PropagatorTest):  # propagators extend PropagatorTest; everything else extends nothing
    @pytest.mark.parametrize(
        "domains,parameters,consistency_result,expected_domains",
        [
            ([(-4, 4), (-5, 5)], [], PROP_CONSISTENCY, [[-4, 4], [0, 4]]),
            # y[DOMAIN_MIN] > 0 branch, x narrowed empty -> inconsistency
            ([(5, 6), (1, 3)], [], PROP_INCONSISTENCY, None),
        ],
    )
    def test_compute_domains(
        self,
        domains: list[int | tuple[int, int]],
        parameters: list[int],
        consistency_result: int,
        expected_domains: list[list[int]] | None,
    ) -> None:
        self.assert_compute_domains(compute_domains_abs_eq, domains, parameters, consistency_result, expected_domains)
```

A case that exercises one branch or one rule gets a one-line comment naming it.

- One class per file, `Test<Module>` in CamelCase.
- Every test method is annotated `-> None`, and so is every helper: mypy checks `tests/`.
- Name a method `test_<what>_<scenario>`, as a sentence about the behaviour: `test_solve_all_sub_cycle_5`,
  `test_build_model_negative_unit_coefficient_linear_routes_to_sum`. Reuse the established names where they fit:
  `test_compute_domains`, `test_solve_all`, `test_find_all`, `test_find_best`, `test_soundness_against_brute_force`,
  `test_idempotent`, `test_live_set_is_sound`.
- Several cases of one behaviour go in one `@pytest.mark.parametrize` table, not in copies of a method.
- A test that needs something the environment may lack is skipped, not failed:
  `@pytest.mark.skipif(bool(NUMBA_DISABLE_JIT), reason=...)` (from `nucs.numba_helper`), or `MINIZINC is None`.

## Which test

- **Curated cases**, for the rules a piece of code was written around: a case table with its expected result. For
  a propagator, `assert_compute_domains` also checks the change report on every call and runs a non-idempotent
  propagator to its fixpoint, as the engine would.
- **A brute-force oracle**, for anything whose bugs hide in corners nobody wrote down: pruning, cascading rules,
  index offsets. Generate small instances, enumerate their solutions, and assert the code never loses one
  (soundness), fails only when there is none, and, when it claims idempotence, narrows nothing on a second call.
  For a propagator, `PropagatorTest.assert_sound_against_brute_force` does all of it, plus the entailment claim, the
  change report and empty domains, from a box and an `is_solution` predicate; `random_bounds` draws the box. See
  `tests/propagators/test_lexleq.py::test_soundness_against_brute_force`, and `test_element_l_eq_alldifferent.py`
  for a propagator that relies on another constraint (`assumption`).
  - Enumerate every instance when the space is small (`itertools.product`, as in `test_mul_eq.py::test_idempotent`);
    sample it with a seeded `random.Random(<date>)` or `np.random.default_rng(seed)` when it is not. Never leave a
    test unseeded: a failure must replay.
  - Put the instance in the assertion message (`f"pruned ...: {bounds}"`), so a failure can be pasted into the case
    table as a regression.
- **Solver-level**, for problems, examples, heuristics and the search: solve and assert on `solver.statistics`
  (`STATS_IDX_SOLUTION_NB`, `STATS_IDX_SOLVER_CHOICE_NB`, ...) against known counts, not on one particular solution.
  Asserting choice or propagation counts pins how much a propagator prunes, not only whether the answer is right.
- **A state block** (live set, trailed data) is checked along a chain of narrowings with backtracks, via
  `assert_live_set_is_sound`.

## A test must earn its place

- **Check it fails without the change.** Revert the fix, or break the rule it guards, and watch the test go red. A
  test that passes either way is only cost.
- **Say what it guards** in a comment when that is not obvious from the name, including why the size was chosen:
  `# within [-3, 3] a second call narrows nothing -- a one-pass mul_eq fails this on 496 of them`.
- **Keep it cheap.** The suite runs in about 20 s with the JIT, and CI also runs it with `NUMBA_DISABLE_JIT=1`,
  where a loop over thousands of instances is interpreted Python. Measure a new test both ways
  (`pytest --durations=10`, and again under `NUMBA_DISABLE_JIT=1`), and shrink domains, sizes and iteration counts
  to the smallest that still fail the broken version. Wait out a timeout or deadline for as short a time as the check allows, a small
  multiple of the deadline and never whole seconds.
