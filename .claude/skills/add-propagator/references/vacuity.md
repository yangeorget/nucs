# Vacuity: `is_vacuous_<name>`

A constraint is vacuous when no assignment its domains allow can violate it; a propagator for it is pure overhead.
This is common in FlatZinc, where a global is emitted with whatever capacities the model happens to give it.

```python
def is_vacuous_<name>(n: int, parameters: Sequence[int], domains: Sequence[tuple[int, int]]) -> bool:
    # plain Python, called once per add_propagator: sum/all/comprehensions are fine
    # True only when no assignment allowed by `domains` can violate the constraint
```

`Problem.add_propagator` calls it before posting and, on True, does not post: no calls, no trigger entries, no slot
in the propagator arrays, and `problem.propagator_nb` does not count it.

## Rules

- **A wrong True silently drops a constraint**: extra solutions come out and nothing reports an error. Prove the
  claim in the docstring; do not pattern-match a special case.
- **The domains are those held at post time.** Domains only shrink during search, so a property that holds of them
  holds throughout — which is what makes it safe to use them and not only the parameters. `is_vacuous_regular` needs
  them: an all-accepting automaton still filters values outside its alphabet, so it is vacuous only once every domain
  already sits inside that alphabet.
- **Vacuity is about the constraint, not the filtering.** "This call prunes nothing" is not vacuity.
  `PROP_ENTAILMENT` is the run-time counterpart, decided per call; vacuity is decided once, at post time.
- **Order the cheap discriminating tests first.** `is_vacuous_gcc` scans the upper capacities before the lower ones
  because that fails fast on a constraint that binds.

Examples: `is_vacuous_cumulative` and `is_vacuous_gcc` (parameters only), `is_vacuous_regular` (parameters and
domains).

## Tests

Write three; the third is the one that catches a wrong claim. See `tests/propagators/test_cumulative.py`,
`test_gcc.py` and `test_regular.py`.

1. **A vacuous case is not posted**: `is_vacuous_<name>(...)` is True and `problem.propagator_nb == 0` after
   `add_propagator`, including the boundary where the constraint only just stops binding.
2. **A binding case is posted**, one parameter away from the vacuous one, so the test pins the boundary rather than
   the direction.
3. **Dropping it preserves the solutions.** Build the same problem twice — once through `add_propagator`, which drops
   it, and once bypassing the check — and assert both enumerate exactly the same solutions:

   ```python
   bypassed.propagators.append((list(variables), ALG_<NAME>, list(parameters)))
   bypassed.propagator_nb += 1
   ```
