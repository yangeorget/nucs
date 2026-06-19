We want to define a meta search that combines several searches.
This is needed for the Minizinc' sequential search.

Currently, NuCS only supports a single search.
When we look at the constructor of a BacktrackSolver see the following parameters:
decision_variables: Optional[Iterable[int]] = None,
var_heuristic: int = VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
var_heuristic_params: List[List[int]] = [[]],
dom_heuristic: int = DOM_HEURISTIC_MIN_VALUE,
dom_heuristic_params: List[List[int]] = [[]]
These parameters define a search:

- a array decision variables
- a variable heuristic (and its optional parameters, bidimensional array)
- a domain heuristic (and its optional parameters, bidimensional array)

A meta-search or sequential search is just an array of searches.
We can then just have:

- an array of decision variables
- an array of variable heuristics (and its optional parameters, tridimensional array)
- an array of domain heuristics (and its optional parameters, tridimensional array)

We need to modify the solve_one method to iterate on this new dimension.

We also need unit tests.
We also need to fix the Flatzinc adapter.