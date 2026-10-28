###############################################################################
# __   _            _____    _____
# | \ | |          / ____|  / ____|
# |  \| |  _   _  | |      | (___
# | . ` | | | | | | |       \___ \
# | |\  | | |_| | | |____   ____) |
# |_| \_|  \__,_|  \_____| |_____/
#
# Fast constraint solving in Python  - https://github.com/yangeorget/nucs
#
# Copyright 2024-2026 - Yan Georget
###############################################################################
"""
Translates the search annotations of a FlatZinc model, and the search options of the command line, into the
searches, the restarts and the last-conflict setting of a :class:`BacktrackSolver`.
"""

import logging
from collections.abc import Sequence

from nucs.fzn.model import FznModel
from nucs.fzn.parser import Ann, Id
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_RANDOM_VALUE,
    DOM_HEURISTIC_SPLIT_HIGH,
    DOM_HEURISTIC_SPLIT_LOW,
    VAR_HEURISTIC_DOM_WDEG,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_GREATEST_DOMAIN,
    VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
    VAR_HEURISTIC_MAX_REGRET,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
    VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
)
from nucs.solvers.restart_policies import (
    RESTART_CONSTANT,
    RESTART_GEOMETRIC,
    RESTART_LINEAR,
    RESTART_LUBY,
    RESTART_POLICY_NEVER,
    RestartPolicy,
)
from nucs.solvers.search import SEARCHES_DEFAULT, Search

logger = logging.getLogger(__name__)

# FlatZinc variable-selection annotations mapped to NuCS variable heuristics; unlisted ones
# (occurrence, most_constrained, ...) fall back to the default, with a warning -- silently
# solving a different search than the model asked for reads as slowness rather than as a missing feature.
_VAR_HEURISTICS = {
    "input_order": VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    "first_fail": VAR_HEURISTIC_SMALLEST_DOMAIN,
    "dom_w_deg": VAR_HEURISTIC_DOM_WDEG,
    "anti_first_fail": VAR_HEURISTIC_GREATEST_DOMAIN,
    "max_regret": VAR_HEURISTIC_MAX_REGRET,
    "smallest": VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
    "largest": VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
}
# FlatZinc value-selection annotations mapped to NuCS domain heuristics; likewise for the unlisted ones
# (indomain_interval, outdomain_min, ...).
_DOM_HEURISTICS = {
    "indomain_min": DOM_HEURISTIC_MIN_VALUE,
    "indomain_max": DOM_HEURISTIC_MAX_VALUE,
    "indomain_median": DOM_HEURISTIC_MID_VALUE,
    "indomain_random": DOM_HEURISTIC_RANDOM_VALUE,
    "indomain_split": DOM_HEURISTIC_SPLIT_LOW,
    "indomain_reverse_split": DOM_HEURISTIC_SPLIT_HIGH,
}


# the MiniZinc restart annotations whose only argument is the scale
_SCALED_RESTARTS = {
    "restart_constant": RESTART_CONSTANT,
    "restart_linear": RESTART_LINEAR,
    "restart_luby": RESTART_LUBY,
}


def restart_policy_of(model: FznModel) -> RestartPolicy:
    """
    Translates the restart annotation of the solve item into a NuCS restart policy. A restart limit counts failures,
    as Gecode's do.

    :param model: the built model
    :type model: FznModel

    :return: the restart policy, for BacktrackSolver; no restart when the model asks for none, or asks for one with
             arguments NuCS cannot use, which is logged as a warning
    :rtype: RestartPolicy
    """
    for annotation in model.solve.annotations:
        args = annotation.args
        try:
            if annotation.name == "restart_none":
                return RESTART_POLICY_NEVER
            if annotation.name in _SCALED_RESTARTS and len(args) == 1 and isinstance(args[0], int):
                return RestartPolicy(_SCALED_RESTARTS[annotation.name], args[0])
            if (
                annotation.name == "restart_geometric"
                and len(args) == 2
                and isinstance(args[0], (int, float))
                and isinstance(args[1], int)
            ):
                return RestartPolicy(RESTART_GEOMETRIC, args[1], float(args[0]))
        except ValueError as e:
            logger.warning(f"{e}, searching without restarts")
            return RESTART_POLICY_NEVER
        if annotation.name.startswith("restart_"):
            logger.warning(f"Unsupported restart annotation {annotation.name}{args}, searching without restarts")
            return RESTART_POLICY_NEVER
    return RESTART_POLICY_NEVER


# The free search (-f) defaults, measured on 41 challenge instances in 60 s against the models' own searches:
# last-conflict won 9 and lost 0 against the same free search without it; Luby restarts at scale 500 won 10 and
# lost 3 (proofs that got slower) against the free search without restarts.
FREE_SEARCH_LAST_CONFLICT = True
RESTART_POLICY_FREE_SEARCH = RestartPolicy(RESTART_LUBY, 500)


def search_options(
    model: FznModel, free_search: bool, last_conflict: bool | None, restart: RestartPolicy | None
) -> tuple[bool, RestartPolicy]:
    """
    Resolves last-conflict and the restart policy from the mode and the command line. What the command line gives
    wins. Otherwise the free search takes its defaults and ignores the model's restart annotation, as it ignores the
    rest of its search annotation; the fixed search takes no last-conflict and follows the restart annotation.

    :param model: the built model
    :type model: FznModel
    :param free_search: whether the search is free (-f)
    :type free_search: bool
    :param last_conflict: last-conflict as the command line gives it, or None for the mode's default
    :type last_conflict: Optional[bool]
    :param restart: the restart policy as the command line gives it, or None for the mode's default
    :type restart: Optional[RestartPolicy]

    :return: whether to use last-conflict, and the restart policy
    :rtype: Tuple[bool, RestartPolicy]
    """
    if last_conflict is None:
        last_conflict = FREE_SEARCH_LAST_CONFLICT if free_search else False
    if restart is None:
        restart = RESTART_POLICY_FREE_SEARCH if free_search else restart_policy_of(model)
    return last_conflict, restart


def parse_restart(text: str) -> RestartPolicy:
    """
    Parses a restart policy given on the command line as ``POLICY[,SCALE[,BASE]]``, such as ``luby,500`` or
    ``geometric,100,1.5``. The scale defaults to 100 and the base to 1.5, as in BacktrackSolver.

    :param text: the policy, one of RESTART_POLICIES, then optionally its scale and its base
    :type text: str

    :return: the restart policy
    :rtype: RestartPolicy
    """
    parts = text.split(",")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Expected POLICY[,SCALE[,BASE]], not {text!r}")
    policy = parts[0].strip().lower()
    scale = int(parts[1]) if len(parts) > 1 else 100
    base = float(parts[2]) if len(parts) > 2 else 1.5
    return RestartPolicy(policy, scale, base)  # raises on a wrong policy, scale or base


def search_heuristics(model: FznModel) -> Sequence[Search]:
    """
    Translates the first ``int_search``/``bool_search``/``seq_search`` annotation on the solve item into a
    NuCS sequential search: one :class:`Search` per nested search, each keeping its own variable and value
    selectors. A ``seq_search`` becomes the ordered list of its nested searches (flattened recursively, since
    MiniZinc nests a ``seq_search`` when decomposing e.g. ``set_search``); a single ``int_search`` /
    ``bool_search`` becomes a one-element list. Each variable is assigned to the first search that lists it,
    and a trailing catch-all search over the remaining variables (NuCS defaults) makes the search ground the
    whole problem. Unknown variable/value selectors fall back to the NuCS defaults, and each one is
    logged as a warning.

    :param model: the built model
    :type model: FznModel

    :return: the ordered searches, or SEARCHES_DEFAULT when there is no supported search annotation
    :rtype: Sequence[Search]
    """
    for annotation in model.solve.annotations:
        nested = _flatten_searches(model, annotation)
        if nested:
            searches: list[Search] = []
            seen: set = set()
            for variables, var_heuristic, dom_heuristic in nested:
                group = [v for v in variables if v not in seen]  # a variable belongs to the first search listing it
                seen.update(group)
                if group:
                    searches.append(Search(group, var_heuristic, [[]], dom_heuristic, [[]]))
            remaining = [v for v in range(model.problem.domain_nb) if v not in seen]
            if remaining:  # ground every remaining variable with the defaults
                searches.append(Search(remaining))
            return searches or SEARCHES_DEFAULT
    return SEARCHES_DEFAULT


def free_search_heuristics(model: FznModel) -> list[Search]:
    """
    Builds the free search (``-f``): the model's search annotations only say which variables it decides, and both
    the order and the values are left to NuCS. The variables the annotations name come first, in one dom/wdeg search,
    since they are the modeller's decisions; the remaining variables follow, in a second one, so that the search
    grounds the whole problem. A model with no search annotation gets one dom/wdeg search over all its variables.

    :param model: the built model
    :type model: FznModel

    :return: the ordered list of searches
    :rtype: List[Search]
    """
    annotated: list[int] = []
    seen: set = set()
    for annotation in model.solve.annotations:
        for variable in _annotated_variables(model, annotation):
            if variable not in seen:
                seen.add(variable)
                annotated.append(variable)
    remaining = [v for v in range(model.problem.domain_nb) if v not in seen]
    return [Search(variables, VAR_HEURISTIC_DOM_WDEG) for variables in (annotated, remaining) if variables]


def _annotated_variables(model: FznModel, annotation: Ann) -> list[int]:
    """
    Returns the variables a search annotation branches on, in order, recursing into ``seq_search``. Unlike
    _single_search it reads no selector, so the selectors a free search ignores raise no warning.

    :param model: the built model
    :type model: FznModel
    :param annotation: the search annotation
    :type annotation: Ann

    :return: the variables, possibly with repetitions
    :rtype: List[int]
    """
    if annotation.name == "seq_search" and annotation.args and isinstance(annotation.args[0], list):
        return [v for item in annotation.args[0] if isinstance(item, Ann) for v in _annotated_variables(model, item)]
    if annotation.name in ("int_search", "bool_search") and annotation.args:
        return model.var_list_of(annotation.args[0])
    return []


def _flatten_searches(model: FznModel, annotation: Ann) -> list[tuple[list[int], int, int]]:
    """
    Flattens a search annotation into an ordered list of (search variables, variable heuristic, domain
    heuristic) triples, recursing into nested ``seq_search`` annotations and dropping unsupported ones.

    :param model: the built model
    :type model: FznModel
    :param annotation: the search annotation
    :type annotation: Ann

    :return: the flattened list of search triples (empty when nothing is supported)
    :rtype: List[Tuple[List[int], int, int]]
    """
    if annotation.name == "seq_search" and annotation.args and isinstance(annotation.args[0], list):
        flattened: list[tuple[list[int], int, int]] = []
        for item in annotation.args[0]:
            if isinstance(item, Ann):
                flattened.extend(_flatten_searches(model, item))
        return flattened
    single = _single_search(model, annotation)
    return [single] if single is not None else []


def _single_search(model: FznModel, annotation: Ann) -> tuple[list[int], int, int] | None:
    """
    Translates a single ``int_search``/``bool_search`` annotation into a NuCS search configuration.

    :param model: the built model
    :type model: FznModel
    :param annotation: the search annotation
    :type annotation: Ann

    :return: a triple (search variables, variable heuristic, domain heuristic), or None when the annotation
             is not a supported search
    :rtype: Optional[Tuple[List[int], int, int]]
    """
    if annotation.name in ("int_search", "bool_search") and annotation.args:
        search_variables = model.var_list_of(annotation.args[0])
        var_heuristic = _var_heuristic_of(annotation.args[1] if len(annotation.args) > 1 else None)
        dom_heuristic = _dom_heuristic_of(annotation.args[2] if len(annotation.args) > 2 else None)
        return search_variables, var_heuristic, dom_heuristic
    return None


def _var_heuristic_of(term: object) -> int:
    """
    Returns the NuCS variable heuristic for a FlatZinc selector term, defaulting to first-not-instantiated.

    :param term: the selector term (an Id) or None
    :type term: object

    :return: the variable heuristic id
    :rtype: int
    """
    if isinstance(term, Id):
        if term.name not in _VAR_HEURISTICS:
            logger.warning(f"Unsupported variable selector {term.name}, branching as input_order instead")
        return _VAR_HEURISTICS.get(term.name, VAR_HEURISTIC_FIRST_NOT_INSTANTIATED)
    return VAR_HEURISTIC_FIRST_NOT_INSTANTIATED


def _dom_heuristic_of(term: object) -> int:
    """
    Returns the NuCS domain heuristic for a FlatZinc selector term, defaulting to min-value.

    :param term: the selector term (an Id) or None
    :type term: object

    :return: the domain heuristic id
    :rtype: int
    """
    if isinstance(term, Id):
        if term.name not in _DOM_HEURISTICS:
            logger.warning(f"Unsupported value selector {term.name}, branching as indomain_min instead")
        return _DOM_HEURISTICS.get(term.name, DOM_HEURISTIC_MIN_VALUE)
    return DOM_HEURISTIC_MIN_VALUE
