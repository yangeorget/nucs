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
Differential tests: NuCS, with and without lazy clause generation, against Gecode and Chuffed on small random MiniZinc
models.

A wrong explanation gives a wrong learned clause, and a wrong learned clause removes solutions without an error. These
tests find such a loss on models that nobody wrote by hand: each model mixes linear, reified, arithmetic and global
constraints over a few small variables. For a satisfaction model, the sets of solutions must be equal; for an
optimization model, the optima must be equal. Each NuCS run is also compared with the debug mode of LcgSolver, which
checks each explanation and the implication trail at each node.

These tests are skipped unless ``minizinc`` is on PATH and lists Gecode or Chuffed. Set ``MZN_SOLVER_PATH`` to add a
solver that the ``minizinc`` binary does not list, for example the solvers of the MiniZinc IDE bundle.
"""

import json
import os
import random
import re
import subprocess
import tempfile
from collections.abc import Callable

import pytest

from nucs.constants import DOMAIN_MIN
from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.search_annotations import search_heuristics, search_options
from nucs.solvers.backtrack.backtrack_solver import BacktrackSolver
from nucs.solvers.lcg.lcg_solver import LcgSolver
from nucs.solvers.solver import OPTIM_PRUNE
from nucs.solvers.tree_search_solver import TreeSearchSolver
from nucs.statistics import STATS_IDX_SOLUTION_NB
from tests.fzn.fzn_test import MINIZINC, compile_to_fzn, solve_fzn
from tests.marks import SLOW_WITHOUT_JIT

pytestmark = pytest.mark.skipif(MINIZINC is None, reason="minizinc is not installed")

REFERENCE_SOLVERS = ("org.gecode.gecode", "org.chuffed.chuffed")

# A solution: the value of each output variable, sorted by name. An answer: the set of the solutions of a satisfaction
# model, or the optimum of an optimization model (None when the model has no solution).
Solution = tuple[tuple[str, int], ...]
Answer = frozenset[Solution] | int | None

# Gecode does not end an enumeration that restarts: it finds the same solutions again. The reference solvers get the
# model without it, since the search does not change the answer.
RESTART_ANNOTATION = " :: restart_luby(2)"
SEARCH_ANNOTATIONS = (
    "",
    " :: int_search(XS, input_order, indomain_min, complete)",
    " :: int_search(XS, first_fail, indomain_max, complete)",
    " :: int_search(XS, dom_w_deg, indomain_split, complete)",
    " :: int_search(XS, input_order, indomain_reverse_split, complete)",
    " :: int_search(XS, smallest, indomain_median, complete)",
    " :: int_search(XS, first_fail, indomain_min, complete)" + RESTART_ANNOTATION,
)


class TestDifferential:
    # without the JIT, the first seeds run the same code at a fraction of the cost
    @pytest.mark.parametrize(
        "seed", [seed if seed < 8 else pytest.param(seed, marks=SLOW_WITHOUT_JIT) for seed in range(40)]
    )
    def test_solve_random_model(self, seed: int, reference_solvers: list[str]) -> None:
        _assert_agrees_with_the_references(seed, reference_solvers)


@pytest.fixture(scope="module")
def reference_solvers() -> list[str]:
    """
    Returns the reference solvers that ``minizinc`` lists, and skips the tests when there is none.

    :return: the ids of the reference solvers
    :rtype: list[str]
    """
    assert MINIZINC is not None
    result = subprocess.run(
        [MINIZINC, "--solvers-json"], capture_output=True, text=True, timeout=60, check=True, env=os.environ
    )
    listed = {solver["id"] for solver in json.loads(result.stdout)}
    solvers = [solver for solver in REFERENCE_SOLVERS if solver in listed]
    if not solvers:
        pytest.skip("minizinc lists neither Gecode nor Chuffed")
    return solvers


def _assert_agrees_with_the_references(seed: int, reference_solvers: list[str]) -> None:
    """
    Solves the random model of a seed with NuCS, with and without LCG, and with each reference solver, and checks that
    all the answers are equal.

    :param seed: the seed of the model
    :type seed: int
    :param reference_solvers: the ids of the reference solvers
    :type reference_solvers: list[str]
    """
    model, objective = _random_model(seed)
    with tempfile.TemporaryDirectory() as tmp:
        ozn_path = os.path.join(tmp, "model.ozn")
        fzn = compile_to_fzn(model, ozn_path)
        nucs_answers = {
            name: _nucs_answer(fzn, ozn_path, objective, lcg) for name, lcg in (("nucs", False), ("nucs --lcg", True))
        }
    runs = {solver: _reference_run(model, solver, objective is None) for solver in reference_solvers}
    rejected = {solver: run.stderr.strip() for solver, run in runs.items() if run.stdout.strip() == "=====ERROR====="}
    if len(rejected) == len(runs):
        # Chuffed rejects a div or a mod on a variable whose sign is not fixed, and so did the Gecode of the MiniZinc
        # 2.10.1 bundle of the CI on two seeds that Gecode 6.4.0 solves: with no answer, the seed shows nothing
        pytest.skip(f"no reference solver accepts the model of seed {seed}: {rejected}")
    reference_answers = {
        solver: _answer(run.stdout, objective) for solver, run in runs.items() if solver not in rejected
    }
    if len(set(reference_answers.values())) > 1:
        # Chuffed 0.13.2 is wrong on some models, most with a div or a mod: with no answer to trust, the seed shows
        # nothing
        pytest.skip(f"the reference solvers disagree on seed {seed}")
    reference, expected = next(iter(reference_answers.items()))
    for name, answer in nucs_answers.items():
        assert answer == expected, f"{name} differs from {reference} on seed {seed}:\n{model}"
    _assert_debug_lcg_agrees_with_backtrack(fzn, seed, model)


def _assert_debug_lcg_agrees_with_backtrack(fzn: str, seed: int, model: str) -> None:
    """
    Solves the FlatZinc with BacktrackSolver and with LcgSolver in debug mode, with the search of the runner, and
    checks that they give the same number of solutions, or the same optimum. Unlike the answers of the runner, the
    count covers all the variables of the FlatZinc, so a solution that LCG finds twice is an error.

    :param fzn: the FlatZinc text
    :type fzn: str
    :param seed: the seed of the model, for the message
    :type seed: int
    :param model: the MiniZinc text, for the message
    :type model: str
    """
    results = [_solve_directly(fzn, lcg) for lcg in (False, True)]
    assert results[0] == results[1], f"LcgSolver(debug=True) differs from BacktrackSolver on seed {seed}:\n{model}"


def _solve_directly(fzn: str, lcg: bool) -> int | None:
    """
    Solves a FlatZinc model with the search of the runner, without the runner, with BacktrackSolver or with LcgSolver
    in debug mode.

    :param fzn: the FlatZinc text
    :type fzn: str
    :param lcg: whether LcgSolver in debug mode solves the model
    :type lcg: bool

    :return: the optimum (None when there is no solution), or the number of solutions of a satisfaction model
    :rtype: Optional[int]
    """
    model = build_model(parse(fzn))
    last_conflict, restart_policy = search_options(model, False, None, None)
    searches = search_heuristics(model)
    solver: TreeSearchSolver = (
        LcgSolver(
            model.problem,
            searches=searches,
            log_level="ERROR",
            restart_policy=restart_policy,
            last_conflict=last_conflict,
            debug=True,
        )
        if lcg
        else BacktrackSolver(
            model.problem,
            searches=searches,
            log_level="ERROR",
            restart_policy=restart_policy,
            last_conflict=last_conflict,
        )
    )
    # the kind of the FlatZinc, not of the model: when the compilation proves that there is no solution, it writes
    # "solve satisfy" with no objective
    if model.solve.kind != "satisfy":
        assert model.solve.objective is not None
        variable = model.var_index_of(model.solve.objective)
        best = solver.find_best(variable, DOMAIN_MIN, OPTIM_PRUNE)
        return None if best is None else int(best[variable])
    solver.solve_all()
    return int(solver.statistics[STATS_IDX_SOLUTION_NB])


def _nucs_answer(fzn: str, ozn_path: str, objective: dict[str, int] | None, lcg: bool) -> Answer:
    """
    Solves a FlatZinc model with the runner, and prints its solutions with the output model as MiniZinc does: the
    runner prints only the variables of the FlatZinc, not the variables that the compilation fixed.

    :param fzn: the FlatZinc text
    :type fzn: str
    :param ozn_path: the output model
    :type ozn_path: str
    :param objective: the coefficient of each variable in the objective to minimize, or None for a satisfaction model
    :type objective: Optional[dict[str, int]]
    :param lcg: whether an LcgSolver solves the model
    :type lcg: bool

    :return: the answer
    :rtype: Answer
    """
    assert MINIZINC is not None
    stream = solve_fzn(fzn, all_solutions=objective is None, lcg=lcg)
    result = subprocess.run(
        [MINIZINC, "--ozn-file", ozn_path], input=stream, capture_output=True, text=True, timeout=60, check=True
    )
    return _answer(result.stdout, objective)


def _reference_run(model: str, solver: str, all_solutions: bool) -> subprocess.CompletedProcess[str]:
    """
    Solves a MiniZinc model, without its restart annotation, with a reference solver.

    :param model: the MiniZinc text
    :type model: str
    :param solver: the id of the solver
    :type solver: str
    :param all_solutions: whether to print all the solutions
    :type all_solutions: bool

    :return: the run, whose stdout is the solution stream, or "=====ERROR=====" when the solver does not accept the
        model
    :rtype: subprocess.CompletedProcess[str]
    """
    assert MINIZINC is not None
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.mzn")
        with open(model_path, "w") as f:
            f.write(model.replace(RESTART_ANNOTATION, ""))
        result = subprocess.run(
            [MINIZINC, "--solver", solver] + (["-a"] if all_solutions else []) + [model_path],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=os.environ,
        )
    assert result.returncode == 0 or result.stdout.strip() == "=====ERROR=====", result.stderr
    return result


def _answer(stream: str, objective: dict[str, int] | None) -> Answer:
    """
    Reads the answer of a complete search from a solution stream.

    :param stream: the solution stream of MiniZinc, with one "name = value;" line for each variable of the model
    :type stream: str
    :param objective: the coefficient of each variable in the objective to minimize, or None for a satisfaction model
    :type objective: Optional[dict[str, int]]

    :return: the answer
    :rtype: Answer
    """
    assert "=====UNKNOWN=====" not in stream, stream
    blocks = stream.split("----------")[:-1]
    assert blocks or "=====UNSATISFIABLE=====" in stream, stream
    if blocks:
        assert "==========" in stream, stream  # the search is complete
    solutions = [
        tuple(sorted((name, int(value)) for name, value in re.findall(r"^(\w+) = (-?\d+);$", block, re.MULTILINE)))
        for block in blocks
    ]
    if objective is None:
        return frozenset(solutions)
    if not solutions:
        return None
    values = dict(solutions[-1])
    return sum(coefficient * values[name] for name, coefficient in objective.items())


def _random_model(seed: int) -> tuple[str, dict[str, int] | None]:
    """
    Makes a small random MiniZinc model: a few integer variables with the same domain, a few constraints of random
    kinds, a random search annotation, and in one case of three a linear objective to minimize. The variables are
    scalars so that all the solvers print the solutions in the same form. To sweep more seeds, call
    _assert_agrees_with_the_references on them.

    :param seed: the seed
    :type seed: int

    :return: the MiniZinc text, and the coefficient of each variable in the objective, or None for a satisfaction model
    :rtype: tuple[str, Optional[dict[str, int]]]
    """
    rng = random.Random(seed)
    lo = rng.randint(-3, 1)
    hi = lo + rng.randint(3, 5)
    xs = [f"x{i}" for i in range(1, rng.randint(4, 5) + 1)]
    lines = ['include "globals.mzn";'] + [f"var {lo}..{hi}: {x};" for x in xs]
    fresh: list[str] = []  # the variables that a constraint declares for itself
    makers = list(CONSTRAINT_MAKERS)
    for _ in range(rng.randint(2, 4)):
        maker = rng.choice(makers)
        if maker is _permutation:
            # each one declares its own variables: two of them can give a million solutions
            makers.remove(_permutation)
        lines += maker(rng, xs, lo, hi, fresh)
    annotation = rng.choice(SEARCH_ANNOTATIONS).replace("XS", f"[{', '.join(xs + fresh)}]")
    if rng.randrange(3) == 0:
        objective = {x: rng.choice([-3, -2, -1, 1, 2, 3]) for x in xs}
        terms = " + ".join(f"{coefficient} * {x}" for x, coefficient in objective.items())
        lines.append(f"solve{annotation} minimize {terms};")
        return "\n".join(lines) + "\n", objective
    lines.append(f"solve{annotation} satisfy;")
    return "\n".join(lines) + "\n", None


def _linear(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    terms = rng.sample(xs, rng.randint(2, 3))
    expression = " + ".join(f"{rng.choice([-3, -2, -1, 1, 2, 3])} * {x}" for x in terms)
    return [f"constraint {expression} {rng.choice(['<=', '>=', '=', '!='])} {rng.randint(-4, 4)};"]


def _logic(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    a, b, c = rng.sample(xs, 3)
    v = rng.randint(lo, hi)
    return [
        rng.choice(
            [
                f"constraint ({a} <= {b}) \\/ ({c} = {v});",
                f"constraint ({a} != {v}) -> ({b} < {c});",
                f"constraint ({a} = {b}) <-> ({c} >= {v});",
                f"constraint bool2int({a} > {v}) + bool2int({b} = {c}) + bool2int({c} <= {v}) >= 2;",
            ]
        )
    ]


def _arithmetic(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    a, b, c = rng.sample(xs, 3)
    d = rng.choice([-3, -2, 2, 3])
    return [
        rng.choice(
            [
                f"constraint {a} * {b} = {c};",
                f"constraint abs({a}) = {b};",
                f"constraint min({a}, {b}) = {c};",
                f"constraint max({a}, {b}) = {c};",
                f"constraint {a} div {d} = {b};",
                f"constraint {a} mod {d} = {b};",
                f"constraint {a} - {b} = {c};",
            ]
        )
    ]


def _all_different(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    return [f"constraint all_different([{', '.join(rng.sample(xs, rng.randint(3, len(xs))))}]);"]


def _element(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    index, value = rng.sample(xs, 2)
    if rng.randrange(2) == 0:
        array = [str(rng.randint(lo, hi)) for _ in range(rng.randint(3, 4))]
    else:
        array = [rng.choice(xs) for _ in range(3)]
    return [f"constraint [{', '.join(array)}][{index}] = {value};"]


def _count(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    array = ", ".join(rng.sample(xs, 3))
    a, b = rng.sample(xs, 2)
    v, w = rng.sample(range(lo, hi + 1), 2)
    return [
        rng.choice(
            [
                f"constraint count([{array}], {v}) = {a};",
                f"constraint among({a}, [{array}], {{{v}, {w}}});",
                f"constraint global_cardinality([{array}], [{v}, {w}], [{a}, {b}]);",
                f"constraint nvalue({a}, [{array}]);",
            ]
        )
    ]


def _table(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    columns = rng.sample(xs, rng.randint(2, 3))
    rows = [" , ".join(str(rng.randint(lo, hi)) for _ in columns) for _ in range(rng.randint(4, 8))]
    return [f"constraint table([{', '.join(columns)}], [| {' | '.join(rows)} |]);"]


def _order(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    a, b, c, d = rng.sample(xs, 4)
    # two values: with s = t, Gecode accepts t, and the decomposition of MiniZinc forbids it
    v, w = rng.sample(range(lo, hi + 1), 2)
    return [
        rng.choice(
            [
                f"constraint lex_lesseq([{a}, {b}], [{c}, {d}]);",
                f"constraint increasing([{a}, {b}, {c}]);",
                f"constraint value_precede({v}, {w}, [{a}, {b}, {c}, {d}]);",
            ]
        )
    ]


def _scheduling(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    starts = rng.sample(xs, 3)
    durations = ", ".join(str(rng.randint(1, 3)) for _ in starts)
    if rng.randrange(2) == 0:
        return [f"constraint disjunctive([{', '.join(starts)}], [{durations}]);"]
    resources = ", ".join(str(rng.randint(1, 2)) for _ in starts)
    return [f"constraint cumulative([{', '.join(starts)}], [{durations}], [{resources}], {rng.randint(2, 3)});"]


def _regular(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    # the symbols are 1..S, so the values of the domain that are not symbols are not accepted
    symbol_nb = max(1, hi)
    rows = [", ".join(str(rng.randint(0, 3)) for _ in range(symbol_nb)) for _ in range(3)]
    finals = ", ".join(str(q) for q in sorted(rng.sample(range(1, 4), rng.randint(1, 2))))
    sequence = ", ".join(rng.choice(xs) for _ in range(rng.randint(3, 4)))
    return [f"constraint regular([{sequence}], 3, {symbol_nb}, [| {' | '.join(rows)} |], 1, {{{finals}}});"]


def _permutation(rng: random.Random, xs: list[str], lo: int, hi: int, fresh: list[str]) -> list[str]:
    # circuit, subcircuit and inverse need values in 1..m, so they get their own variables, linked to the others
    m = rng.randint(3, 4)
    ys = [f"y{len(fresh) + i}" for i in range(1, m + 1)]
    fresh += ys
    lines = [f"var 1..{m}: {y};" for y in ys] + [f"constraint {ys[0]} != {rng.choice(xs)};"]
    kind = rng.choice(["circuit", "subcircuit", "inverse"])
    if kind != "inverse":
        return lines + [f"constraint {kind}([{', '.join(ys)}]);"]
    zs = [f"y{len(fresh) + i}" for i in range(1, m + 1)]
    fresh += zs
    return lines + [f"var 1..{m}: {z};" for z in zs] + [f"constraint inverse([{', '.join(ys)}], [{', '.join(zs)}]);"]


CONSTRAINT_MAKERS: tuple[Callable[[random.Random, list[str], int, int, list[str]], list[str]], ...] = (
    _linear,
    _logic,
    _arithmetic,
    _all_different,
    _element,
    _count,
    _table,
    _order,
    _scheduling,
    _regular,
    _permutation,
)
