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
import io
import os
import signal
import subprocess
import sys
import tempfile

import pytest

from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.fzn.runner import run
from nucs.solvers.restarts import RESTART_LUBY, RestartPolicy
from tests.fzn.fzn_test import solve_fzn


class TestRunner:
    def test_solve_fzn_minimize(self) -> None:
        out = solve_fzn(
            "var 0..10: x :: output_var;\nvar 0..10: y :: output_var;\n"
            "constraint int_plus(x, y, 10);\nconstraint int_le(y, x);\n"
            "solve minimize x;"
        )
        assert "x = 5;" in out
        assert out.strip().endswith("==========")

    OPTIMIZATION_MODEL = (
        "var 0..9: x :: output_var;\nvar 0..18: z;\n"
        "constraint int_eq(x, z);\nconstraint int_le(z, 4);\nsolve maximize z;"
    )

    def test_solve_fzn_optimization_prints_only_the_optimum_by_default(self) -> None:
        # input_order/indomain_min finds z=0 first and branch-and-bound improves it up to the optimum 4, but
        # the FlatZinc interface asks for the intermediate solutions with -a / -i: without either, only the
        # optimal solution is printed, followed by the search-complete marker
        out = solve_fzn(self.OPTIMIZATION_MODEL, output_objective=True)
        objectives = [int(line.split("=")[1].strip(" ;")) for line in out.splitlines() if "_objective" in line]
        assert objectives == [4]
        assert out.rstrip().endswith("==========")  # optimality proven after the solution

    @pytest.mark.parametrize("all_solutions,intermediate", [(True, False), (False, True)])
    def test_solve_fzn_optimization_streams_improving_solutions_on_request(
        self, all_solutions: bool, intermediate: bool
    ) -> None:
        # both -a and -i ask for the intermediate solutions: every improving one is streamed, in
        # non-decreasing objective order, before the search-complete marker
        out = solve_fzn(
            self.OPTIMIZATION_MODEL,
            all_solutions=all_solutions,
            output_objective=True,
            intermediate_solutions=intermediate,
        )
        objectives = [int(line.split("=")[1].strip(" ;")) for line in out.splitlines() if "_objective" in line]
        assert len(objectives) > 1  # streamed, not just the optimum
        assert objectives == sorted(objectives)  # each solution improves on the previous
        assert objectives[-1] == 4  # the last streamed solution is the optimum
        assert out.rstrip().endswith("==========")

    def test_build_model_statistics_go_to_the_solution_stream(self) -> None:
        # the FlatZinc interface puts statistics on standard output as comments: on stderr, `minizinc -s`
        # never sees them
        out = io.StringIO()
        run(build_model(parse("var 0..2: x :: output_var;\nsolve satisfy;")), out, statistics=True)
        text = out.getvalue()
        assert "%%%mzn-stat: SOLUTION_NB=1" in text
        assert text.rstrip().endswith("%%%mzn-stat-end")  # the block concludes the stream
        assert text.index("----------") < text.index("%%%mzn-stat:")  # after the solutions, not before

    def test_solve_fzn_a_time_limit_streams_so_the_incumbent_survives(self) -> None:
        # MiniZinc may end the run from outside before the solver's own deadline stops it, so under a time
        # limit the best solution found so far has to be on the stream already, or the kill loses it. A
        # generous limit still proves optimality here; what matters is that the improving solutions were
        # printed on the way.
        out = solve_fzn(self.OPTIMIZATION_MODEL, output_objective=True, time_limit_ms=600_000)
        objectives = [int(line.split("=")[1].strip(" ;")) for line in out.splitlines() if "_objective" in line]
        assert len(objectives) > 1
        assert objectives[-1] == 4

    def test_solve_fzn_time_limit_reports_unknown_when_nothing_is_found(self) -> None:
        # a deadline already in the past stops the satisfy search before its first solution: nothing was
        # found but the space was not exhausted, which is exactly what the unknown marker reports
        out = solve_fzn("var 0..9: x :: output_var;\nsolve satisfy;", time_limit_ms=0)
        assert out.strip() in ("=====UNKNOWN=====", "x = 0;\n----------")

    @pytest.mark.skipif(sys.platform == "win32", reason="MiniZinc sends SIGTERM on POSIX only")
    def test_fzn_nucs_prints_its_statistics_on_sigterm(self) -> None:
        # MiniZinc enforces its time limit with SIGTERM, then SIGKILL a second later. The first solution
        # comes at once, but proving that 13 pigeons do not fit in 12 holes does not end, so the signal
        # lands inside the compiled search: the run must still conclude its stream with the statistics.
        xs = [f"x{i}" for i in range(13)]
        lines = [f"var 0..12: {x} :: output_var;" for x in xs] + ["var 0..12: m :: output_var;"]
        lines += [f"constraint int_ne({x}, {y});" for i, x in enumerate(xs) for y in xs[i + 1 :]]
        lines += [f"constraint int_le({x}, m);" for x in xs]
        lines.append(f"solve :: int_search([{', '.join(xs)}], input_order, indomain_min, complete) minimize m;")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pigeons.fzn")
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            process = subprocess.Popen(
                [sys.executable, "-m", "nucs.fzn", "-a", "-s", path], stdout=subprocess.PIPE, text=True
            )
            try:
                assert process.stdout is not None
                # the first solution proves the handler is installed, since it is installed before the search
                while (line := process.stdout.readline()) != "----------\n":
                    assert line, "fzn-nucs exited before its first solution"
                process.send_signal(signal.SIGTERM)
                rest, _ = process.communicate(timeout=60)
            finally:
                process.kill()
        assert process.returncode == 0
        assert "==========" not in rest  # interrupted, so no optimality claim
        assert rest.rstrip().endswith("%%%mzn-stat-end")
        assert "%%%mzn-stat: SOLUTION_NB=1" in rest

    def test_fzn_nucs_time_limit_stops_the_compiled_search(self) -> None:
        # run without MiniZinc, nothing but -t stops a proof that 13 pigeons do not fit in 12 holes: the
        # limit has to stop the compiled search itself, and the run still concludes its stream
        xs = [f"x{i}" for i in range(13)]
        lines = [f"var 0..11: {x} :: output_var;" for x in xs]
        lines += [f"constraint int_ne({x}, {y});" for i, x in enumerate(xs) for y in xs[i + 1 :]]
        lines.append(f"solve :: int_search([{', '.join(xs)}], input_order, indomain_min, complete) satisfy;")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pigeons.fzn")
            with open(path, "w") as f:
                f.write("\n".join(lines) + "\n")
            process = subprocess.run(
                [sys.executable, "-m", "nucs.fzn", "-s", "-t", "200", path],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        assert process.returncode == 0
        assert "=====UNKNOWN=====" in process.stdout
        assert "=====UNSATISFIABLE=====" not in process.stdout
        assert process.stdout.rstrip().endswith("%%%mzn-stat-end")

    @pytest.mark.parametrize("last_conflict", [False, True])
    def test_run_restarts_and_last_conflict(self, last_conflict: bool) -> None:
        # 4 pigeons in 3 holes behind a restart annotation: the proof must still end, with or without last-conflict
        text = (
            "array [1..4] of var 0..2: x;\n"
            + "".join(f"constraint int_ne(x[{i}], x[{j}]);\n" for i in range(1, 5) for j in range(i + 1, 5))
            + "solve :: restart_luby(1) :: int_search(x, dom_w_deg, indomain_min, complete) satisfy;\n"
        )
        out = io.StringIO()
        run(build_model(parse(text)), out, statistics=True, last_conflict=last_conflict)
        assert "=====UNSATISFIABLE=====" in out.getvalue()
        assert "SOLVER_RESTART_NB=0" not in out.getvalue()

    @pytest.mark.parametrize("free_search", [False, True])
    def test_run_free_search_finds_the_optimum(self, free_search: bool) -> None:
        # the free search ignores the annotation's order and values, and must still prove the optimum
        text = (
            "array [1..4] of var 0..3: x;\nvar 0..12: s;\n"
            + "".join(f"constraint int_ne(x[{i}], x[{j}]);\n" for i in range(1, 5) for j in range(i + 1, 5))
            + "constraint int_lin_eq([1, 1, 1, 1, -1], [x[1], x[2], x[3], x[4], s], 0);\n"
            + "solve :: int_search(x, input_order, indomain_max, complete) minimize x[1];\n"
        )
        out = io.StringIO()
        run(
            build_model(parse(text)),
            out,
            free_search=free_search,
            last_conflict=True,
            restart=RestartPolicy(RESTART_LUBY, 1),
        )
        assert "==========" in out.getvalue()  # the optimum is proven

    def test_solve_fzn_all_solutions_terminator(self) -> None:
        out = solve_fzn(
            "var 1..3: x :: output_var;\nvar 1..3: y :: output_var;\nconstraint int_lt(x, y);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3
        assert out.strip().endswith("==========")

    def test_solve_fzn_unsatisfiable(self) -> None:
        out = solve_fzn("var 1..2: x :: output_var;\nconstraint int_eq(x, 5);\nsolve satisfy;")
        assert out.strip() == "=====UNSATISFIABLE====="
