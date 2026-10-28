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
import logging

import pytest

from nucs.fzn.model import build_model
from nucs.fzn.parser import Id, parse
from nucs.fzn.search_annotations import (
    FREE_SEARCH_RESTART,
    free_search_heuristics,
    parse_restart,
    restart_policy_of,
    search_heuristics,
    search_options,
)
from nucs.heuristics.heuristics import (
    DOM_HEURISTIC_MAX_VALUE,
    DOM_HEURISTIC_MID_VALUE,
    DOM_HEURISTIC_MIN_VALUE,
    DOM_HEURISTIC_RANDOM_VALUE,
    DOM_HEURISTIC_SPLIT_HIGH,
    VAR_HEURISTIC_DOM_WDEG,
    VAR_HEURISTIC_FIRST_NOT_INSTANTIATED,
    VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE,
    VAR_HEURISTIC_SMALLEST_DOMAIN,
    VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE,
)
from nucs.solvers.restarts import RESTART_CONSTANT, RESTART_GEOMETRIC, RESTART_LINEAR, RESTART_LUBY, RestartPolicy
from nucs.solvers.search import DEFAULT_SEARCHES
from tests.fzn.fzn_test import solve_fzn


class TestSearchAnnotations:
    def test_solve_fzn_seq_search_annotation(self) -> None:
        # seq_search nests int_search/bool_search calls inside an array: the parser must accept call terms
        # and the runner must flatten the nested searches into a single decision order.
        out = solve_fzn(
            "var 0..2: x :: output_var;\nvar bool: b :: output_var;\n"
            "constraint int_le(x, 1);\n"
            "solve :: seq_search(["
            "int_search([x], first_fail, indomain_min, complete), "
            "bool_search([b], input_order, indomain_max, complete)]) satisfy;"
        )
        assert "x = 0;" in out

    def test_build_model_search_heuristics_maps_int_search(self) -> None:
        # x is declared first (index 0), y second (index 1); int_search lists y before x
        model = build_model(
            parse(
                "var 0..3: x;\nvar 0..3: y;\nsolve :: int_search([y, x], first_fail, indomain_max, complete) satisfy;"
            )
        )
        result = search_heuristics(model)
        assert result
        assert len(result) == 1  # y and x cover every variable, so there is no trailing catch-all search
        assert result[0].var_heuristic == VAR_HEURISTIC_SMALLEST_DOMAIN
        assert result[0].dom_heuristic == DOM_HEURISTIC_MAX_VALUE
        assert result[0].decision_variables == [1, 0]  # search variables in annotation order (y, x)
        covered = [v for search in result for v in (search.decision_variables or [])]
        assert sorted(covered) == list(range(model.problem.domain_nb))  # every variable is branched

    def test_build_model_search_heuristics_default_without_annotation(self) -> None:
        model = build_model(parse("var 0..3: x;\nsolve satisfy;"))
        assert search_heuristics(model) is DEFAULT_SEARCHES

    def test_build_model_search_heuristics_unknown_selectors_fall_back(self, caplog: pytest.LogCaptureFixture) -> None:
        """A selector NuCS does not implement is replaced, and says so.

        The substitution is invisible in the solution stream -- a model annotated occurrence solved as
        input_order looks like NuCS being slow rather than NuCS solving a different search -- so the
        warning is the only thing standing between that and a wrong benchmark conclusion.
        """
        model = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], occurrence, outdomain_min, complete) satisfy;")
        )
        with caplog.at_level(logging.WARNING, logger="nucs.fzn.search_annotations"):
            result = search_heuristics(model)
        assert result
        assert result[0].var_heuristic == VAR_HEURISTIC_FIRST_NOT_INSTANTIATED  # 'occurrence' has no NuCS equivalent
        assert result[0].dom_heuristic == DOM_HEURISTIC_MIN_VALUE
        assert "occurrence" in caplog.text and "input_order" in caplog.text
        assert "outdomain_min" in caplog.text and "indomain_min" in caplog.text

    @pytest.mark.parametrize(
        "annotation,policy",
        [
            ("", RestartPolicy()),
            (":: restart_none", RestartPolicy()),
            (":: restart_luby(50)", RestartPolicy(RESTART_LUBY, 50)),
            (":: restart_geometric(1.5, 100)", RestartPolicy(RESTART_GEOMETRIC, 100, 1.5)),
            (":: restart_geometric(2, 100)", RestartPolicy(RESTART_GEOMETRIC, 100, 2.0)),  # an int base
            (":: restart_linear(10)", RestartPolicy(RESTART_LINEAR, 10)),
            (":: restart_constant(10)", RestartPolicy(RESTART_CONSTANT, 10)),
            # the restart annotation comes before the search annotation, as MiniZinc writes them
            (
                ":: restart_luby(5) :: int_search([x], dom_w_deg, indomain_min, complete)",
                RestartPolicy(RESTART_LUBY, 5),
            ),
        ],
    )
    def test_restart_policy_of(self, annotation: str, policy: RestartPolicy) -> None:
        model = build_model(parse(f"var 0..3: x;\nsolve {annotation} satisfy;"))
        assert restart_policy_of(model) == policy

    @pytest.mark.parametrize(
        "annotation", [":: restart_luby(0)", ":: restart_geometric(0.5, 10)", ":: restart_fibonacci(3)"]
    )
    def test_restart_policy_of_falls_back_to_no_restart(
        self, annotation: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = build_model(parse(f"var 0..3: x;\nsolve {annotation} satisfy;"))
        with caplog.at_level(logging.WARNING, logger="nucs.fzn.search_annotations"):
            assert restart_policy_of(model) == RestartPolicy()
        assert "without restarts" in caplog.text

    def test_free_search_heuristics(self, caplog: pytest.LogCaptureFixture) -> None:
        # the annotated variables first (y, x), then the rest (z), all with dom/wdeg; the selectors are ignored,
        # so an unsupported one (occurrence) raises no warning
        model = build_model(
            parse(
                "var 0..3: x;\nvar 0..3: y;\nvar 0..3: z;\n"
                "solve :: seq_search([int_search([y], occurrence, indomain_max, complete),"
                " int_search([x, y], input_order, indomain_min, complete)]) satisfy;"
            )
        )
        with caplog.at_level(logging.WARNING, logger="nucs.fzn.search_annotations"):
            searches = free_search_heuristics(model)
        assert caplog.text == ""
        x, y, z = (model.var_index_of(Id(name)) for name in "xyz")
        assert [list(s.decision_variables or []) for s in searches] == [[y, x], [z]]
        assert all(s.var_heuristic == VAR_HEURISTIC_DOM_WDEG for s in searches)

    def test_free_search_heuristics_without_annotation(self) -> None:
        model = build_model(parse("var 0..3: x;\nvar 0..3: y;\nsolve satisfy;"))
        searches = free_search_heuristics(model)
        assert len(searches) == 1
        assert sorted(searches[0].decision_variables or []) == list(range(model.problem.domain_nb))

    @pytest.mark.parametrize(
        "text,policy",
        [
            ("luby", RestartPolicy(RESTART_LUBY)),
            ("luby,500", RestartPolicy(RESTART_LUBY, 500)),
            ("Geometric,100,2.0", RestartPolicy(RESTART_GEOMETRIC, 100, 2.0)),
            ("none", RestartPolicy()),
        ],
    )
    def test_parse_restart(self, text: str, policy: RestartPolicy) -> None:
        assert parse_restart(text) == policy

    @pytest.mark.parametrize("text", ["fibonacci,10", "luby,0", "geometric,10,1.0", "luby,10,2,3"])
    def test_parse_restart_rejects_a_wrong_policy(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_restart(text)

    @pytest.mark.parametrize(
        "free_search,last_conflict,restart,expected",
        [
            # the fixed search: no last-conflict, and the model's restart annotation (restart_luby(7) here)
            (False, None, None, (False, RestartPolicy(RESTART_LUBY, 7))),
            # the free search: its defaults, and the model's restart annotation is ignored
            (True, None, None, (True, FREE_SEARCH_RESTART)),
            # what the command line gives wins, in both modes
            (True, False, RestartPolicy(), (False, RestartPolicy())),
            (False, True, RestartPolicy(RESTART_LUBY, 50), (True, RestartPolicy(RESTART_LUBY, 50))),
        ],
    )
    def test_search_options(
        self,
        free_search: bool,
        last_conflict: bool | None,
        restart: RestartPolicy | None,
        expected: tuple[bool, RestartPolicy],
    ) -> None:
        model = build_model(parse("var 0..3: x;\nsolve :: restart_luby(7) satisfy;"))
        assert search_options(model, free_search, last_conflict, restart) == expected

    def test_build_model_search_heuristics_dom_w_deg(self, caplog: pytest.LogCaptureFixture) -> None:
        model = build_model(parse("var 0..3: x;\nsolve :: int_search([x], dom_w_deg, indomain_min, complete) satisfy;"))
        with caplog.at_level(logging.WARNING, logger="nucs.fzn.search_annotations"):
            result = search_heuristics(model)
        assert result
        assert result[0].var_heuristic == VAR_HEURISTIC_DOM_WDEG
        assert caplog.text == ""

    def test_build_model_search_heuristics_does_not_warn_about_a_supported_selector(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], first_fail, indomain_median, complete) satisfy;")
        )
        with caplog.at_level(logging.WARNING, logger="nucs.fzn.search_annotations"):
            result = search_heuristics(model)
        assert result
        assert result[0].dom_heuristic == DOM_HEURISTIC_MID_VALUE
        assert caplog.text == ""

    def test_build_model_search_heuristics_maps_indomain_random(self) -> None:
        model = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], input_order, indomain_random, complete) satisfy;")
        )
        result = search_heuristics(model)
        assert result
        assert result[0].dom_heuristic == DOM_HEURISTIC_RANDOM_VALUE

    def test_build_model_search_heuristics_maps_smallest_and_largest(self) -> None:
        smallest = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], smallest, indomain_min, complete) satisfy;")
        )
        result = search_heuristics(smallest)
        assert result
        assert result[0].var_heuristic == VAR_HEURISTIC_SMALLEST_MINIMAL_VALUE
        largest = build_model(parse("var 0..3: x;\nsolve :: int_search([x], largest, indomain_min, complete) satisfy;"))
        result = search_heuristics(largest)
        assert result
        assert result[0].var_heuristic == VAR_HEURISTIC_LARGEST_MAXIMAL_VALUE

    def test_solve_fzn_search_annotation_value_heuristic_changes_first_solution(self) -> None:
        out = solve_fzn(
            "var 0..2: x :: output_var;\nsolve :: int_search([x], input_order, indomain_max, complete) satisfy;"
        )
        assert "x = 2;" in out  # indomain_max takes the largest value first
        assert "x = 0;" in solve_fzn("var 0..2: x :: output_var;\nsolve satisfy;")  # default is min

    def test_build_model_search_heuristics_maps_reverse_split(self) -> None:
        model = build_model(
            parse("var 0..3: x;\nsolve :: int_search([x], input_order, indomain_reverse_split, complete) satisfy;")
        )
        result = search_heuristics(model)
        assert result
        assert result[0].dom_heuristic == DOM_HEURISTIC_SPLIT_HIGH

    def test_build_model_search_heuristics_seq_search_keeps_groups_and_appends_catch_all(self) -> None:
        # seq_search becomes one Search per nested search (each with its own selectors), plus a catch-all
        model = build_model(
            parse(
                "var 0..3: a;\nvar 0..3: b;\nvar 0..3: c;\n"
                "solve :: seq_search(["
                "int_search([b], input_order, indomain_max, complete),"
                "int_search([a], first_fail, indomain_min, complete)"
                "]) satisfy;"
            )
        )
        result = search_heuristics(model)
        assert result is not None
        assert len(result) == 3  # the two nested searches and a catch-all for c
        assert result[0].decision_variables == [1]  # b
        assert result[0].var_heuristic == VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
        assert result[0].dom_heuristic == DOM_HEURISTIC_MAX_VALUE
        assert result[1].decision_variables == [0]  # a
        assert result[1].var_heuristic == VAR_HEURISTIC_SMALLEST_DOMAIN
        assert result[1].dom_heuristic == DOM_HEURISTIC_MIN_VALUE
        assert result[2].decision_variables == [2]  # c, the remaining variable, with the defaults
        assert result[2].var_heuristic == VAR_HEURISTIC_FIRST_NOT_INSTANTIATED
        assert result[2].dom_heuristic == DOM_HEURISTIC_MIN_VALUE

    def test_build_model_search_heuristics_flattens_nested_seq_search(self) -> None:
        # MiniZinc nests a seq_search when decomposing e.g. set_search; the nested searches must be flattened
        # in order instead of dropped (which would lose their selectors)
        model = build_model(
            parse(
                "var 0..3: a;\nvar 0..3: b;\n"
                "solve :: seq_search(["
                "int_search([a], input_order, indomain_max, complete),"
                "seq_search([int_search([b], first_fail, indomain_min, complete)])"
                "]) satisfy;"
            )
        )
        result = search_heuristics(model)
        assert result is not None
        assert len(result) == 2  # a and b cover every variable, so no catch-all
        assert result[0].decision_variables == [0]  # a
        assert result[0].dom_heuristic == DOM_HEURISTIC_MAX_VALUE
        assert result[1].decision_variables == [1]  # b, lifted out of the nested seq_search
        assert result[1].var_heuristic == VAR_HEURISTIC_SMALLEST_DOMAIN
        assert result[1].dom_heuristic == DOM_HEURISTIC_MIN_VALUE

    def test_solve_fzn_search_annotation_reverse_split_takes_upper_half_first(self) -> None:
        out = solve_fzn(
            "var 0..3: x :: output_var;\n"
            "solve :: int_search([x], input_order, indomain_reverse_split, complete) satisfy;"
        )
        assert "x = 3;" in out  # reverse split explores the upper half first
