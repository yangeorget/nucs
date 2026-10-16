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


from nucs.fzn.model import build_model
from nucs.fzn.parser import parse
from nucs.propagators.propagators import ALG_EQ, ALG_LINEAR_EQ_C
from tests.fzn.fzn_test import solve_fzn


class TestModel:
    def test_equal_variables_share_one_variable(self) -> None:
        # int_eq/bool_eq/bool2int between two declared variables is represented by a single NuCS variable,
        # so neither the channelling propagator nor the second variable is created
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\nvar bool: b;\nvar 0..1: i;\n"
                "constraint int_eq(x, y);\nconstraint bool2int(b, i);\nsolve satisfy;"
            )
        )
        assert model.problem.propagators == []
        assert model.problem.domain_nb == 2  # {x, y} and {b, i}
        assert model.vars["x"] == model.vars["y"]
        assert model.vars["b"] == model.vars["i"]

    def test_build_model_aliasing_is_transitive(self) -> None:
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\nvar 0..9: z;\n"
                "constraint int_eq(x, y);\nconstraint int_eq(y, z);\nsolve satisfy;"
            )
        )
        assert model.problem.domain_nb == 1
        assert model.vars["x"] == model.vars["y"] == model.vars["z"]

    def test_build_model_aliasing_intersects_domains(self) -> None:
        # the shared variable gets the intersection of the two declared domains, which is at least as tight
        # as what the channelling propagator would have derived
        model = build_model(parse("var 0..9: x;\nvar 4..20: y;\nconstraint int_eq(x, y);\nsolve satisfy;"))
        assert model.problem.domain_nb == 1
        assert list(model.problem.domains[model.vars["x"]]) == [4, 9]

    def test_build_model_aliasing_intersects_non_contiguous_domains(self) -> None:
        model = build_model(parse("var {1, 3, 5, 7}: x;\nvar {3, 4, 5}: y;\nconstraint int_eq(x, y);\nsolve satisfy;"))
        assert model.problem.domain_nb == 1
        assert list(model.problem.domains[model.vars["x"]]) == [3, 5]
        assert model.problem.propagators[0][2] == [3, 5]  # the hole at 4 still needs a member constraint

    def test_build_model_disjoint_domains_are_not_aliased(self) -> None:
        # merging would need an empty domain, so the equality is left to propagation to refute
        model = build_model(parse("var 0..3: x;\nvar 5..9: y;\nconstraint int_eq(x, y);\nsolve satisfy;"))
        assert model.problem.domain_nb == 2
        assert [prop[1] for prop in model.problem.propagators] == [ALG_EQ]

    def test_solve_fzn_disjoint_domains_stay_unsatisfiable(self) -> None:
        assert "=====UNSATISFIABLE=====" in solve_fzn(
            "var 0..3: x;\nvar 5..9: y;\nconstraint int_eq(x, y);\nsolve satisfy;"
        )

    def test_solve_fzn_aliased_variables_both_print_in_their_own_type(self) -> None:
        # b and i share a variable but keep their own output formatting
        out = solve_fzn(
            "var bool: b :: output_var;\nvar 0..1: i :: output_var;\n"
            "constraint bool2int(b, i);\nconstraint int_le(1, i);\nsolve satisfy;"
        )
        assert "b = true;" in out
        assert "i = 1;" in out

    def test_solve_fzn_aliasing_preserves_the_solution_set(self) -> None:
        out = solve_fzn(
            "var 0..2: x :: output_var;\nvar 0..2: y :: output_var;\nvar 1..3: z :: output_var;\n"
            "constraint int_eq(x, y);\nconstraint int_lin_eq([1, 1], [y, z], 3);\n"
            "solve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3  # y + z = 3 with y in 0..2, z in 1..3, and x tracking y
        assert "x = 0;\ny = 0;\nz = 3;" in out
        assert "x = 1;\ny = 1;\nz = 2;" in out
        assert "x = 2;\ny = 2;\nz = 1;" in out

    def test_solve_fzn_non_contiguous_var_domain(self) -> None:
        out = solve_fzn(
            "var {1, 3, 5}: x :: output_var;\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3
        assert "x = 1;" in out and "x = 3;" in out and "x = 5;" in out

    def test_solve_fzn_alias_declaration_narrows_its_right_hand_side(self) -> None:
        # `var 0..2: b = a;` constrains a as much as it constrains b: the equality class takes the
        # intersection of the two declared domains, so a is 0..2 and not 0..10
        out = solve_fzn(
            "var 0..10: a :: output_var;\nvar 0..2: b = a;\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 3
        assert "a = 3;" not in out

    def test_solve_fzn_alias_declaration_does_not_widen_its_right_hand_side(self) -> None:
        # the intersection cannot widen either: a stays 0..10 under `var 0..100: b = a;`
        out = solve_fzn(
            "var 0..10: a :: output_var;\nvar 0..100: b = a;\nconstraint int_le(4, a);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 7  # 4..10
        assert "a = 11;" not in out

    def test_solve_fzn_alias_declaration_with_a_disjoint_domain_is_unsatisfiable(self) -> None:
        # the class cannot represent both domains, so the two stay separate variables and the equality is
        # left to a propagator -- which refutes it, rather than the constraint being silently dropped
        out = solve_fzn("var 0..10: a :: output_var;\nvar 20..30: b = a;\nsolve satisfy;")
        assert out.strip() == "=====UNSATISFIABLE====="

    def test_solve_fzn_alias_declaration_domain_reaches_the_other_constraints(self) -> None:
        # the reproduction from the review: b = a with b in 0..2 forces a <= 2, contradicting a >= 4
        out = solve_fzn(
            "var 0..10: a :: output_var;\nvar 0..2: b = a;\nconstraint int_le(4, a);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.strip() == "=====UNSATISFIABLE====="

    def test_build_model_unit_linear_equality_is_absorbed_as_an_alias(self) -> None:
        # int_lin_eq([1, -1], [x, y], 0) states x = y, so the two share one variable and no propagator is
        # posted -- as for int_eq. MiniZinc 2.9.x aliases this shape away itself, so it reaches NuCS only
        # from hand-written FlatZinc or another flattener.
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\narray [1..2] of var int: v = [x, y];\n"
                "constraint int_lin_eq([1, -1], v, 0);\nsolve satisfy;"
            )
        )
        assert len(model.problem.domains) == 1
        assert model.problem.propagators == []
        assert model.vars["x"] == model.vars["y"]

    def test_build_model_offset_linear_equality_is_not_absorbed(self) -> None:
        # a non-zero right-hand side makes y an offset of x, which an equality class cannot express: the
        # constraint must stay a propagator over two distinct variables
        model = build_model(
            parse(
                "var 0..9: x;\nvar 0..9: y;\narray [1..2] of var int: v = [x, y];\n"
                "constraint int_lin_eq([1, -1], v, 3);\nsolve satisfy;"
            )
        )
        assert len(model.problem.domains) == 2
        assert [prop[1] for prop in model.problem.propagators] == [ALG_LINEAR_EQ_C]

    def test_solve_fzn_unit_linear_equality_alias_keeps_the_solution_set(self) -> None:
        # the absorbed form must constrain exactly as the propagator would: x = y over 0..3 with x >= 2
        out = solve_fzn(
            "var 0..3: x :: output_var;\nvar 0..3: y :: output_var;\narray [1..2] of var int: v = [x, y];\n"
            "constraint int_lin_eq([-1, 1], v, 0);\nconstraint int_le(2, x);\nsolve satisfy;",
            all_solutions=True,
        )
        assert out.count("----------") == 2
        assert "x = 2;\ny = 2;" in out and "x = 3;\ny = 3;" in out
