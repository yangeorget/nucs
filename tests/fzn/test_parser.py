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
import pytest

from nucs.fzn.errors import FznUnsupportedError
from nucs.fzn.parser import (
    ArrayAccess,
    ArrayDecl,
    Constraint,
    Id,
    ParDecl,
    Range,
    SetLit,
    Solve,
    VarDecl,
    parse,
)


class TestParser:
    def test_par_decls(self) -> None:
        statements = parse("int: n = 3;\nbool: b = true;\narray [1..3] of int: A = [1, -2, 3];")
        assert statements[0] == ParDecl("n", 3)
        assert statements[1] == ParDecl("b", True)
        # a parameter array is parsed as a (non-var) ArrayDecl, resolved to a constant list by the model
        assert statements[2] == ArrayDecl("A", [1, -2, 3], [], False, None, None, 3)

    def test_array_access_in_constraint(self) -> None:
        statements = parse("constraint int_eq(s[1], s[2]);")
        assert statements[0] == Constraint("int_eq", [ArrayAccess("s", 1), ArrayAccess("s", 2)], [])

    def test_var_array_without_definition_keeps_size(self) -> None:
        statements = parse("array [1..2] of var 0..2: s;")
        assert statements[0] == ArrayDecl("s", [], [], True, 0, 2, 2)

    def test_var_decls_domains(self) -> None:
        statements = parse("var 1..4: x;\nvar bool: b;\nvar int: y;\nvar {1, 2, 3}: z;")
        assert statements[0] == VarDecl("x", 1, 4, [], None)
        assert statements[1] == VarDecl("b", 0, 1, [], None, True)
        y, z = statements[2], statements[3]
        assert isinstance(y, VarDecl) and isinstance(z, VarDecl)
        assert y.lo < 0 < y.hi  # unbounded int falls back to a wide interval
        assert (z.lo, z.hi) == (1, 3)

    def test_var_decl_alias_and_output(self) -> None:
        (decl,) = parse("var 0..9: x :: output_var = y;")
        assert isinstance(decl, VarDecl)
        assert decl.rhs == Id("y")
        assert decl.annotations[0].name == "output_var"

    def test_var_array_decl(self) -> None:
        (decl,) = parse("array [1..3] of var 0..2: q :: output_array([1..3]) = [a, b, c];")
        assert isinstance(decl, ArrayDecl)
        assert decl.is_var and (decl.lo, decl.hi) == (0, 2)
        assert decl.elems == [Id("a"), Id("b"), Id("c")]
        assert decl.annotations[0].name == "output_array"
        index_set = decl.annotations[0].args[0]
        assert isinstance(index_set, list) and index_set[0] == Range(1, 3)

    def test_constraint_and_solve(self) -> None:
        statements = parse("constraint int_lin_eq([1, 1], [x, y], 5);\nsolve minimize x;")
        cons = statements[0]
        assert isinstance(cons, Constraint)
        assert cons.name == "int_lin_eq"
        assert cons.args == [[1, 1], [Id("x"), Id("y")], 5]
        assert statements[1] == Solve("minimize", Id("x"))

    def test_satisfy_and_comments(self) -> None:
        (solve,) = parse("% a comment\nsolve satisfy; % trailing\n")
        assert solve == Solve("satisfy", None)

    def test_solve_search_annotation_is_kept(self) -> None:
        (solve,) = parse("solve :: int_search([x, y], first_fail, indomain_max, complete) satisfy;")
        assert isinstance(solve, Solve) and solve.kind == "satisfy"
        annotation = solve.annotations[0]
        assert annotation.name == "int_search"
        assert annotation.args == [[Id("x"), Id("y")], Id("first_fail"), Id("indomain_max"), Id("complete")]

    def test_predicate_ignored(self) -> None:
        statements = parse("predicate foo(var int: x);\nsolve satisfy;")
        assert statements == [Solve("satisfy", None)]

    def test_unsupported_float(self) -> None:
        with pytest.raises(FznUnsupportedError):
            parse("var float: x;")

    def test_non_contiguous_set_domain(self) -> None:
        # a non-contiguous domain is kept as its interval plus the explicit list of allowed values
        (decl,) = parse("var {1, 3, 5}: x;")
        assert isinstance(decl, VarDecl)
        assert (decl.lo, decl.hi) == (1, 5)
        assert decl.values == [1, 3, 5]

    def test_set_literal_term(self) -> None:
        (cons,) = parse("constraint set_in(x, {1, 3, 5});")
        assert isinstance(cons, Constraint)
        assert cons.name == "set_in"
        assert cons.args == [Id("x"), SetLit([1, 3, 5])]
