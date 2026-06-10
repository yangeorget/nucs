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
The :class:`FznModel` turns a list of parsed FlatZinc statements into a NuCS :class:`Problem`.

It maintains an ordered symbol table mapping FlatZinc identifiers to NuCS variable indices or constants,
allocates NuCS variables on demand, and dispatches each constraint through the builtin registry.
"""

from typing import Dict, List, Optional, Tuple, Union

from nucs.fzn.builtins import BUILTINS
from nucs.fzn.errors import FznParseError, FznUnsupportedError
from nucs.fzn.parser import (
    ArrayAccess,
    ArrayDecl,
    Constraint,
    Id,
    ParDecl,
    Range,
    SetLit,
    Solve,
    Statement,
    Term,
    VarDecl,
)
from nucs.problems.problem import Problem
from nucs.propagators.propagators import ALG_MEMBER


class FznModel:
    """
    A symbol table and NuCS problem built from parsed FlatZinc statements.
    """

    def __init__(self) -> None:
        """
        Inits an empty model.
        """
        self.problem = Problem([])
        self.consts: Dict[str, Union[int, List[int]]] = {}
        self.vars: Dict[str, int] = {}
        self.arrays: Dict[str, List[Term]] = {}
        self.const_var_cache: Dict[int, int] = {}
        # output_items: ("scalar", name, is_bool) or ("array", name, lo, hi, is_bool)
        self.output_items: List[Tuple] = []
        self.solve: Solve = Solve("satisfy")

    def build(self, statements: List[Statement]) -> "FznModel":
        """
        Builds the model from parsed statements: a declaration pass then a constraint pass.

        :param statements: the parsed statements
        :type statements: List[Statement]

        :return: this model
        :rtype: FznModel
        """
        constraints: List[Constraint] = []
        for statement in statements:
            if isinstance(statement, ParDecl):
                self.consts[statement.name] = statement.value  # type: ignore[assignment]
            elif isinstance(statement, VarDecl):
                self._declare_var(statement)
            elif isinstance(statement, ArrayDecl):
                self._declare_array(statement)
            elif isinstance(statement, Solve):
                self.solve = statement
            elif isinstance(statement, Constraint):
                constraints.append(statement)
        # The constraint pass happens after every declaration so that forward references resolve.
        for constraint in constraints:
            handler = BUILTINS.get(constraint.name)
            if handler is None:
                raise FznUnsupportedError(f"constraint '{constraint.name}' is not supported")
            handler(self, constraint.args)
        return self

    def _declare_var(self, decl: VarDecl) -> None:
        """
        Declares a scalar variable, resolving literal assignments to constants and identifier
        assignments to aliases.

        :param decl: the variable declaration
        :type decl: VarDecl
        """
        if isinstance(decl.rhs, bool):
            self.consts[decl.name] = int(decl.rhs)
        elif isinstance(decl.rhs, int):
            self.consts[decl.name] = decl.rhs
        elif isinstance(decl.rhs, Id):
            if decl.rhs.name in self.vars:
                self.vars[decl.name] = self.vars[decl.rhs.name]
            elif decl.rhs.name in self.consts:
                self.consts[decl.name] = self.consts[decl.rhs.name]
            else:
                raise FznParseError(f"unknown identifier '{decl.rhs.name}'")
        else:
            index = self.problem.add_variable((decl.lo, decl.hi))
            self.vars[decl.name] = index
            if decl.values is not None:
                # A non-contiguous domain is stored as its interval plus a member constraint for the holes.
                self.problem.add_propagator(ALG_MEMBER, [index], decl.values)
        for ann in decl.annotations:
            if ann.name == "output_var":
                self.output_items.append(("scalar", decl.name, decl.is_bool))

    def _declare_array(self, decl: ArrayDecl) -> None:
        """
        Declares an array of parameters or variables.

        :param decl: the array declaration
        :type decl: ArrayDecl
        """
        if decl.is_var and not decl.elems:
            # An array of fresh variables, accessed as name[i]; allocate one NuCS variable per element.
            if decl.size is None or decl.lo is None or decl.hi is None:
                raise FznUnsupportedError(f"variable array '{decl.name}' without a known size is not supported")
            elems: List[Term] = []
            for i in range(decl.size):
                elem_name = f"{decl.name}[{i + 1}]"
                index = self.problem.add_variable((decl.lo, decl.hi))
                self.vars[elem_name] = index
                if decl.values is not None:
                    # Each element of a non-contiguous element domain gets its own member constraint.
                    self.problem.add_propagator(ALG_MEMBER, [index], decl.values)
                elems.append(Id(elem_name))
            self.arrays[decl.name] = elems
        elif decl.is_var:
            self.arrays[decl.name] = decl.elems
        else:
            self.consts[decl.name] = [int(e) for e in decl.elems]  # type: ignore[arg-type]
            self.arrays[decl.name] = decl.elems
        for ann in decl.annotations:
            if ann.name == "output_array":
                lo, hi = _index_set_bounds(decl, ann)
                self.output_items.append(("array", decl.name, lo, hi, decl.is_bool))

    def var_index_of(self, term: Term) -> int:
        """
        Returns a NuCS variable index for any term, creating a cached singleton-domain variable for an
        integer constant.

        :param term: the term to resolve
        :type term: Term

        :return: a NuCS variable index
        :rtype: int
        """
        term = self._deref(term)
        if isinstance(term, bool):
            return self._const_var(int(term))
        if isinstance(term, int):
            return self._const_var(term)
        if isinstance(term, Id):
            if term.name in self.vars:
                return self.vars[term.name]
            if term.name in self.consts and isinstance(self.consts[term.name], int):
                return self._const_var(int(self.consts[term.name]))  # type: ignore[arg-type]
            raise FznParseError(f"'{term.name}' is not a scalar variable")
        raise FznParseError(f"cannot use {term!r} as a variable")

    def _const_var(self, value: int) -> int:
        """
        Returns a cached NuCS variable bound to a single constant value.

        :param value: the constant value
        :type value: int

        :return: the NuCS variable index
        :rtype: int
        """
        if value not in self.const_var_cache:
            self.const_var_cache[value] = self.problem.add_variable((value, value))
        return self.const_var_cache[value]

    def const_of(self, term: Term) -> int:
        """
        Resolves a term to a scalar integer constant.

        :param term: the term to resolve
        :type term: Term

        :return: the constant value
        :rtype: int
        """
        term = self._deref(term)
        if isinstance(term, bool):
            return int(term)
        if isinstance(term, int):
            return term
        if isinstance(term, Id) and term.name in self.consts and isinstance(self.consts[term.name], int):
            return int(self.consts[term.name])  # type: ignore[arg-type]
        raise FznUnsupportedError(f"expected an integer constant, got {term!r}")

    def var_list_of(self, term: Term) -> List[int]:
        """
        Resolves an array term (an identifier or an inline literal) to a list of NuCS variable indices.

        :param term: the term to resolve
        :type term: Term

        :return: the list of NuCS variable indices
        :rtype: List[int]
        """
        return [self.var_index_of(e) for e in self._elements_of(term)]

    def int_list_of(self, term: Term) -> List[int]:
        """
        Resolves an array term to a list of integer constants.

        :param term: the term to resolve
        :type term: Term

        :return: the list of constants
        :rtype: List[int]
        """
        return [self.const_of(e) for e in self._elements_of(term)]

    def set_values_of(self, term: Term) -> List[int]:
        """
        Resolves a set term (a ``{..}`` literal or a ``lo..hi`` range) to its sorted list of values.

        :param term: the term to resolve
        :type term: Term

        :return: the allowed values, in strictly ascending order
        :rtype: List[int]
        """
        term = self._deref(term)
        if isinstance(term, SetLit):
            return term.values
        if isinstance(term, Range):
            return list(range(term.lo, term.hi + 1))
        raise FznUnsupportedError(f"expected a set, got {term!r}")

    def value_of(self, term: Term, solution) -> int:  # type: ignore[no-untyped-def]
        """
        Resolves a term to its concrete value in a solution, without allocating any variable.

        :param term: the term to resolve
        :type term: Term
        :param solution: the solution array indexed by NuCS variable
        :type solution: NDArray

        :return: the value of the term
        :rtype: int
        """
        term = self._deref(term)
        if isinstance(term, bool):
            return int(term)
        if isinstance(term, int):
            return term
        if isinstance(term, Id):
            if term.name in self.vars:
                return int(solution[self.vars[term.name]])
            if term.name in self.consts and isinstance(self.consts[term.name], int):
                return int(self.consts[term.name])  # type: ignore[arg-type]
        raise FznParseError(f"cannot resolve value of {term!r}")

    def elements_of(self, term: Term) -> List[Term]:
        """
        Returns the element terms of an array term, public wrapper around the internal resolver.

        :param term: the array term
        :type term: Term

        :return: the element terms
        :rtype: List[Term]
        """
        return self._elements_of(term)

    def _deref(self, term: Term) -> Term:
        """
        Resolves an array element access ``name[index]`` to its underlying element term, leaving any other
        term unchanged.

        :param term: the term to dereference
        :type term: Term

        :return: the dereferenced term
        :rtype: Term
        """
        if isinstance(term, ArrayAccess):
            if term.name not in self.arrays:
                raise FznParseError(f"'{term.name}' is not an array")
            elems = self.arrays[term.name]
            if not 1 <= term.index <= len(elems):
                raise FznParseError(f"array index {term.index} out of bounds for '{term.name}'")
            return elems[term.index - 1]
        return term

    def _elements_of(self, term: Term) -> List[Term]:
        """
        Returns the element terms of an array term (an inline literal or a named array).

        :param term: the array term
        :type term: Term

        :return: the element terms
        :rtype: List[Term]
        """
        if isinstance(term, list):
            return term
        if isinstance(term, Id) and term.name in self.arrays:
            return self.arrays[term.name]
        raise FznUnsupportedError(f"expected an array, got {term!r}")


def _index_set_bounds(decl: ArrayDecl, ann) -> Tuple[int, int]:  # type: ignore[no-untyped-def]
    """
    Returns the (lo, hi) index-set bounds for an output_array annotation, falling back to ``1..len``.

    :param decl: the array declaration
    :type decl: ArrayDecl
    :param ann: the output_array annotation
    :type ann: Ann

    :return: the index-set bounds
    :rtype: Tuple[int, int]
    """
    if ann.args and isinstance(ann.args[0], list) and ann.args[0] and isinstance(ann.args[0][0], Range):
        rng = ann.args[0][0]
        return rng.lo, rng.hi
    return 1, len(decl.elems)


def build_model(statements: List[Statement]) -> FznModel:
    """
    Builds a :class:`FznModel` from parsed statements.

    :param statements: the parsed statements
    :type statements: List[Statement]

    :return: the built model
    :rtype: FznModel
    """
    return FznModel().build(statements)


# Re-exported for callers that resolve the objective term after building.
Objective = Optional[Term]
