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
allocates one NuCS variable per class of always-equal FlatZinc variables, and dispatches each constraint
through the builtin registry.
"""

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

# The arity-2 builtins stating that two variables are always equal, which an equality class represents
# for free. int_lin_eq states it too, for the shape MiniZinc emits when it cannot alias at flattening
# time; see _alias_operands.
_ALIAS_BUILTINS = ("bool2int", "bool_eq", "int_eq")


class FznModel:
    """
    A symbol table and NuCS problem built from parsed FlatZinc statements.
    """

    def __init__(self) -> None:
        """
        Inits an empty model.
        """
        self.problem = Problem([])
        self.consts: dict[str, int | list[int]] = {}
        self.vars: dict[str, int] = {}
        self.arrays: dict[str, list[Term]] = {}
        self.const_var_cache: dict[int, int] = {}
        # Variable allocation is deferred until the equality classes are known (see build), so declaring a
        # variable only records it here: the declaration order of pending_names fixes the allocation order.
        self.pending_names: list[str] = []
        self.alias_parent: dict[str, str] = {}  # union-find over declared variable names
        self.alias_domain: dict[str, tuple[int, int, list[int] | None]] = {}  # (lo, hi, values) per class root
        # Equalities from `var ...: b = a;` declarations whose domains turned out to be disjoint.
        self.deferred_constraints: list[Constraint] = []
        # output_items: ("scalar", name, is_bool) or ("array", name, lo, hi, is_bool)
        self.output_items: list[tuple] = []
        self.solve: Solve = Solve("satisfy")

    def build(self, statements: list[Statement]) -> "FznModel":
        """
        Builds the model from parsed statements: a declaration pass, an aliasing pass, the allocation of one
        NuCS variable per equality class, then a constraint pass.

        The declaration pass only records variables, because ``bool2int``/``bool_eq``/``int_eq`` state that
        two of them are always equal: a bounds solver represents such a pair as a single variable, so
        allocating eagerly would waste both a variable and the propagator channelling it. The aliasing pass
        therefore runs first and collapses those constraints into equality classes.

        :param statements: the parsed statements
        :type statements: List[Statement]

        :return: this model
        :rtype: FznModel
        """
        constraints: list[Constraint] = []
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
        # Both passes below happen after every declaration so that forward references resolve.
        constraints = self._absorb_aliases(constraints) + self.deferred_constraints
        self._allocate_variables()
        for constraint in constraints:
            handler = BUILTINS.get(constraint.name)
            if handler is None:
                raise FznUnsupportedError(f"constraint '{constraint.name}' is not supported")
            handler(self, constraint.args)
        return self

    def _find(self, name: str) -> str:
        """
        Returns the representative of a declared variable's equality class, compressing the path to it.

        :param name: the variable name
        :type name: str

        :return: the name representing the class
        :rtype: str
        """
        root = name
        while self.alias_parent[root] != root:
            root = self.alias_parent[root]
        while self.alias_parent[name] != root:
            self.alias_parent[name], name = root, self.alias_parent[name]
        return root

    def _union(self, name: str, other: str) -> bool:
        """
        Merges the equality classes of two declared variables, intersecting their domains.

        :param name: a variable name
        :type name: str
        :param other: the name of a variable always equal to it
        :type other: str

        :return: False when the two domains do not intersect, leaving the classes untouched
        :rtype: bool
        """
        root, other_root = self._find(name), self._find(other)
        if root == other_root:
            return True
        domain, other_domain = self.alias_domain.get(root), self.alias_domain.get(other_root)
        if domain is None or other_domain is None:  # defensive: every declared name carries a domain
            merged = other_domain if domain is None else domain
        else:
            intersection = _intersect_domains(domain, other_domain)
            if intersection is None:
                return False
            merged = intersection
        self.alias_parent[other_root] = root
        self.alias_domain.pop(other_root, None)
        if merged is not None:
            self.alias_domain[root] = merged
        return True

    def _declare_pending(self, name: str, lo: int, hi: int, values: list[int] | None) -> None:
        """
        Records a variable to allocate once its equality class is known.

        :param name: the variable name
        :type name: str
        :param lo: the lower bound of its domain
        :type lo: int
        :param hi: the upper bound of its domain
        :type hi: int
        :param values: the explicit allowed values of a non-contiguous domain, or None
        :type values: Optional[List[int]]
        """
        self.alias_parent[name] = name
        self.alias_domain[name] = (lo, hi, values)
        self.pending_names.append(name)

    def _absorb_aliases(self, constraints: list[Constraint]) -> list[Constraint]:
        """
        Merges the equality classes of every constraint stating that two declared variables are equal, and
        returns the constraints that remain to be posted.

        A constraint is only absorbed when both operands are declared variables -- an operand fixed to a
        constant is left to its regular handler -- and when their domains intersect, so an unsatisfiable
        equality is reported by propagation rather than silently dropped.

        :param constraints: the parsed constraints
        :type constraints: List[Constraint]

        :return: the constraints that were not absorbed
        :rtype: List[Constraint]
        """
        kept = []
        for constraint in constraints:
            operands = self._alias_operands(constraint)
            if operands is not None and self._union(*operands):
                continue
            kept.append(constraint)
        return kept

    def _alias_operands(self, constraint: Constraint) -> tuple[str, str] | None:
        """
        Returns the two declared variable names a constraint states to be always equal, or None when it
        states no such thing.

        Besides ``bool2int`` / ``bool_eq`` / ``int_eq``, this recognises ``int_lin_eq([1, -1], [x, y], 0)``
        (and the negated coefficient order), which is how MiniZinc writes x = y whenever it could not
        collapse the two variables itself while flattening.

        A non-zero right-hand side is deliberately not absorbed: ``x - y = c`` makes y an *offset* of x,
        which an equality class cannot express -- representing it would need a view mechanism NuCS does not
        have. Only operands that are both declared variables qualify; a constant one is left to the regular
        handler.

        :param constraint: the parsed constraint
        :type constraint: Constraint

        :return: the pair of variable names, or None
        :rtype: Optional[Tuple[str, str]]
        """
        if constraint.name in _ALIAS_BUILTINS and len(constraint.args) == 2:
            terms = [self._deref(constraint.args[0]), self._deref(constraint.args[1])]
        elif constraint.name == "int_lin_eq" and len(constraint.args) == 3:
            if self.const_of(constraint.args[2]) != 0 or self.int_list_of(constraint.args[0]) not in (
                [1, -1],
                [-1, 1],
            ):
                return None
            terms = [self._deref(term) for term in self._elements_of(constraint.args[1])]
        else:
            return None
        if len(terms) != 2:
            return None
        left, right = terms
        if (
            isinstance(left, Id)
            and isinstance(right, Id)
            and left.name in self.alias_parent
            and right.name in self.alias_parent
        ):
            return left.name, right.name
        return None

    def _allocate_variables(self) -> None:
        """
        Allocates one NuCS variable per equality class, in declaration order, and points every name of a
        class at it.
        """
        allocated: dict[str, int] = {}
        for name in self.pending_names:
            root = self._find(name)
            if root not in allocated:
                lo, hi, values = self.alias_domain[root]
                index = self.problem.add_variable((lo, hi))
                allocated[root] = index
                if values is not None:
                    # A non-contiguous domain is stored as its interval plus a member constraint for the holes.
                    self.problem.add_propagator(ALG_MEMBER, [index], values)
        for name in self.alias_parent:
            self.vars[name] = allocated[self._find(name)]

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
            if decl.rhs.name in self.alias_parent:
                # An alias joins the class of its right-hand side, contributing its own declared domain:
                # `var 0..2: b = a;` constrains a just as much as it constrains b.
                self._declare_pending(decl.name, decl.lo, decl.hi, decl.values)
                if not self._union(decl.name, decl.rhs.name):
                    # The two declared domains are disjoint, so the class cannot represent both. They are
                    # left as separate variables and the equality is deferred to a propagator, which refutes
                    # it -- the model is unsatisfiable, and must be reported as such rather than dropped.
                    self.deferred_constraints.append(Constraint("int_eq", [Id(decl.name), Id(decl.rhs.name)]))
            elif decl.rhs.name in self.consts:
                self.consts[decl.name] = self.consts[decl.rhs.name]
            else:
                raise FznParseError(f"unknown identifier '{decl.rhs.name}'")
        else:
            self._declare_pending(decl.name, decl.lo, decl.hi, decl.values)
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
            elems: list[Term] = []
            for i in range(decl.size):
                elem_name = f"{decl.name}[{i + 1}]"
                self._declare_pending(elem_name, decl.lo, decl.hi, decl.values)
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

    def var_list_of(self, term: Term) -> list[int]:
        """
        Resolves an array term (an identifier or an inline literal) to a list of NuCS variable indices.

        :param term: the term to resolve
        :type term: Term

        :return: the list of NuCS variable indices
        :rtype: List[int]
        """
        return [self.var_index_of(e) for e in self._elements_of(term)]

    def int_list_of(self, term: Term) -> list[int]:
        """
        Resolves an array term to a list of integer constants.

        :param term: the term to resolve
        :type term: Term

        :return: the list of constants
        :rtype: List[int]
        """
        return [self.const_of(e) for e in self._elements_of(term)]

    def set_values_of(self, term: Term) -> list[int]:
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

    def elements_of(self, term: Term) -> list[Term]:
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

    def _elements_of(self, term: Term) -> list[Term]:
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


def _intersect_values(values: list[int] | None, other: list[int] | None, lo: int, hi: int) -> list[int] | None:
    """
    Intersects two lists of allowed values, restricted to the interval lo..hi.

    A None list means every value of the interval is allowed. The interval is never materialised, so an
    unbounded domain (whose interval spans the whole default range) stays cheap.

    :param values: the allowed values, or None
    :type values: Optional[List[int]]
    :param other: the other allowed values, or None
    :type other: Optional[List[int]]
    :param lo: the lower bound to restrict to
    :type lo: int
    :param hi: the upper bound to restrict to
    :type hi: int

    :return: the allowed values in ascending order, or None when the interval is not restricted
    :rtype: Optional[List[int]]
    """
    if values is None:
        if other is None:
            return None
        return [value for value in other if lo <= value <= hi]
    if other is None:
        return [value for value in values if lo <= value <= hi]
    allowed = set(other)
    return [value for value in values if lo <= value <= hi and value in allowed]


def _intersect_domains(
    domain: tuple[int, int, list[int] | None], other: tuple[int, int, list[int] | None]
) -> tuple[int, int, list[int] | None] | None:
    """
    Intersects two variable domains, each an interval plus an optional list of allowed values.

    :param domain: a (lo, hi, values) domain
    :type domain: Tuple[int, int, Optional[List[int]]]
    :param other: the other (lo, hi, values) domain
    :type other: Tuple[int, int, Optional[List[int]]]

    :return: the intersected domain, or None when the two do not intersect
    :rtype: Optional[Tuple[int, int, Optional[List[int]]]]
    """
    lo = max(domain[0], other[0])
    hi = min(domain[1], other[1])
    if lo > hi:
        return None
    values = _intersect_values(domain[2], other[2], lo, hi)
    if values is None:
        return lo, hi, None
    if not values:
        return None
    lo, hi = values[0], values[-1]
    # A contiguous set is exactly its interval, so no member constraint is needed.
    return lo, hi, None if len(values) == hi - lo + 1 else values


def _index_set_bounds(decl: ArrayDecl, ann) -> tuple[int, int]:  # type: ignore[no-untyped-def]
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


def build_model(statements: list[Statement]) -> FznModel:
    """
    Builds a :class:`FznModel` from parsed statements.

    :param statements: the parsed statements
    :type statements: List[Statement]

    :return: the built model
    :rtype: FznModel
    """
    return FznModel().build(statements)
