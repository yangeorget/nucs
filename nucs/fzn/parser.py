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
A hand-rolled tokenizer and recursive-descent parser for the FlatZinc subset that NuCS supports.

The parser turns FlatZinc text into a flat list of statement IR objects (:class:`ParDecl`,
:class:`VarDecl`, :class:`ArrayDecl`, :class:`Constraint`, :class:`Solve`). It performs no semantic
resolution: identifiers are kept as :class:`Id` and ranges as :class:`Range`; the model layer turns these
into NuCS variables and constants.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from nucs.fzn.errors import FznParseError, FznUnsupportedError

# A wide finite fallback for unbounded "var int" declarations (NuCS only supports finite interval domains).
DEFAULT_INT_MIN = -(1 << 27)
DEFAULT_INT_MAX = 1 << 27


@dataclass
class Id:
    """
    A reference to a FlatZinc identifier.
    """

    name: str


@dataclass
class Range:
    """
    An integer range literal ``lo..hi``.
    """

    lo: int
    hi: int


@dataclass
class ArrayAccess:
    """
    An array element access ``name[index]`` with a 1-based integer index.
    """

    name: str
    index: int


@dataclass
class SetLit:
    """
    A set literal ``{v0, v1, ...}`` of integer constants, kept in strictly ascending order.
    """

    values: List[int]


# A term is an int, a bool, an identifier, a range, an array access, a set literal, or a (possibly nested)
# list of terms.
Term = Union[int, bool, Id, Range, "ArrayAccess", "SetLit", list]


@dataclass
class Ann:
    """
    A FlatZinc annotation ``name`` or ``name(args)``.
    """

    name: str
    args: List[Term] = field(default_factory=list)


@dataclass
class ParDecl:
    """
    A parameter declaration ``<type>: name = value``.
    """

    name: str
    value: Union[int, bool, List[int]]


@dataclass
class VarDecl:
    """
    A scalar variable declaration ``var <domain>: name [annotations] [= rhs]``.
    """

    name: str
    lo: int
    hi: int
    annotations: List[Ann] = field(default_factory=list)
    rhs: Optional[Term] = None
    is_bool: bool = False
    values: Optional[List[int]] = None  # explicit allowed values for a non-contiguous domain


@dataclass
class ArrayDecl:
    """
    An array declaration, either of parameters or of variables.
    """

    name: str
    elems: List[Term]
    annotations: List[Ann] = field(default_factory=list)
    is_var: bool = False
    lo: Optional[int] = None
    hi: Optional[int] = None
    size: Optional[int] = None
    is_bool: bool = False
    values: Optional[List[int]] = None  # explicit allowed values for a non-contiguous element domain


@dataclass
class Constraint:
    """
    A constraint call ``constraint name(args) [annotations]``.
    """

    name: str
    args: List[Term]
    annotations: List[Ann] = field(default_factory=list)


@dataclass
class Solve:
    """
    A solve item: ``satisfy``, ``minimize <expr>`` or ``maximize <expr>``.
    """

    kind: str
    objective: Optional[Term] = None


Statement = Union[ParDecl, VarDecl, ArrayDecl, Constraint, Solve]

# Multi-character punctuation, longest first so that "::" and ".." are matched before ":" and ".".
_PUNCT2 = ("::", "..")
_PUNCT1 = set(":;,()[]{}=")


def tokenize(text: str) -> List[Tuple[str, object]]:
    """
    Splits FlatZinc text into a list of ``(kind, value)`` tokens.

    :param text: the FlatZinc source
    :type text: str

    :return: a list of tokens, each a pair of a kind (INT, IDENT, PUNCT, STRING) and a value
    :rtype: List[Tuple[str, object]]
    """
    tokens: List[Tuple[str, object]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "%":  # line comment
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == '"':  # string literal
            i += 1
            start = i
            while i < n and text[i] != '"':
                i += 1
            tokens.append(("STRING", text[start:i]))
            i += 1
            continue
        if c == "-" and i + 1 < n and text[i + 1].isdigit():
            j = i + 1
            while j < n and text[j].isdigit():
                j += 1
            tokens.append(("INT", int(text[i:j])))
            i = j
            continue
        if c.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            tokens.append(("INT", int(text[i:j])))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("IDENT", text[i:j]))
            i = j
            continue
        if text[i : i + 2] in _PUNCT2:
            tokens.append(("PUNCT", text[i : i + 2]))
            i += 2
            continue
        if c in _PUNCT1:
            tokens.append(("PUNCT", c))
            i += 1
            continue
        raise FznParseError(f"unexpected character {c!r} at offset {i}")
    return tokens


class Parser:
    """
    A recursive-descent parser over a token list produced by :func:`tokenize`.
    """

    def __init__(self, tokens: List[Tuple[str, object]]) -> None:
        """
        Inits the parser.

        :param tokens: the tokens to parse
        :type tokens: List[Tuple[str, object]]
        """
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Tuple[str, object]]:
        """
        Returns the current token without consuming it, or None at the end of input.

        :return: the current token or None
        :rtype: Optional[Tuple[str, object]]
        """
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> Tuple[str, object]:
        """
        Consumes and returns the current token.

        :return: the consumed token
        :rtype: Tuple[str, object]
        """
        if self.pos >= len(self.tokens):
            raise FznParseError("unexpected end of input")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, kind: str, value: Optional[object] = None) -> object:
        """
        Consumes the current token, asserting its kind and optionally its value.

        :param kind: the expected token kind
        :type kind: str
        :param value: the expected token value, or None to accept any
        :type value: Optional[object]

        :return: the value of the consumed token
        :rtype: object
        """
        tk, tv = self.next()
        if tk != kind or (value is not None and tv != value):
            raise FznParseError(f"expected {kind} {value if value is not None else ''}, got {tk} {tv!r}")
        return tv

    def accept(self, kind: str, value: object) -> bool:
        """
        Consumes the current token if it matches the given kind and value.

        :param kind: the token kind to match
        :type kind: str
        :param value: the token value to match
        :type value: object

        :return: True if a token was consumed
        :rtype: bool
        """
        tok = self.peek()
        if tok is not None and tok[0] == kind and tok[1] == value:
            self.pos += 1
            return True
        return False

    def expect_int(self) -> int:
        """
        Consumes the current token, asserting it is an integer literal.

        :return: the integer value
        :rtype: int
        """
        value = self.expect("INT")
        assert isinstance(value, int)
        return value

    def parse(self) -> List[Statement]:
        """
        Parses the whole token stream into a list of statements.

        :return: the parsed statements
        :rtype: List[Statement]
        """
        statements: List[Statement] = []
        while self.peek() is not None:
            statement = self.parse_statement()
            if statement is not None:
                statements.append(statement)
        return statements

    def parse_statement(self) -> Optional[Statement]:
        """
        Parses a single top-level statement.

        :return: the parsed statement, or None for ignored statements (e.g. predicate declarations)
        :rtype: Optional[Statement]
        """
        tk, tv = self.peek()  # type: ignore[misc]
        if tk != "IDENT":
            raise FznParseError(f"expected a statement keyword, got {tk} {tv!r}")
        if tv == "predicate":
            while not self.accept("PUNCT", ";"):
                self.next()
            return None
        if tv == "constraint":
            return self.parse_constraint()
        if tv == "solve":
            return self.parse_solve()
        if tv == "array":
            return self.parse_array_decl()
        if tv == "var":
            return self.parse_var_decl()
        return self.parse_par_decl()

    def parse_annotations(self) -> List[Ann]:
        """
        Parses a (possibly empty) sequence of ``:: annotation`` items.

        :return: the parsed annotations
        :rtype: List[Ann]
        """
        annotations: List[Ann] = []
        while self.accept("PUNCT", "::"):
            name = self.expect("IDENT")
            args: List[Term] = []
            if self.accept("PUNCT", "("):
                args = self.parse_arg_list()
                self.expect("PUNCT", ")")
            annotations.append(Ann(str(name), args))
        return annotations

    def parse_arg_list(self) -> List[Term]:
        """
        Parses a comma-separated list of terms up to (but not consuming) the closing delimiter.

        :return: the parsed terms
        :rtype: List[Term]
        """
        args: List[Term] = []
        if self.peek() == ("PUNCT", ")") or self.peek() == ("PUNCT", "]"):
            return args
        args.append(self.parse_term())
        while self.accept("PUNCT", ","):
            args.append(self.parse_term())
        return args

    def parse_term(self) -> Term:
        """
        Parses a term: an int, a bool, an identifier, a range ``lo..hi``, or an array literal ``[..]``.

        :return: the parsed term
        :rtype: Term
        """
        tk, tv = self.next()
        if tk == "INT":
            assert isinstance(tv, int)
            if self.accept("PUNCT", ".."):
                return Range(tv, self.expect_int())
            return tv
        if tk == "STRING":
            return Id(str(tv))  # strings only appear in annotations we ignore; keep as a opaque id
        if tk == "IDENT":
            if tv == "true":
                return True
            if tv == "false":
                return False
            if self.accept("PUNCT", "["):  # array element access name[index]
                index = self.expect_int()
                self.expect("PUNCT", "]")
                return ArrayAccess(str(tv), index)
            return Id(str(tv))
        if tk == "PUNCT" and tv == "[":
            elems = self.parse_arg_list()
            self.expect("PUNCT", "]")
            return elems
        if tk == "PUNCT" and tv == "{":
            values: List[int] = []
            if not self.accept("PUNCT", "}"):
                values.append(self.expect_int())
                while self.accept("PUNCT", ","):
                    values.append(self.expect_int())
                self.expect("PUNCT", "}")
            return SetLit(sorted(set(values)))
        raise FznParseError(f"unexpected term token {tk} {tv!r}")

    def parse_constraint(self) -> Constraint:
        """
        Parses a constraint statement.

        :return: the parsed constraint
        :rtype: Constraint
        """
        self.expect("IDENT", "constraint")
        name = str(self.expect("IDENT"))
        self.expect("PUNCT", "(")
        args = self.parse_arg_list()
        self.expect("PUNCT", ")")
        annotations = self.parse_annotations()
        self.expect("PUNCT", ";")
        return Constraint(name, args, annotations)

    def parse_solve(self) -> Solve:
        """
        Parses a solve statement.

        :return: the parsed solve item
        :rtype: Solve
        """
        self.expect("IDENT", "solve")
        self.parse_annotations()  # search annotations are ignored in v1
        kind = str(self.expect("IDENT"))
        if kind == "satisfy":
            self.expect("PUNCT", ";")
            return Solve("satisfy")
        if kind in ("minimize", "maximize"):
            objective = self.parse_term()
            self.expect("PUNCT", ";")
            return Solve(kind, objective)
        raise FznParseError(f"unexpected solve kind {kind!r}")

    def parse_domain(self) -> Tuple[int, int, Optional[List[int]]]:
        """
        Parses a scalar integer/bool domain after ``var``: ``int``, ``lo..hi``, ``{v,..}`` or ``bool``.

        NuCS domains are intervals, so a non-contiguous ``{v,..}`` domain is returned as its (min, max)
        interval together with the explicit list of allowed values; the model layer posts a member
        constraint to enforce the holes. A contiguous domain returns ``None`` as its value list.

        :return: the (lo, hi) bounds and, for a non-contiguous set, the list of allowed values
        :rtype: Tuple[int, int, Optional[List[int]]]
        """
        tk, tv = self.peek()  # type: ignore[misc]
        if tk == "IDENT" and tv == "bool":
            self.next()
            return 0, 1, None
        if tk == "IDENT" and tv == "int":
            self.next()
            return DEFAULT_INT_MIN, DEFAULT_INT_MAX, None
        if tk == "IDENT" and tv in ("float", "set"):
            raise FznUnsupportedError(f"domain '{tv}' is not supported")
        if tk == "PUNCT" and tv == "{":
            self.next()
            values = [self.expect_int()]
            while self.accept("PUNCT", ","):
                values.append(self.expect_int())
            self.expect("PUNCT", "}")
            values = sorted(set(values))
            lo, hi = values[0], values[-1]
            # A contiguous set is exactly its interval, so no member constraint is needed.
            if values == list(range(lo, hi + 1)):
                return lo, hi, None
            return lo, hi, values
        if tk == "INT":
            lo = self.expect_int()
            self.expect("PUNCT", "..")
            hi = self.expect_int()
            return lo, hi, None
        raise FznParseError(f"unexpected domain token {tk} {tv!r}")

    def parse_var_decl(self) -> VarDecl:
        """
        Parses a scalar variable declaration.

        :return: the parsed variable declaration
        :rtype: VarDecl
        """
        self.expect("IDENT", "var")
        is_bool = self.peek() == ("IDENT", "bool")
        lo, hi, values = self.parse_domain()
        self.expect("PUNCT", ":")
        name = str(self.expect("IDENT"))
        annotations = self.parse_annotations()
        rhs: Optional[Term] = None
        if self.accept("PUNCT", "="):
            rhs = self.parse_term()
        self.expect("PUNCT", ";")
        return VarDecl(name, lo, hi, annotations, rhs, is_bool, values)

    def parse_par_decl(self) -> ParDecl:
        """
        Parses a scalar parameter declaration ``<type>: name = value``.

        :return: the parsed parameter declaration
        :rtype: ParDecl
        """
        tv = str(self.expect("IDENT"))
        if tv in ("float", "set"):
            raise FznUnsupportedError(f"parameter type '{tv}' is not supported")
        if tv not in ("int", "bool"):
            raise FznParseError(f"unexpected parameter type {tv!r}")
        self.expect("PUNCT", ":")
        name = str(self.expect("IDENT"))
        self.expect("PUNCT", "=")
        term = self.parse_term()
        value = _term_to_par_value(term)
        self.expect("PUNCT", ";")
        return ParDecl(name, value)

    def parse_array_decl(self) -> ArrayDecl:
        """
        Parses an array declaration, of parameters or of variables.

        :return: the parsed array declaration
        :rtype: ArrayDecl
        """
        self.expect("IDENT", "array")
        self.expect("PUNCT", "[")
        index_set = self.parse_term()  # the index set, e.g. 1..N; its size is needed for arrays without a literal
        self.expect("PUNCT", "]")
        size = index_set.hi - index_set.lo + 1 if isinstance(index_set, Range) else None
        self.expect("IDENT", "of")
        is_var = self.accept("IDENT", "var")
        lo: Optional[int] = None
        hi: Optional[int] = None
        values: Optional[List[int]] = None
        is_bool = False
        if is_var:
            is_bool = self.peek() == ("IDENT", "bool")
            lo, hi, values = self.parse_domain()
        else:
            elem_type = str(self.expect("IDENT"))
            if elem_type not in ("int", "bool"):
                raise FznUnsupportedError(f"array element type '{elem_type}' is not supported")
            is_bool = elem_type == "bool"
        self.expect("PUNCT", ":")
        name = str(self.expect("IDENT"))
        annotations = self.parse_annotations()
        elems: List[Term] = []
        if self.accept("PUNCT", "="):
            term = self.parse_term()
            if not isinstance(term, list):
                raise FznParseError("expected an array literal on the right-hand side of an array declaration")
            elems = term
        self.expect("PUNCT", ";")
        return ArrayDecl(name, elems, annotations, is_var, lo, hi, size, is_bool, values)


def _term_to_par_value(term: Term) -> Union[int, bool, List[int]]:
    """
    Converts a parsed term into a parameter value (int, bool or list of ints).

    :param term: the parsed term
    :type term: Term

    :return: the parameter value
    :rtype: Union[int, bool, List[int]]
    """
    if isinstance(term, bool):
        return term
    if isinstance(term, int):
        return term
    if isinstance(term, list):
        return [int(e) if not isinstance(e, bool) else int(e) for e in term]
    raise FznParseError(f"unsupported parameter value {term!r}")


def parse(text: str) -> List[Statement]:
    """
    Tokenizes and parses FlatZinc text into a list of statements.

    :param text: the FlatZinc source
    :type text: str

    :return: the parsed statements
    :rtype: List[Statement]
    """
    return Parser(tokenize(text)).parse()
