# Write docstring

## Instructions

- Triple double-quotes on their own lines.
- One-line summary directly after the opening `"""`.
- `:param` / `:type` pair per parameter, in declaration order; `:return` / `:rtype` for the return.
- Descriptions are lowercase, no trailing period.

## Examples

```python
"""
Returns the time complexity of the propagator as an int.

:param n: the number of variables
:type n: int
:param parameters: the parameters, unused here
:type parameters: NDArray

:return: the time complexity of the propagator
:rtype: int
"""
```