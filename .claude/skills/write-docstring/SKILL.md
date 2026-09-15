---
name: write-docstring
description: The reST docstring format used across NuCS — summary sentence, :param/:type pairs, :return/:rtype. Use when writing or editing the docstring of a function, method or class under nucs/, tests/ or scripts/.
---

# Write docstring

- Put the triple double-quotes on their own lines.
- Start with a one-line summary sentence directly after the opening `"""`.
- Give one `:param` / `:type` pair per parameter, in declaration order, then `:return` / `:rtype` for the return.
- Write the `:param` and `:return` descriptions in lowercase, with no trailing period.

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
