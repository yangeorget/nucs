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
Keeps ``nucs/fzn/README.md`` honest: its two generated listings must match what the code actually does.

The builtin list drifted badly before (14 names documented against 97 registered), which is exactly the
kind of rot a test catches for free.
"""

import os
import re

from nucs.fzn.builtins import BUILTINS

README = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "nucs", "fzn", "README.md"
)
GLOBALS_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "nucs",
    "fzn",
    "share",
    "minizinc",
    "nucs",
)


def _fenced_block_after(heading: str) -> set[str]:
    """Returns the comma-separated names of the first fenced block following a heading in the README."""
    with open(README) as f:
        text = f.read()
    tail = text[text.index(heading) :]
    block = re.search(r"```\n(.*?)```", tail, re.DOTALL)
    assert block is not None, f"no fenced listing after {heading!r}"
    return {name.strip() for name in block.group(1).replace("\n", " ").split(",") if name.strip()}


def test_readme_lists_every_supported_builtin() -> None:
    """The README's builtin listing is exactly the BUILTINS registry."""
    documented = _fenced_block_after("## Supported builtins")
    assert documented == set(BUILTINS), (
        f"missing from the README: {sorted(set(BUILTINS) - documented)}; "
        f"stale in the README: {sorted(documented - set(BUILTINS))}"
    )


def test_readme_states_the_builtin_count() -> None:
    """The count quoted in the prose matches the registry."""
    with open(README) as f:
        text = f.read()
    match = re.search(r"dispatches these (\d+) FlatZinc builtins", text)
    assert match is not None
    assert int(match.group(1)) == len(BUILTINS)


def test_readme_lists_every_kept_global() -> None:
    """The README's globals listing is exactly the fzn_*.mzn files of the globals library."""
    documented = _fenced_block_after("The globals library in")
    on_disk = {f[len("fzn_") : -len(".mzn")] for f in os.listdir(GLOBALS_LIB) if f.startswith("fzn_")}
    assert documented == on_disk, (
        f"missing from the README: {sorted(on_disk - documented)}; stale in the README: {sorted(documented - on_disk)}"
    )
