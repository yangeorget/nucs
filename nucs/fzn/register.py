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
Registers NuCS as a MiniZinc solver by writing a fully-resolved ``nucs.msc`` into MiniZinc's user solvers
directory. The version is read from the installed package metadata and the ``executable`` and ``mznlib``
paths are made absolute, so ``minizinc --solver nucs`` works without any environment variable.
"""

import json
import os
import re
import shutil
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from nucs.fzn.errors import FznError


def package_version() -> str:
    """
    Returns the NuCS package version.

    When running from a source checkout (an editable install), the version is read directly from
    ``pyproject.toml`` so it always reflects the current value even if the installed metadata is stale.
    Otherwise it falls back to the installed package metadata, then to ``0.0.0``.

    :return: the package version
    :rtype: str
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        try:
            project = tomllib.loads(pyproject.read_text()).get("project", {})
            if str(project.get("name", "")).lower() == "nucs":
                return str(project["version"])
        except (tomllib.TOMLDecodeError, KeyError, OSError):
            pass
    try:
        return version("NuCS")
    except PackageNotFoundError:
        return "0.0.0"


def share_dir() -> Path:
    """
    Returns the absolute path to the packaged ``share`` directory (containing ``nucs.msc`` and the globals library).

    :return: the share directory
    :rtype: Path
    """
    import nucs.fzn

    return Path(nucs.fzn.__file__).resolve().parent / "share"


def executable_path() -> str:
    """
    Returns the absolute path to the installed ``fzn-nucs`` executable.

    :return: the absolute executable path
    :rtype: str
    """
    sibling = Path(sys.executable).resolve().parent / "fzn-nucs"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("fzn-nucs")
    if found:
        return str(Path(found).resolve())
    raise FznError("could not locate the 'fzn-nucs' executable; is NuCS installed in this environment?")


def user_solvers_dir() -> Path:
    """
    Returns MiniZinc's per-user solver configuration directory for the current platform.

    :return: the user solvers directory
    :rtype: Path
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "MiniZinc" / "solvers"
    return Path.home() / ".minizinc" / "solvers"


def sync_template_msc() -> Path:
    """
    Updates the version field of the packaged ``nucs.msc`` template to the current package version, so the
    shipped file does not drift from ``pyproject.toml``. Run from the build script.

    :return: the path of the updated template
    :rtype: Path
    """
    path = share_dir() / "nucs.msc"
    text = re.sub(r'("version":\s*")[^"]*(")', rf"\g<1>{package_version()}\g<2>", path.read_text(), count=1)
    path.write_text(text)
    return path


def register(target_dir: Path | None = None) -> Path:
    """
    Writes a resolved ``nucs.msc`` (absolute paths, version from package metadata) into MiniZinc's user
    solvers directory.

    :param target_dir: the directory to write into, defaults to the per-user solvers directory
    :type target_dir: Path | None

    :return: the path of the written solver configuration
    :rtype: Path
    """
    config = json.loads((share_dir() / "nucs.msc").read_text())
    config["version"] = package_version()
    config["mznlib"] = str((share_dir() / "minizinc" / "nucs").resolve())
    config["executable"] = executable_path()
    target_dir = target_dir or user_solvers_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "nucs.msc"
    target.write_text(json.dumps(config, indent=2) + "\n")
    return target
