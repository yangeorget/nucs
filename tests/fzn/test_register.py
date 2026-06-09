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
import json
import os

from nucs.fzn.register import package_version, register, share_dir


class TestRegister:
    def test_template_msc_version_matches_package(self) -> None:
        # the shipped nucs.msc template must not drift from pyproject; run scripts/bash/build.sh to refresh it
        config = json.loads((share_dir() / "nucs.msc").read_text())
        assert config["version"] == package_version()

    def test_register_writes_resolved_msc(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        target = register(target_dir=tmp_path)
        assert target == tmp_path / "nucs.msc"
        config = json.loads(target.read_text())
        # the version comes from the installed package metadata, not a hardcoded literal
        assert config["version"] == package_version()
        # executable and mznlib are absolute and exist
        assert os.path.isabs(config["executable"])
        assert os.path.isabs(config["mznlib"])
        assert os.path.isdir(config["mznlib"])
        assert config["id"] == "org.nucs.nucs"
