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


class FznError(Exception):
    """
    Base class for all FlatZinc adapter errors.
    """


class FznParseError(FznError):
    """
    Raised when the FlatZinc input cannot be parsed.
    """


class FznUnsupportedError(FznError):
    """
    Raised when a FlatZinc feature or builtin is not supported by NuCS.
    """
