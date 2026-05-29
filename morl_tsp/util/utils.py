# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

import math
from collections.abc import Callable
from functools import wraps
from heapq import nlargest
from typing import TypeVar

try:
    # Python 3.10+
    from typing import Concatenate, ParamSpec
except Exception:  # pragma: no cover - fallback for older Pythons
    from typing import Concatenate

    from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def cleanup_decorator(func: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:  # noqa: UP047
    @wraps(func)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(self, *args, **kwargs)
        except Exception:
            # attempt best-effort cleanup, re-raise original exception
            try:
                self._cleanup()
            except Exception:
                pass
            raise

    return wrapper

def nseconds_from_str(day_time:str) -> int:
        h, m, s = map(int, day_time.split(':'))
        return h * 3600 + m * 60 + s

def discretize_density(density: float) -> int:
        return min(int(density * 10), 9)

def check_common(list1: list[object], list2: list[object]) -> bool:
    return not set(list1).isdisjoint(set(list2))

def min_max_normalization(value:float,
                            min_value:float,
                            max_value:float)->float:
    return (value-min_value)/(max_value-min_value)

def clipped_min_max_normalization(value:float,min_value:float,max_value:float)->float:
    return (max(min(value,max_value),min_value) - min_value) / (max_value - min_value)

def cvar_tail_mean(values: list[float], alpha: float = 0.10) -> float:
    """
    CVaR / tail mean of the worst alpha fraction of values.
    For delays/crossing-times, 'worst' typically means largest values.
    """
    assert 0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"

    n = len(values)
    if n == 0:
        return 0.0

    k = max(1, int(math.ceil(alpha * n)))

    # Worst tail = k largest values. nlargest is O(n log k) and avoids full sort.
    tail = nlargest(k, values)
    return sum(tail) / k

def cvar_tail_mean_per_group(
    groups: dict[str, list[float]],
    alpha: float = 0.10,
) -> list[float]:
    """CVaR tail mean for each group; empty group -> 0.0."""
    if not groups:
        return []
    return [cvar_tail_mean(vals, alpha) if vals else 0.0 for vals in groups.values()]
