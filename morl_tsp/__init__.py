# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Public package surface for morl_tsp."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from morl_tsp.environment.env import ObservationFunction, ObservationFunctionPT, SumoEnvironment
    from morl_tsp.environment.traffic_signal import TrafficSignal

__all__ = [
    "ObservationFunction",
    "ObservationFunctionPT",
    "SumoEnvironment",
    "TrafficSignal",
    "config",
]


def __getattr__(name: str) -> Any:
    if name == "config":
        return import_module("morl_tsp.config")
    if name in {"ObservationFunction", "ObservationFunctionPT", "SumoEnvironment"}:
        module = import_module("morl_tsp.environment.env")
        return getattr(module, name)
    if name == "TrafficSignal":
        module = import_module("morl_tsp.environment.traffic_signal")
        return getattr(module, name)
    raise AttributeError(f"module 'morl_tsp' has no attribute {name!r}")
