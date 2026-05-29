# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Shared utilities for experiment module.

This module provides:
- Object loading and introspection utilities
- Pareto front and hypervolume calculation utilities
- Name sanitization helpers
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from morl_tsp import config

# -----------------------------------------------------------------------------
# Object loading and introspection
# -----------------------------------------------------------------------------


def load_object(import_path: str) -> Any:
    """Load an object from `module.submodule:object_name` import path."""
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path '{import_path}'. Expected format 'module.submodule:object_name'."
        )
    module_path, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, object_name)


def supports_kwargs(callable_obj: Any) -> bool:
    """Check if a callable accepts **kwargs."""
    sig = inspect.signature(callable_obj)
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only those accepted by the callable."""
    if supports_kwargs(callable_obj):
        return kwargs
    sig = inspect.signature(callable_obj)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def safe_name(name: str, fallback: str = "metric") -> str:
    """Sanitize a string for use as a metric/file name."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))
    return cleaned.strip("_") or fallback


# -----------------------------------------------------------------------------
# Plain-data and config utilities
# -----------------------------------------------------------------------------


def deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, with overrides taking precedence."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_hpt_base_config(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve the base config for an HP tuning YAML with optional override selection."""
    assert isinstance(data.get("base_config"), dict), "base_config should be a dictionary."
    base_config = data["base_config"]

    config_overwrite = data.get("config_overwrite")
    if not isinstance(config_overwrite, list) or len(config_overwrite) == 0:
        return dict(base_config)

    hp_object = data.get("hyperparameter_object", {})
    config_override_idx = int(hp_object.get("config_override_idx", 0))
    assert 0 <= config_override_idx < len(config_overwrite), (
        f"config_override_idx={config_override_idx} out of range for "
        f"{len(config_overwrite)} config_overwrite."
    )

    selected_override = config_overwrite[config_override_idx]
    assert isinstance(selected_override, dict), "Selected config override must be a dictionary."
    selected_override = {k: v for k, v in selected_override.items() if k != "name"}
    return deep_merge_dicts(base_config, selected_override)


def to_plain_data(value: Any) -> Any:
    """Convert nested values into JSON/YAML-serializable plain python data."""
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# -----------------------------------------------------------------------------
# Pareto front and hypervolume calculations
# -----------------------------------------------------------------------------


def to_reward_vector(step_reward: Any) -> np.ndarray | None:
    """Convert a step reward to a numpy array, handling various input types.

    Args:
        step_reward: A scalar, list, tuple, or array-like reward value.

    Returns:
        A 1D numpy array of floats, or None if conversion fails or values are invalid.
    """
    if step_reward is None:
        return None
    reward_array = np.asarray(step_reward, dtype=np.float64).reshape(-1)
    if reward_array.size == 0:
        return None
    if not np.all(np.isfinite(reward_array)):
        return None
    return reward_array


def pareto_nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Compute a boolean mask indicating which points are Pareto non-dominated.

    Uses maximization semantics: point A dominates point B if A >= B in all
    dimensions and A > B in at least one dimension.

    Args:
        points: A 2D numpy array of shape (n_points, n_dimensions).

    Returns:
        A boolean array of shape (n_points,) where True indicates non-dominated.
    """
    n_points = points.shape[0]
    keep = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not keep[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                keep[i] = False
                break
    return keep


def hypervolume_2d_maximize(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the 2D hypervolume for maximization problems.

    Args:
        points: A 2D numpy array of shape (n_points, 2).
        ref_point: A 1D numpy array of shape (2,) representing the reference point.

    Returns:
        The hypervolume as a non-negative float.

    Raises:
        ValueError: If points or ref_point have incorrect shapes.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected 2D points with shape (n,2), got {points.shape}.")
    if ref_point.shape != (2,):
        raise ValueError(f"Expected ref_point shape (2,), got {ref_point.shape}.")

    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]
    if points.shape[0] == 0:
        return 0.0

    improving_mask = np.logical_and(points[:, 0] > ref_point[0], points[:, 1] > ref_point[1])
    points = points[improving_mask]
    if points.shape[0] == 0:
        return 0.0

    # Largest x first.
    order = np.argsort(points[:, 0])[::-1]
    points = points[order]

    hv = 0.0
    current_max_y = float(ref_point[1])
    for x, y in points:
        if y <= current_max_y:
            continue
        hv += float(x - ref_point[0]) * float(y - current_max_y)
        current_max_y = float(y)

    return max(0.0, float(hv))


def nondominated_points_minimize(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Find non-dominated points for minimization problems (2D).

    Args:
        points: A list of (x, y) tuples representing objective values to minimize.

    Returns:
        A list of non-dominated (x, y) tuples.
    """
    nondom: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(points):
        dominated = False
        for j, (x2, y2) in enumerate(points):
            if i == j:
                continue
            if (x2 <= x and y2 <= y) and (x2 < x or y2 < y):
                dominated = True
                break
        if not dominated:
            nondom.append((x, y))
    return nondom


def pareto_hv_2d_minimize(points: list[tuple[float, float]]) -> float:
    """Compute the 2D hypervolume for minimization problems with auto reference point.

    The reference point is automatically set to 1.05x the maximum values
    (or max + 1.0 if values are non-positive).

    Args:
        points: A list of (x, y) tuples representing objective values to minimize.

    Returns:
        The hypervolume as a float, or NaN if points is empty.
    """
    if len(points) == 0:
        return float("nan")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ref_x = max(xs) * 1.05
    ref_y = max(ys) * 1.05
    if ref_x <= 0:
        ref_x = max(xs) + 1.0
    if ref_y <= 0:
        ref_y = max(ys) + 1.0

    nondom = nondominated_points_minimize(points)
    clipped = [(x, y) for x, y in nondom if x < ref_x and y < ref_y]
    if len(clipped) == 0:
        return 0.0

    hv = 0.0
    best_y = ref_y
    for x, y in sorted(clipped, key=lambda p: p[0]):
        if y >= best_y:
            continue
        width = ref_x - x
        height = best_y - y
        if width > 0 and height > 0:
            hv += width * height
        best_y = y
    return float(hv)


def extract_env_history_from_candidate(candidate: Any) -> list[Any] | None:
    """Extract environment history from an environment or wrapper.

    Traverses the environment wrapper chain to find env_history attribute,
    handling both single environments and vectorized environments.

    Args:
        candidate: An environment instance or wrapper.

    Returns:
        The env_history list if found, otherwise None.
    """
    to_visit = [candidate]
    visited: set[int] = set()
    while len(to_visit) > 0:
        current = to_visit.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        get_attr_fn = getattr(current, "get_attr", None)
        if callable(get_attr_fn):
            try:
                env_histories = get_attr_fn("env_history")
                if isinstance(env_histories, list) and len(env_histories) > 0:
                    merged_histories: list[Any] = []
                    for history_chunk in env_histories:
                        if isinstance(history_chunk, list):
                            merged_histories.extend(history_chunk)
                    if len(merged_histories) > 0:
                        return merged_histories
            except Exception:
                pass

        env_history = getattr(current, "env_history", None)
        if isinstance(env_history, list):
            return env_history

        wrapped_env = getattr(current, "env", None)
        if wrapped_env is not None:
            to_visit.append(wrapped_env)

        unwrapped_env = getattr(current, "unwrapped", None)
        if unwrapped_env is not None:
            to_visit.append(unwrapped_env)

        sub_envs = getattr(current, "envs", None)
        if isinstance(sub_envs, (list, tuple)):
            to_visit.extend(sub_envs)

    return None


def compute_pareto_score(
    episode_vectors: np.ndarray,
    ref_point: np.ndarray,
    metric_name: str = "pareto_hv",
    nondominated_bonus: float = 0.0,
) -> dict[str, Any]:
    """Compute Pareto hypervolume score with optional non-dominated bonus.

    Args:
        episode_vectors: A 2D array of shape (n_episodes, 2) with reward vectors.
        ref_point: Reference point for hypervolume calculation.
        metric_name: One of "pareto_hv", "pareto_hypervolume", or "pareto_hv_nondom".
        nondominated_bonus: Coefficient for bonus based on fraction of non-dominated points.

    Returns:
        A dictionary containing pareto_hv, pareto_nondominated_points, pareto_score,
        and reference point values.
    """
    if episode_vectors.shape[1] != 2:
        fallback_score = float(np.mean(np.sum(episode_vectors, axis=1)))
        return {
            "pareto_metric_name": metric_name,
            "pareto_metric_fallback": "mean_scalar_reward",
            "pareto_reward_dim": int(episode_vectors.shape[1]),
            "pareto_episode_count": int(episode_vectors.shape[0]),
            "pareto_nondominated_points": 0,
            "pareto_hv": 0.0,
            "pareto_score": fallback_score,
        }

    nondominated_mask = pareto_nondominated_mask(episode_vectors)
    nondominated_points = episode_vectors[nondominated_mask]
    hv = hypervolume_2d_maximize(points=nondominated_points, ref_point=ref_point)

    score = float(hv)
    if metric_name == "pareto_hv_nondom" and nondominated_bonus > 0:
        score += nondominated_bonus * (
            float(nondominated_points.shape[0]) / float(max(1, episode_vectors.shape[0]))
        )

    return {
        "pareto_metric_name": metric_name,
        "pareto_reward_dim": int(episode_vectors.shape[1]),
        "pareto_episode_count": int(episode_vectors.shape[0]),
        "pareto_nondominated_points": int(nondominated_points.shape[0]),
        "pareto_hv": float(hv),
        "pareto_ref_point_0": float(ref_point[0]),
        "pareto_ref_point_1": float(ref_point[1]),
        "pareto_score": float(score),
    }


def resolve_pareto_ref_point(
    reward_dim: int,
    episode_vectors: np.ndarray,
    configured_ref: Any = None,
    morl_ref: Any = None,
) -> np.ndarray:
    """Resolve the reference point for Pareto hypervolume calculation.

    Priority:
    1. Explicit configured reference point (hpt_pareto_ref_point or similar)
    2. MORL reference point from config
    3. Auto-computed from episode vectors (min - margin)

    Args:
        reward_dim: Number of reward dimensions.
        episode_vectors: Episode reward vectors for auto-computation fallback.
        configured_ref: Explicit reference point from config.
        morl_ref: MORL reference point from config.

    Returns:
        A numpy array of shape (reward_dim,) representing the reference point.
    """
    if (
        isinstance(configured_ref, (list, tuple))
        and len(configured_ref) >= reward_dim
        and all(isinstance(v, (int, float)) for v in configured_ref[:reward_dim])
    ):
        return np.asarray(configured_ref[:reward_dim], dtype=np.float64)

    if (
        isinstance(morl_ref, (list, tuple))
        and len(morl_ref) >= reward_dim
        and all(isinstance(v, (int, float)) for v in morl_ref[:reward_dim])
    ):
        return np.asarray(morl_ref[:reward_dim], dtype=np.float64)

    # Auto-compute from episode vectors
    min_values = np.min(episode_vectors, axis=0)
    margin = np.maximum(0.05 * np.maximum(np.abs(min_values), 1.0), 1e-3)
    return (min_values - margin).astype(np.float64)


# -----------------------------------------------------------------------------
# Environment utilities
# -----------------------------------------------------------------------------


def ensure_env_spec_id(env: Any, default_id: str = "morl_tsp_env") -> None:
    """Ensure an environment has a valid spec.id for gymnasium compatibility.

    Args:
        env: The environment to check/fix.
        default_id: The default ID to use if spec is missing or invalid.
    """
    from gymnasium.envs.registration import EnvSpec

    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "id", None) is not None:
        return

    target = getattr(env, "env", None)
    if target is not None and target is not env:
        # Some wrappers expose spec as a read-only property. Set the wrapped env
        # instead so vectorized/SB3 compatibility code can still discover spec.id.
        ensure_env_spec_id(target, default_id)
        spec = getattr(env, "spec", None)
        if spec is not None and getattr(spec, "id", None) is not None:
            return

    if spec is not None:
        try:
            spec.id = default_id
            return
        except AttributeError:
            pass

    try:
        env.spec = EnvSpec(id=default_id)
    except AttributeError:
        return

# -----------------------------------------------------------------------------
# HPT Connection String
# -----------------------------------------------------------------------------

def _safe_study_file_name(study_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name).strip("._")
    if not safe_name:
        raise ValueError("Database study name must contain at least one safe filename character.")
    return safe_name

def resolve_optuna_connection_string(
    db_study_name: str,
    db_connection_string: str | None = None,
) -> str:
    if db_connection_string is not None and db_connection_string.strip():
        return db_connection_string
    if db_study_name is None or not str(db_study_name).strip():
        raise ValueError("Database study name must be provided for Optuna.")

    optuna_dir = Path(os.getenv("OPTUNA_DB_DIR", f"{config.ROOT_PATH}/artifacts/optuna"))
    optuna_dir.mkdir(parents=True, exist_ok=True)
    db_path = optuna_dir / f"{_safe_study_file_name(str(db_study_name).strip())}.db"
    return f"sqlite:///{db_path}"