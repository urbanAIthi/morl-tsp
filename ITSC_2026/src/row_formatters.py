# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any

from morl_tsp.experiment.reproducibility import backend_provenance, sha256_file


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _scenario_file_details(sim_config: dict[str, Any]) -> dict[str, str]:
    scenarios = sim_config.get("scenarios", [])
    if not isinstance(scenarios, list) or len(scenarios) == 0:
        return {
            "net_file": "",
            "net_sha256": "",
            "route_file": "",
            "route_sha256": "",
            "route_generation_hash": "",
        }
    first = scenarios[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        return {
            "net_file": "",
            "net_sha256": "",
            "route_file": "",
            "route_sha256": "",
            "route_generation_hash": "",
        }

    net_file = Path(str(first[0]))
    route_file = Path(str(first[1]))
    route_hash = ""
    try:
        parts = route_file.parts
        cache_idx = parts.index("_iz_scenario_cache")
        if cache_idx + 1 < len(parts):
            route_hash = parts[cache_idx + 1]
    except Exception:
        route_hash = ""

    return {
        "net_file": str(net_file),
        "net_sha256": sha256_file(net_file) if net_file.exists() and net_file.is_file() else "",
        "route_file": str(route_file),
        "route_sha256": (
            sha256_file(route_file) if route_file.exists() and route_file.is_file() else ""
        ),
        "route_generation_hash": route_hash,
    }


def paper_eval_rows(
    rows: list[dict[str, Any]], experiment: Any | None = None
) -> list[dict[str, Any]]:
    """Shape generic eval rows into the columns used by the paper plots and tables."""
    out: list[dict[str, Any]] = []
    sim_config = getattr(experiment, "sim_config", {}) if experiment is not None else {}
    scenario_details = _scenario_file_details(sim_config)
    model_path = str(sim_config.get("model_path", ""))
    model_file = Path(model_path) if model_path else None
    provenance_file = Path(str(sim_config.get("log_dir", ""))) / "experiment_provenance.json"
    backend = backend_provenance()
    for row in rows:
        weights = str(row.get("weights", "")).strip("[] ")
        weight_parts = [part for part in weights.split(",") if part.strip()]
        weight_bus = _to_float(weight_parts[0]) if weight_parts else None
        model_type = str(row.get("model_type", row.get("policy_source", "")))
        name = str(row.get("policy_name", row.get("name", model_type)))
        route_seed = row.get("route_seed", row.get("_route_seed", ""))
        timing_seed = row.get("timing_seed", row.get("_timing_seed", route_seed))

        shaped = dict(row)
        shaped.update(
            {
                "model_type": model_type,
                "name": name,
                "seed": row.get("seed", route_seed),
                "route_seed": route_seed,
                "timing_seed": timing_seed,
                "weight_bus": "" if weight_bus is None else weight_bus,
                "j_bus": _first_float(row, "Bus_crossing_time", "Bus_crossing_time_mean"),
                "j_all": _first_float(row, "Car_crossing_time", "Car_crossing_time_mean"),
                "j_bus_median": _first_float(row, "Bus_crossing_time_median"),
                "j_all_median": _first_float(row, "Car_crossing_time_median"),
                "tail_cvar_10": _first_float(row, "cvar_crossing_times_10%"),
                "tail_cvar_bus_10": _first_float(row, "cvar_CT_bus_10%"),
                "phase_change_rate_per_min": _first_float(row, "phase_change_rate_per_min"),
                "mean_green_duration_seconds": _first_float(row, "mean_green_duration_seconds"),
                "infeasible_action_rate": _first_float(row, "infeasible_action_rate"),
                "episodes": row.get("n_eval_episodes", row.get("episodes", "")),
                "network": ",".join(str(p) for p in sim_config.get("iz_net_rel_paths", [])),
                "traffic_mode": "seeded_generated",
                "run_log_dir": sim_config.get("log_dir", ""),
                "run_provenance_file": str(provenance_file),
                "run_provenance_sha256": (
                    sha256_file(provenance_file)
                    if provenance_file.exists() and provenance_file.is_file()
                    else ""
                ),
                "runtime_backend": backend.get("actual_backend", ""),
                "model_sha256": (
                    sha256_file(model_file)
                    if model_file is not None and model_file.exists() and model_file.is_file()
                    else ""
                ),
                **scenario_details,
            }
        )
        out.append(shaped)
    return out
