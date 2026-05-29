# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""
CLI runner for MORL policy evaluation experiments from YAML.

Supports:
1. Single-run evaluation (default)
2. Multi-seed route/timing evaluation
"""

import argparse
import copy
import csv
import inspect
import json
import logging
import os
import sys
import traceback
import warnings
from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast

import yaml

from morl_tsp import config
from morl_tsp.experiment.morl_eval_experiment import morl_eval_experiment
from morl_tsp.experiment.utils import load_object

if config.ROOT_PATH not in sys.path:
    sys.path.insert(0, config.ROOT_PATH)

logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*env\\.trajectory.*")

type YamlDict = dict[str, Any]
type SeedPair = tuple[int, int]
type BaselineName = Literal["fixed", "fixed_time", "fixed_ts", "tsp", "rule_tsp"]

CITY_PREFIXES: dict[str, str] = {
    "atlanta": "atl",
    "boston": "bos",
    "chicago": "chi",
    "dallas": "dal",
    "los-angeles": "la",
    "new-york-city": "nyc",
    "salt-lake-city": "slc",
    "san-francisco": "sf",
    "seattle": "sea",
}


class EvalConfig(TypedDict, total=False):
    rows_root: str
    rows_mode: str
    rows_append: bool
    rows_formatter: str
    rows_key_columns: list[str]
    route_seeds: list[int]
    timing_seeds: list[int]
    morl_eval_route_seeds: list[int]
    morl_eval_timing_seeds: list[int]
    reward_weight_keys: list[str]
    baselines: list[Any]


class EvalYaml(TypedDict, total=False):
    group: str
    stable_baselines_model: str
    callbacks: list[str]
    experiment_type: str
    workers: int
    n_parallel: int
    eval: EvalConfig
    config_overwrite: list[YamlDict]
    hyperparameters: YamlDict
    base_config: YamlDict


def _deep_merge_dicts(base: Mapping[str, Any], overwrite: Mapping[str, Any]) -> YamlDict:
    merged: YamlDict = copy.deepcopy(dict(base))
    for key, value in overwrite.items():
        old_value = merged.get(key)
        if isinstance(old_value, dict) and isinstance(value, dict):
            merged[str(key)] = _deep_merge_dicts(old_value, value)
        else:
            merged[str(key)] = copy.deepcopy(value)
    return merged


def _as_mapping(value: Any, *, name: str) -> YamlDict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return cast(YamlDict, value)


def _optional_mapping(value: Any, *, name: str) -> YamlDict:
    if value is None:
        return {}
    return _as_mapping(value, name=name)


def _as_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return value


def _load_eval_yaml(path: str) -> EvalYaml:
    with open(path) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return cast(EvalYaml, data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one policy bus-flow eval experiment from a yaml file."
    )
    parser.add_argument("--yaml_path", type=str, help="Path to the yaml experiment file")
    parser.add_argument("--experiment_idx", type=int, help="Index from config_overwrite to run")
    parser.add_argument(
        "--nth_run", type=int, default=0, help="Unused parity arg for compatibility"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional output path for JSON result payload.",
    )
    return parser


def _filter_init_data(data: Mapping[str, Any]) -> YamlDict:
    """Filter data to only kwargs accepted by morl_eval_experiment.__init__."""
    init_params = [
        param_name
        for param_name in inspect.signature(morl_eval_experiment.__init__).parameters
        if param_name != "self"
    ]
    return {k: v for k, v in data.items() if k in init_params}


def _is_rl_eval_config(
    config_override: Mapping[str, Any], base_config: Mapping[str, Any] | None = None
) -> bool:
    """Detect if config is for RL (SB3) or MORL evaluation."""
    merged = dict(base_config or {})
    merged.update(config_override)

    rl_models = merged.get("rl_eval_models")
    if isinstance(rl_models, list) and len(rl_models) > 0:
        return True

    model_path = merged.get("model_path")
    return isinstance(model_path, str) and model_path.lower().endswith(".zip")


def _eval_tag(
    config_override: Mapping[str, Any], base_config: Mapping[str, Any] | None = None
) -> str:
    """Generate W&B tag based on eval type."""
    return "rl_eval" if _is_rl_eval_config(config_override, base_config) else "morl_eval"


def _city_tag(config_override: Mapping[str, Any]) -> str:
    iz_net_rel_paths = config_override.get("iz_net_rel_paths")
    if isinstance(iz_net_rel_paths, list) and len(iz_net_rel_paths) > 0:
        first_path = iz_net_rel_paths[0]
        if isinstance(first_path, str):
            parts = first_path.split("/")
            if len(parts) >= 2:
                prefix = CITY_PREFIXES.get(parts[0], parts[0])
                return f"{prefix}{parts[1]}"

    return str(config_override.get("name", "")).split("_", 1)[0]


def _rows_csv_path(eval_config: EvalConfig, config_override: Mapping[str, Any]) -> str | None:
    rows_root = eval_config.get("rows_root")
    if not isinstance(rows_root, str) or not rows_root:
        return None

    city = _city_tag(config_override)
    if not city:
        return None
    return os.path.join(rows_root, city, "rows.csv")


def _first_eval_flow_name(data: EvalYaml, config_override: Mapping[str, Any]) -> str:
    raw_flows = config_override.get("eval_bus_flows")
    if not isinstance(raw_flows, list):
        base_config = data.get("base_config", {})
        if isinstance(base_config, dict):
            raw_flows = base_config.get("eval_bus_flows")

    if isinstance(raw_flows, list) and len(raw_flows) > 0:
        first_flow = raw_flows[0]
        if isinstance(first_flow, dict):
            name = first_flow.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    return "eval"


def _baseline_spec(name: BaselineName, flow_name: str) -> YamlDict:
    if name in {"fixed", "fixed_time", "fixed_ts"}:
        return {
            "name": f"{flow_name}_fixed_ts",
            "controller": "fixed_time",
            "model_type": "fixed_ts",
            "sim_config_overrides": {
                "fixed_ts": True,
            },
        }
    if name in {"tsp", "rule_tsp"}:
        return {
            "name": f"{flow_name}_tsp",
            "controller": "tsp",
            "model_type": "tsp",
            "sim_config_overrides": {
                "phase_controller": "RuleBasedTSPPhaseController",
                "fixed_ts": True,
                "actuated_ts": False,
            },
        }
    raise ValueError(f"Unsupported baseline shorthand {name!r}.")


def _normalize_baselines(
    flow_name: str,
    config_override: YamlDict,
) -> None:
    if "baseline_eval_models" in config_override or "baselines" not in config_override:
        return

    raw_baselines = config_override.pop("baselines")
    if not isinstance(raw_baselines, list) or len(raw_baselines) == 0:
        raise ValueError("'baselines' must be a non-empty list when provided.")

    baseline_eval_models: list[YamlDict] = []
    for raw_baseline in raw_baselines:
        if isinstance(raw_baseline, str):
            baseline_eval_models.append(
                _baseline_spec(cast(BaselineName, raw_baseline), flow_name)
            )
        elif isinstance(raw_baseline, dict):
            baseline_eval_models.append(copy.deepcopy(raw_baseline))
        else:
            raise ValueError("'baselines' entries must be strings or dictionaries.")
    config_override["baseline_eval_models"] = baseline_eval_models


def _reward_weights_from_setpoint(
    eval_config: EvalConfig,
    setpoint_weights: Any,
) -> dict[str, float] | None:
    reward_weight_keys = eval_config.get("reward_weight_keys")
    if not isinstance(reward_weight_keys, list) or len(reward_weight_keys) == 0:
        return None
    if not isinstance(setpoint_weights, (list, tuple)) or len(setpoint_weights) != len(
        reward_weight_keys
    ):
        return None
    return {
        str(key): float(setpoint_weights[idx])
        for idx, key in enumerate(reward_weight_keys)
    }


def _normalize_rl_models(eval_config: EvalConfig, config_override: YamlDict) -> None:
    if "rl_eval_models" in config_override or "models" not in config_override:
        return

    raw_models = config_override.pop("models")
    if not isinstance(raw_models, list) or len(raw_models) == 0:
        raise ValueError("'models' must be a non-empty list when provided.")

    rl_eval_models: list[YamlDict] = []
    for model_idx, raw_model in enumerate(raw_models):
        raw_model_dict = _as_mapping(raw_model, name=f"models[{model_idx}]")
        model = copy.deepcopy(raw_model_dict)
        model.setdefault("model_type", "ppo")
        model.setdefault(
            "stable_baselines_model",
            config_override.get("stable_baselines_model", "MaskablePPO"),
        )

        setpoint_weights = model.get("setpoint_weights")
        reward_weights = _reward_weights_from_setpoint(eval_config, setpoint_weights)
        if reward_weights is not None:
            sim_overrides = _optional_mapping(
                model.get("sim_config_overrides"),
                name=f"models[{model_idx}].sim_config_overrides",
            )
            model["sim_config_overrides"] = _deep_merge_dicts(
                sim_overrides,
                {"reward_weights": reward_weights},
            )
        rl_eval_models.append(model)

    config_override["rl_eval_models"] = rl_eval_models


def _has_eval_model(config_override: Mapping[str, Any]) -> bool:
    return any(
        key in config_override
        for key in ("model_path", "models", "rl_eval_models", "morl_eval_sources")
    )


def _baseline_suffix(raw_baseline: Any, idx: int) -> str:
    if isinstance(raw_baseline, str):
        return raw_baseline
    if isinstance(raw_baseline, dict):
        name = raw_baseline.get("name", raw_baseline.get("model_type"))
        if isinstance(name, str) and name.strip():
            return name.strip()
    return f"baseline_{idx}"


def _shared_baselines(data: EvalYaml) -> list[Any] | None:
    raw_eval_config = data.get("eval", {})
    if isinstance(raw_eval_config, dict):
        raw_baselines = raw_eval_config.get("baselines")
        if isinstance(raw_baselines, list):
            return raw_baselines
    return None


def _expanded_eval_overrides(data: EvalYaml, config_override: Mapping[str, Any]) -> list[YamlDict]:
    raw_baselines = config_override.get("baselines", _shared_baselines(data))
    if raw_baselines is None or not _has_eval_model(config_override):
        return [copy.deepcopy(dict(config_override))]

    if not isinstance(raw_baselines, list) or len(raw_baselines) == 0:
        raise ValueError("'baselines' must be a non-empty list when provided.")

    base_name = str(config_override.get("name", "eval"))
    model_override = copy.deepcopy(dict(config_override))
    model_override.pop("baselines", None)
    expanded_overrides = [model_override]

    for baseline_idx, raw_baseline in enumerate(raw_baselines):
        baseline_override = copy.deepcopy(dict(config_override))
        for model_key in (
            "model_path",
            "models",
            "rl_eval_models",
            "morl_eval_sources",
            "morl_eval_weights",
            "morl_load_kwargs",
        ):
            baseline_override.pop(model_key, None)
        baseline_override["name"] = f"{base_name}_{_baseline_suffix(raw_baseline, baseline_idx)}"
        baseline_override["baselines"] = [copy.deepcopy(raw_baseline)]
        expanded_overrides.append(baseline_override)

    return expanded_overrides


def _normalize_eval_override(data: EvalYaml, config_override: Mapping[str, Any]) -> YamlDict:
    raw_eval_config = data.get("eval", {})
    if raw_eval_config is None:
        raw_eval_config = {}
    if not isinstance(raw_eval_config, dict):
        raise ValueError("'eval' must be a mapping when provided.")
    eval_config = cast(EvalConfig, raw_eval_config)

    normalized: YamlDict = copy.deepcopy(dict(config_override))

    if "morl_eval_route_timing_seed_pairs" not in normalized:
        route_seeds = eval_config.get("route_seeds")
        timing_seeds = eval_config.get("timing_seeds")
        if "morl_eval_route_seeds" not in normalized and isinstance(route_seeds, list):
            normalized["morl_eval_route_seeds"] = copy.deepcopy(route_seeds)
        if "morl_eval_timing_seeds" not in normalized and isinstance(timing_seeds, list):
            normalized["morl_eval_timing_seeds"] = copy.deepcopy(timing_seeds)

    rows_csv = _rows_csv_path(eval_config, normalized)
    if rows_csv is not None:
        normalized.setdefault("rows_csv", rows_csv)
    for source_key, target_key in (
        ("rows_mode", "rows_csv_mode"),
        ("rows_append", "rows_csv_append"),
        ("rows_formatter", "rows_formatter"),
        ("rows_key_columns", "rows_csv_key_columns"),
    ):
        if source_key in eval_config:
            normalized.setdefault(target_key, copy.deepcopy(eval_config[source_key]))

    _normalize_baselines(_first_eval_flow_name(data, normalized), normalized)
    _normalize_rl_models(eval_config, normalized)
    return normalized


def _normalize_eval_overrides(data: EvalYaml, config_override: Mapping[str, Any]) -> list[YamlDict]:
    return [
        _normalize_eval_override(data, expanded_override)
        for expanded_override in _expanded_eval_overrides(data, config_override)
    ]


def _seed_pairs(config_override: Mapping[str, Any]) -> list[SeedPair]:
    route_seeds = config_override.get("morl_eval_route_seeds")
    timing_seeds = config_override.get("morl_eval_timing_seeds")
    if isinstance(route_seeds, list) and len(route_seeds) > 0:
        if not isinstance(timing_seeds, list) or len(timing_seeds) != len(route_seeds):
            raise ValueError("morl_eval_timing_seeds must match morl_eval_route_seeds length.")
        return [(int(route_seed), int(timing_seed)) for route_seed, timing_seed in zip(route_seeds, timing_seeds, strict=True)]

    return []


def _format_rows(
    rows: list[dict[str, Any]],
    formatter_path: str | None,
    sim_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not formatter_path:
        return rows
    formatter = load_object(formatter_path)
    try:
        formatted = formatter(rows=rows, experiment=type("_RowsContext", (), {"sim_config": sim_config})())
    except TypeError:
        formatted = formatter(rows)
    if not isinstance(formatted, list):
        raise ValueError("rows_formatter must return list[dict].")
    return formatted


def _run_eval_override(data: EvalYaml, normalized_override: YamlDict) -> dict[str, Any]:
    run_data = copy.deepcopy(data)
    run_data["config_overwrite"] = normalized_override
    run_data["name"] = normalized_override["name"]
    run_data["wandb_tags"] = [_eval_tag(normalized_override, base_config=run_data.get("base_config"))]

    seed_pairs = _seed_pairs(normalized_override)
    result_payload: dict[str, Any] = {}

    if len(seed_pairs) == 0:
        # Single run mode
        init_data = _filter_init_data(run_data)
        print(f"Running policy eval experiment: {init_data}")
        ex = morl_eval_experiment(**init_data)
        rows = ex.run()
        result_payload = {"mode": "single", "rows": rows}
    else:
        # Multi-seed mode
        log_seed_runs = bool(normalized_override.get("morl_eval_log_seed_runs", True))
        seed_results: list[dict[str, Any]] = []

        for seed_idx, (route_seed, timing_seed) in enumerate(seed_pairs):
            seed_data = copy.deepcopy(run_data)
            seed_data["config_overwrite"] = morl_eval_experiment._apply_route_timing_seed(
                seed_data["config_overwrite"],
                route_seed=route_seed,
                timing_seed=timing_seed,
            )
            seed_data["name"] = (
                f"{run_data['config_overwrite']['name']}_route{route_seed}_timing{timing_seed}"
            )

            if log_seed_runs:
                tags = list(seed_data.get("wandb_tags", []))
                tags.extend([f"route_seed_{route_seed}", f"timing_seed_{timing_seed}"])
                seed_data["wandb_tags"] = tags
            else:
                base_config = dict(seed_data.get("base_config", {}))
                base_config["wandb"] = False
                seed_data["base_config"] = base_config

            init_data = _filter_init_data(seed_data)
            print(
                "Running policy eval experiment "
                f"route_seed={route_seed} timing_seed={timing_seed} idx={seed_idx}: {init_data}"
            )

            ex = morl_eval_experiment(**init_data)
            rows = ex.run()

            run_id = None
            try:
                run_id = (
                    ex.wandb_run.id
                    if hasattr(ex, "wandb_run") and ex.wandb_run is not None
                    else None
                )
            except Exception:
                pass

            for row in rows:
                row["_seed_run_id"] = run_id
                row["_seed"] = int(route_seed)
                row["_route_seed"] = int(route_seed)
                row["_timing_seed"] = int(timing_seed)

            seed_results.append(
                {
                    "seed": int(route_seed),
                    "route_seed": int(route_seed),
                    "timing_seed": int(timing_seed),
                    "seed_idx": int(seed_idx),
                    "run_id": run_id,
                    "rows": rows,
                }
            )

        aggregated_rows = morl_eval_experiment.aggregate_seed_results(seed_results)

        # Log aggregated results using a temporary experiment instance
        if bool(run_data.get("base_config", {}).get("wandb", False)):
            agg_ex = morl_eval_experiment.__new__(morl_eval_experiment)
            agg_ex.name = run_data["name"]
            agg_ex.group = run_data.get("group", "morl_eval")
            agg_ex.wandb_tags = run_data.get("wandb_tags", [])
            agg_ex.sim_config = copy.deepcopy(run_data.get("base_config", {}))
            agg_ex.sim_config.update(run_data["config_overwrite"])
            agg_ex._log_aggregated_seed_results(
                [route_seed for route_seed, _timing_seed in seed_pairs],
                seed_results,
                aggregated_rows,
            )

        result_payload = {
            "mode": "multi_seed",
            "seed_pairs": seed_pairs,
            "seed_results": seed_results,
            "aggregated_rows": aggregated_rows,
            "rows": aggregated_rows,
        }

        rows_csv = run_data["config_overwrite"].get(
            "rows_csv", run_data.get("base_config", {}).get("rows_csv")
        )
        if isinstance(rows_csv, str) and rows_csv:
            rows_csv_mode = str(
                run_data["config_overwrite"].get(
                    "rows_csv_mode",
                    run_data.get("base_config", {}).get("rows_csv_mode", "aggregate"),
                )
            ).lower()
            csv_rows = (
                [
                    row
                    for seed_result in seed_results
                    for row in seed_result.get("rows", [])
                ]
                if rows_csv_mode in {"seed", "seed_level", "seed-level"}
                else aggregated_rows
            )
            formatter_path = run_data["config_overwrite"].get(
                "rows_formatter", run_data.get("base_config", {}).get("rows_formatter")
            )
            merged_sim_config = copy.deepcopy(run_data.get("base_config", {}))
            merged_sim_config.update(run_data["config_overwrite"])
            csv_rows = _format_rows(
                csv_rows,
                formatter_path if isinstance(formatter_path, str) else None,
                merged_sim_config,
            )
            out_dir = os.path.dirname(rows_csv)
            if len(out_dir) > 0:
                os.makedirs(out_dir, exist_ok=True)
            if bool(
                run_data["config_overwrite"].get(
                    "rows_csv_append",
                    run_data.get("base_config", {}).get("rows_csv_append", False),
                )
            ) and os.path.exists(rows_csv):
                with open(rows_csv, encoding="utf-8", newline="") as file:
                    existing_rows = list(csv.DictReader(file))
                key_columns_raw = run_data["config_overwrite"].get(
                    "rows_csv_key_columns",
                    run_data.get("base_config", {}).get(
                        "rows_csv_key_columns",
                        ["model_type", "name", "route_seed", "timing_seed", "weight_bus"],
                    ),
                )
                if isinstance(key_columns_raw, list) and len(key_columns_raw) > 0:
                    key_columns = [str(column) for column in key_columns_raw]
                    merged: dict[tuple[str, ...], dict[str, Any]] = {}
                    order: list[tuple[str, ...]] = []
                    for row in [*existing_rows, *csv_rows]:
                        key = tuple(str(row.get(column, "")) for column in key_columns)
                        if key not in merged:
                            order.append(key)
                        merged[key] = row
                    csv_rows = [merged[key] for key in order]
                else:
                    csv_rows = [*existing_rows, *csv_rows]
            columns = sorted({key for row in csv_rows for key in row.keys()})
            with open(rows_csv, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(csv_rows)

    return result_payload


def main() -> None:
    args = _build_parser().parse_args()

    data = _load_eval_yaml(args.yaml_path)

    config_overwrites = _as_list(data.get("config_overwrite"), name="config_overwrite")
    config_overwrite = _as_mapping(
        config_overwrites[args.experiment_idx],
        name=f"config_overwrite[{args.experiment_idx}]",
    )
    normalized_overrides = _normalize_eval_overrides(data, config_overwrite)
    result_payloads = [
        _run_eval_override(data, normalized_override)
        for normalized_override in normalized_overrides
    ]
    result_payload: dict[str, Any]
    if len(result_payloads) == 1:
        result_payload = result_payloads[0]
    else:
        result_payload = {
            "mode": "expanded",
            "runs": result_payloads,
            "rows": [
                row
                for payload in result_payloads
                for row in payload.get("rows", [])
            ],
        }

    if isinstance(args.output_json, str) and len(args.output_json) > 0:
        out_dir = os.path.dirname(args.output_json)
        if len(out_dir) > 0:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(result_payload, file)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
