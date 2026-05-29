# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

import csv
import inspect
import os
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

import wandb
from morl_tsp.experiment.callbacks import log_to_wandb, save_trajectory
from morl_tsp.experiment.experiment import experiment, model_map
from morl_tsp.experiment.reproducibility import build_run_provenance, provenance_summary
from morl_tsp.experiment.utils import (
    deep_merge_dicts,
    ensure_env_spec_id,
    filter_kwargs,
    load_object,
    safe_name,
)


@dataclass
class _EvalPolicyCandidate:
    policy_name: str
    policy_source: str
    weights: np.ndarray | None
    action_fn_factory: Callable[[Any], Callable[[np.ndarray, np.ndarray], Any]]
    sim_config_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stable_baselines_model: str | Callable | None = None


class _EvalPolicyProvider(ABC):
    def __init__(self, owner: "morl_eval_experiment") -> None:
        self.owner = owner

    @property
    @abstractmethod
    def share_env_across_candidates(self) -> bool:
        """If true, a single env is reused for all candidates in each flow."""

    @abstractmethod
    def get_candidates(self, env: Any | None = None) -> list[_EvalPolicyCandidate]:
        """Return policy candidates to evaluate."""


class _MorlEvalPolicyProvider(_EvalPolicyProvider):
    @property
    def share_env_across_candidates(self) -> bool:
        return True

    def get_candidates(self, env: Any | None = None) -> list[_EvalPolicyCandidate]:
        if env is None:
            raise ValueError("MORL policy provider requires a flow environment.")

        model = self.owner._build_morl_model_for_env(env)
        candidates: list[_EvalPolicyCandidate] = []
        for policy_name, policy, raw_weights in self.owner._get_morl_eval_candidates(model):
            weights_arr = np.asarray(raw_weights, dtype=np.float64).flatten()
            log_weights = weights_arr if weights_arr.size > 0 else None

            def _make_action_fn(_env, p=policy, w=weights_arr):
                # PCN-style models need desired_return/desired_horizon set before eval
                if hasattr(p, "set_desired_return_and_horizon") and w is not None and w.size > 0:
                    # Map weight vector to desired return: scale max_return by weights
                    max_return = getattr(p, "max_return", None)
                    if max_return is not None:
                        desired_return = np.float32(
                            np.asarray(max_return, dtype=np.float32) * w / max(w.sum(), 1e-8)
                        )
                    else:
                        desired_return = np.float32(w * 100.0)
                    p.set_desired_return_and_horizon(desired_return, np.float32(900.0))
                return lambda obs, acc: self.owner._predict_morl_action(p, obs, w, acc)

            candidates.append(
                _EvalPolicyCandidate(
                    policy_name=str(policy_name),
                    policy_source="morl",
                    weights=log_weights,
                    action_fn_factory=_make_action_fn,
                    metadata={"model_path": str(self.owner.model_path)},
                )
            )
        return candidates


class _RlEvalPolicyProvider(_EvalPolicyProvider):
    def __init__(self, owner: "morl_eval_experiment") -> None:
        super().__init__(owner)
        self.model_specs = owner._load_rl_eval_model_specs()

    @property
    def share_env_across_candidates(self) -> bool:
        # Fixed-setpoint RL models may require distinct env reward_weights.
        return False

    def _build_action_fn_factory(
        self,
        loaded_model: Any,
        model_type: str | Callable | None,
    ) -> Callable[[Any], Callable[[np.ndarray, np.ndarray], Any]]:
        def _factory(eval_env: Any) -> Callable[[np.ndarray, np.ndarray], Any]:
            def _action_fn(obs: np.ndarray, _acc: np.ndarray) -> Any:
                return self.owner._predict_sb3_action_with_model(
                    model=loaded_model,
                    stable_baselines_model=model_type,
                    obs=obs,
                    env=eval_env,
                )

            return _action_fn

        return _factory

    def get_candidates(self, env: Any | None = None) -> list[_EvalPolicyCandidate]:
        del env
        candidates: list[_EvalPolicyCandidate] = []
        for spec in self.model_specs:
            model = spec["model"]
            stable_baselines_model = spec["stable_baselines_model"]
            candidates.append(
                _EvalPolicyCandidate(
                    policy_name=str(spec["name"]),
                    policy_source="sb3",
                    weights=spec["setpoint_weights"],
                    sim_config_overrides=deepcopy(spec["sim_config_overrides"]),
                    stable_baselines_model=stable_baselines_model,
                    action_fn_factory=self._build_action_fn_factory(model, stable_baselines_model),
                    metadata={
                        "model_path": str(spec["model_path"]),
                        "model_type": str(spec.get("model_type", "ppo")),
                    },
                )
            )
        return candidates


class _ControllerEvalPolicyProvider(_EvalPolicyProvider):
    @property
    def share_env_across_candidates(self) -> bool:
        return False

    def get_candidates(self, env: Any | None = None) -> list[_EvalPolicyCandidate]:
        del env
        raw_specs = self.owner.sim_config.get("baseline_eval_models")
        if raw_specs is None:
            raw_specs = [{"name": "fixed_ts", "controller": "fixed_time"}]
        if not isinstance(raw_specs, list) or len(raw_specs) == 0:
            raise ValueError("'baseline_eval_models' must be a non-empty list when provided.")

        candidates: list[_EvalPolicyCandidate] = []
        for idx, raw_spec in enumerate(raw_specs):
            if isinstance(raw_spec, str):
                spec = {"controller": raw_spec}
            elif isinstance(raw_spec, dict):
                spec = deepcopy(raw_spec)
            else:
                raise ValueError(
                    f"Invalid baseline_eval_models[{idx}] type '{type(raw_spec).__name__}'."
                )

            controller = str(spec.get("controller", spec.get("model_type", ""))).lower()
            if controller in {"fixed_time", "fixed-ts", "fixed_ts", "fixed"}:
                policy_source = "fixed_time"
                default_name = "fixed_ts"
                controller_overrides = {"fixed_ts": True}
            elif controller in {"tsp", "rule_based_tsp", "rule-based-tsp"}:
                policy_source = "tsp"
                default_name = "tsp"
                controller_overrides = {
                    "phase_controller": "RuleBasedTSPPhaseController",
                    "fixed_ts": True,
                    "actuated_ts": False,
                }
            else:
                raise ValueError(
                    f"Unsupported baseline_eval_models[{idx}].controller='{controller}'. "
                    "Use fixed_time or tsp."
                )

            sim_config_overrides = deepcopy(spec.get("sim_config_overrides", {}))
            if not isinstance(sim_config_overrides, dict):
                raise ValueError(
                    f"baseline_eval_models[{idx}].sim_config_overrides must be a dict."
                )
            sim_config_overrides = deep_merge_dicts(controller_overrides, sim_config_overrides)
            model_type = str(spec.get("model_type", policy_source))

            def _make_action_fn(_env: Any) -> Callable[[np.ndarray, np.ndarray], Any]:
                # Controller baselines are implemented by the environment's
                # phase controller. Fixed-time and TSP-overlay baselines both
                # run through the no-action path.
                return lambda _obs, _acc: None

            candidates.append(
                _EvalPolicyCandidate(
                    policy_name=str(spec.get("name", default_name)),
                    policy_source=policy_source,
                    weights=None,
                    sim_config_overrides=sim_config_overrides,
                    action_fn_factory=_make_action_fn,
                    metadata={
                        "model_type": model_type,
                        "model_path": str(spec.get("model_path", "")),
                    },
                )
            )
        return candidates


class morl_eval_experiment(experiment):
    """Standalone evaluator for MORL checkpoints and fixed-setpoint RL checkpoints.

    Runs multiple evaluation sweeps across bus-flow settings inside a single W&B run.
    """

    _FLOW_META_KEYS = {"name", "description", "notes", "sim_config_overrides"}
    _EVAL_META_KEYS = {
        "model_path",
        "eval_episodes",
        "episodes",
        "eval_bus_flows",
        "eval_bus_multipliers",
        "iz_bus_multipliers",
        "eval_log_on_step",
        "save_eval_trajectories",
        "reuse_base_scenarios",
        "morl_algorithm",
        "morl_algorithm_kwargs",
        "morl_load_kwargs",
        "morl_eval_sources",
        "morl_eval_policy_ids",
        "morl_eval_limit",
        "morl_eval_weights",
        "eval_log_mode",
        "morl_eval_log_mode",
        "morl_eval_prefer_best_hv_checkpoint",
        "morl_eval_seed_workers",
        "morl_eval_log_seed_runs",
        "rl_eval_models",
        "baseline_eval_models",
        "rows_csv",
        "rows_formatter",
        "rows_csv_append",
        "rows_csv_key_columns",
        "rows_csv_mode",
    }

    def __init__(
        self,
        model_path: str | None = None,
        name: str = "morl_eval",
        group: str = "morl_eval",
        experiment_type: Literal["intersection_zoo", None] = None,
        base_config: dict | None = None,
        config_overwrite: dict | None = None,
        callbacks: list[Literal["wandb", "optuna_prune"]] | None = None,
        stable_baselines_model: Literal["PPO", "A2C", "DQN", "MaskablePPO"] = "PPO",
        name_suffix: str = "",
        wandb_tags: list[str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            group=group,
            experiment_type=experiment_type,
            base_config=base_config,
            config_overwrite=config_overwrite,
            callbacks=callbacks,
            stable_baselines_model=stable_baselines_model,
            hyperparameters={},
            name_suffix=name_suffix,
            wandb_tags=wandb_tags,
        )

        self.model_path = model_path or self.sim_config.get("model_path")
        self.model_format = self._get_model_format()
        self.morl_eval_prefer_best_hv_checkpoint: bool = bool(
            self.sim_config.get("morl_eval_prefer_best_hv_checkpoint", True)
        )

        if self.model_format == "morl_checkpoint" and isinstance(self.model_path, str):
            self.model_path = self._resolve_preferred_morl_checkpoint_path(self.model_path)

        if self.model_format == "morl_checkpoint" and self.model_path is None:
            raise ValueError("Missing required config key 'model_path' for MORL evaluation.")

        rl_eval_models = self.sim_config.get("rl_eval_models")
        has_rl_models = isinstance(rl_eval_models, list) and len(rl_eval_models) > 0
        has_baseline_models = isinstance(self.sim_config.get("baseline_eval_models"), list)
        if (
            self.model_format == "stable_baselines"
            and self.model_path is None
            and not has_rl_models
            and not has_baseline_models
        ):
            raise ValueError(
                "Missing RL model inputs. Provide 'model_path' or 'rl_eval_models' for stable_baselines evaluation."
            )

        if self.model_path is not None and not os.path.exists(self.model_path):
            if self.model_format == "morl_checkpoint" or not has_rl_models:
                raise FileNotFoundError(f"model_path does not exist: {self.model_path}")

        self.deterministic: bool = bool(self.sim_config.get("deterministic", True))
        self.eval_episodes: int = int(
            self.sim_config.get("eval_episodes", self.sim_config.get("episodes", 3))
        )
        # Step-level eval logging is expensive at scale; default to compact episode-level logging.
        self.eval_log_on_step: bool = bool(self.sim_config.get("eval_log_on_step", False))
        self.save_eval_trajectories: bool = bool(
            self.sim_config.get("save_eval_trajectories", False)
        )
        self.reuse_base_scenarios: bool = bool(self.sim_config.get("reuse_base_scenarios", False))
        self.bus_flow_tests: list[dict[str, Any]] = self._get_bus_flow_tests()

        self.morl_algorithm_path: str = str(
            self.sim_config.get("morl_algorithm", "morl_baselines.multi_policy.morld.morld:MORLD")
        )
        self.morl_algorithm_kwargs: dict[str, Any] = dict(
            self.sim_config.get("morl_algorithm_kwargs", {})
        )
        self.morl_load_kwargs: dict[str, Any] = dict(self.sim_config.get("morl_load_kwargs", {}))
        self.morl_eval_sources: list[str] = list(
            self.sim_config.get("morl_eval_sources", ["archive"])
        )
        self.morl_eval_policy_ids: list[int] | None = self.sim_config.get("morl_eval_policy_ids")
        self.morl_eval_limit: int | None = self.sim_config.get("morl_eval_limit")
        self.morl_eval_weights: list[list[float]] | None = self.sim_config.get("morl_eval_weights")
        self.eval_log_mode: str = str(
            self.sim_config.get("eval_log_mode", self.sim_config.get("morl_eval_log_mode", "both"))
        ).lower()
        # Keep this alias for backwards compatibility with older configs.
        self.morl_eval_log_mode: str = self.eval_log_mode
        if self.eval_log_mode not in {"detailed", "compact", "both"}:
            raise ValueError(
                "Invalid eval_log_mode/morl_eval_log_mode. Use one of: detailed | compact | both."
            )
        self._compact_metric_sections: set[str] = set()

        self.eval_base_sim_config = {
            k: deepcopy(v) for k, v in self.sim_config.items() if k not in self._EVAL_META_KEYS
        }
        if self.experiment_type == "intersection_zoo" and not self.reuse_base_scenarios:
            self.eval_base_sim_config.pop("scenarios", None)

        self.rows_csv: str | None = self.sim_config.get("rows_csv")
        self.rows_formatter: str | None = self.sim_config.get("rows_formatter")

        if has_baseline_models:
            self._policy_provider = _ControllerEvalPolicyProvider(self)
        elif self.model_format == "stable_baselines":
            self._policy_provider: _EvalPolicyProvider = _RlEvalPolicyProvider(self)
        else:
            self._policy_provider = _MorlEvalPolicyProvider(self)
        self.model = None

    def _get_model_format(self) -> Literal["stable_baselines", "morl_checkpoint"]:
        rl_eval_models = self.sim_config.get("rl_eval_models")
        if isinstance(rl_eval_models, list) and len(rl_eval_models) > 0:
            return "stable_baselines"

        baseline_eval_models = self.sim_config.get("baseline_eval_models")
        if isinstance(baseline_eval_models, list) and len(baseline_eval_models) > 0:
            return "stable_baselines"

        if isinstance(self.model_path, str):
            path = self.model_path.lower()
            if path.endswith(".zip"):
                return "stable_baselines"
            if path.endswith(".tar"):
                return "morl_checkpoint"

        if "morl_algorithm" in self.sim_config:
            return "morl_checkpoint"

        if self.model_path is None:
            raise ValueError(
                "Could not infer model format because no model_path was provided."
            )

        raise ValueError(
            f"Could not infer model format from model_path='{self.model_path}'."
        )

    def _resolve_preferred_morl_checkpoint_path(self, model_path: str) -> str:
        if not self.morl_eval_prefer_best_hv_checkpoint:
            return model_path

        base_name = os.path.basename(model_path)
        base_stem, _ = os.path.splitext(base_name)
        if base_stem.endswith("_best_hv"):
            return model_path

        model_dir = os.path.dirname(model_path) or "."
        if not os.path.isdir(model_dir):
            return model_path

        candidate_files = [
            file_name
            for file_name in os.listdir(model_dir)
            if os.path.splitext(file_name)[1] in {".tar", ".pt", ".pth", ".zip"}
            and os.path.splitext(file_name)[0].endswith("_best_hv")
        ]
        if len(candidate_files) == 0:
            return model_path

        sorted_candidates = sorted(candidate_files)
        preferred: str | None = None
        for candidate in sorted_candidates:
            candidate_stem = os.path.splitext(candidate)[0]
            candidate_prefix = candidate_stem.removesuffix("_best_hv")
            if len(candidate_prefix) > 0 and base_stem.startswith(candidate_prefix):
                preferred = candidate
                break
        if preferred is None:
            return model_path

        resolved_path = os.path.join(model_dir, preferred)
        if os.path.exists(resolved_path):
            print(f"[morl_eval] Using best-HV checkpoint alias: {resolved_path}")
            return resolved_path

        return model_path

    def _resolve_stable_baselines_model_class(self, stable_baselines_model: str | Callable) -> Any:
        model_class: Any
        if isinstance(stable_baselines_model, str):
            model_class = model_map.get(stable_baselines_model)
        else:
            model_class = stable_baselines_model
        if model_class is None:
            raise ValueError(f"Unknown stable_baselines_model '{stable_baselines_model}'.")
        return model_class

    def _load_stable_baselines_model(
        self,
        model_path: str,
        stable_baselines_model: str | Callable,
        load_kwargs: dict[str, Any],
    ):
        model_class = self._resolve_stable_baselines_model_class(stable_baselines_model)
        return model_class.load(model_path, **load_kwargs)

    def _base_reward_weight_keys(self) -> list[str]:
        for reward_weights in (
            self.eval_base_sim_config.get("reward_weights"),
            self.sim_config.get("reward_weights"),
        ):
            if isinstance(reward_weights, dict) and len(reward_weights) > 0:
                return [str(k) for k in reward_weights.keys()]
        return []

    def _coerce_reward_weights_dict(self, raw_weights: Any, context: str) -> dict[str, float]:
        reward_keys = self._base_reward_weight_keys()

        if isinstance(raw_weights, dict):
            parsed: dict[str, float] = {}
            for key, value in raw_weights.items():
                if not isinstance(value, (int, float, np.floating)):
                    raise ValueError(f"{context}['{key}'] must be numeric.")
                parsed[str(key)] = float(value)
            if len(parsed) == 0:
                raise ValueError(f"{context} cannot be an empty dict.")

            if len(reward_keys) > 0:
                missing = [k for k in reward_keys if k not in parsed]
                extra = [k for k in parsed if k not in reward_keys]
                if len(missing) > 0 or len(extra) > 0:
                    raise ValueError(
                        f"{context} keys must match configured reward_weights keys. "
                        f"Missing={missing}, extra={extra}"
                    )
                return {k: parsed[k] for k in reward_keys}
            return parsed

        if isinstance(raw_weights, (list, tuple, np.ndarray)):
            arr = np.asarray(raw_weights, dtype=np.float64).flatten()
            if len(reward_keys) == 0:
                raise ValueError(
                    f"{context} was given as a list but no base reward_weights keys are configured."
                )
            if arr.shape[0] != len(reward_keys):
                raise ValueError(
                    f"{context} has dim={arr.shape[0]} but expected {len(reward_keys)} "
                    "from base reward_weights."
                )
            return {key: float(arr[i]) for i, key in enumerate(reward_keys)}

        raise ValueError(f"{context} must be a dict or list-like of numeric values.")

    def _weights_dict_to_array(self, reward_weights: dict[str, float] | None) -> np.ndarray | None:
        if reward_weights is None:
            return None
        reward_keys = self._base_reward_weight_keys()
        if len(reward_keys) > 0 and all(key in reward_weights for key in reward_keys):
            ordered = [float(reward_weights[key]) for key in reward_keys]
        else:
            ordered = [float(value) for value in reward_weights.values()]
        return np.asarray(ordered, dtype=np.float64)

    def _load_rl_eval_model_specs(self) -> list[dict[str, Any]]:
        raw_model_specs = self.sim_config.get("rl_eval_models")
        if raw_model_specs is None:
            if self.model_path is None:
                raise ValueError("Missing 'model_path' for RL evaluation.")
            raw_model_specs = [{"name": "sb3_model", "model_path": self.model_path}]

        if not isinstance(raw_model_specs, list) or len(raw_model_specs) == 0:
            raise ValueError("'rl_eval_models' must be a non-empty list when provided.")

        default_load_kwargs = self.sim_config.get("model_load_kwargs", {})
        if not isinstance(default_load_kwargs, dict):
            raise ValueError("'model_load_kwargs' must be a dict.")

        specs: list[dict[str, Any]] = []
        for idx, raw_spec in enumerate(raw_model_specs):
            if isinstance(raw_spec, str):
                spec = {"model_path": raw_spec}
            elif isinstance(raw_spec, dict):
                spec = deepcopy(raw_spec)
            else:
                raise ValueError(
                    f"Invalid rl_eval_models[{idx}] type '{type(raw_spec).__name__}'. "
                    "Expected dict or string model path."
                )

            model_path = spec.get("model_path", self.model_path if idx == 0 else None)
            if model_path is None:
                raise ValueError(f"Missing model_path in rl_eval_models[{idx}].")
            model_path = str(model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"rl_eval_models[{idx}].model_path does not exist: {model_path}"
                )

            stable_baselines_model = spec.get("stable_baselines_model", self.stable_baselines_model)
            model_load_kwargs = dict(default_load_kwargs)
            custom_load_kwargs_raw: Any = spec.get("model_load_kwargs", {})
            if custom_load_kwargs_raw is None:
                custom_load_kwargs = {}
            elif not isinstance(custom_load_kwargs_raw, dict):
                raise ValueError(f"rl_eval_models[{idx}].model_load_kwargs must be a dict.")
            else:
                custom_load_kwargs = custom_load_kwargs_raw
            model_load_kwargs = self._merge_dicts(model_load_kwargs, custom_load_kwargs)

            model = self._load_stable_baselines_model(
                model_path=model_path,
                stable_baselines_model=stable_baselines_model,
                load_kwargs=model_load_kwargs,
            )

            sim_config_overrides_raw: Any = spec.get("sim_config_overrides", {})
            if sim_config_overrides_raw is None:
                sim_config_overrides = {}
            elif not isinstance(sim_config_overrides_raw, dict):
                raise ValueError(f"rl_eval_models[{idx}].sim_config_overrides must be a dict.")
            else:
                sim_config_overrides = sim_config_overrides_raw
            sim_config_overrides = deepcopy(sim_config_overrides)

            setpoint_input = spec.get("setpoint_weights", spec.get("reward_weights"))
            if setpoint_input is None and "reward_weights" in sim_config_overrides:
                setpoint_input = sim_config_overrides["reward_weights"]

            setpoint_weights_dict: dict[str, float] | None = None
            if setpoint_input is not None:
                setpoint_weights_dict = self._coerce_reward_weights_dict(
                    setpoint_input, context=f"rl_eval_models[{idx}]"
                )
                sim_config_overrides = self._merge_dicts(
                    sim_config_overrides,
                    {"reward_weights": deepcopy(setpoint_weights_dict)},
                )

            specs.append(
                {
                    "name": str(spec.get("name", f"sb3_model_{idx}")),
                    "model_type": str(spec.get("model_type", "ppo")),
                    "model": model,
                    "model_path": model_path,
                    "stable_baselines_model": stable_baselines_model,
                    "sim_config_overrides": sim_config_overrides,
                    "setpoint_weights": self._weights_dict_to_array(setpoint_weights_dict),
                }
            )

        return specs

    def _get_bus_flow_tests(self) -> list[dict[str, Any]]:
        raw_flows = self.sim_config.get("eval_bus_flows")
        if raw_flows is None:
            multipliers = self.sim_config.get(
                "eval_bus_multipliers", self.sim_config.get("iz_bus_multipliers")
            )
            if multipliers is not None:
                raw_flows = [
                    {
                        "name": f"bus_x{float(multiplier):.2f}",
                        "iz_bus_multiplier": float(multiplier),
                    }
                    for multiplier in multipliers
                ]

        if not isinstance(raw_flows, list) or len(raw_flows) == 0:
            raise ValueError(
                "No bus-flow tests configured. Set 'eval_bus_flows' (list[dict]) or "
                "'eval_bus_multipliers' (list[float])."
            )

        normalized_flows: list[dict[str, Any]] = []
        for idx, flow in enumerate(raw_flows):
            if isinstance(flow, (int, float)):
                normalized_flows.append(
                    {"name": f"bus_x{float(flow):.2f}", "iz_bus_multiplier": float(flow)}
                )
                continue
            if not isinstance(flow, dict):
                raise ValueError(
                    f"Invalid flow entry at index {idx}: expected dict|float, got {type(flow).__name__}."
                )
            entry = deepcopy(flow)
            entry.setdefault("name", f"flow_{idx}")
            normalized_flows.append(entry)

        return normalized_flows

    def _compact_section_key(self, flow_slug: str, flow_overrides: dict[str, Any]) -> str:
        if "iz_bus_multiplier" in flow_overrides:
            mult = self._to_metric_token(flow_overrides["iz_bus_multiplier"])
            return self._safe_name(f"bus_multiplier_{mult}")

        timetable_cfg = flow_overrides.get("iz_bus_timetable_config")
        if isinstance(timetable_cfg, dict):
            parts: list[str] = []
            route_count = timetable_cfg.get("route_count")
            if route_count is not None:
                parts.append(f"route_count_{self._to_metric_token(route_count)}")
            hmin = timetable_cfg.get("headway_min_seconds")
            hmax = timetable_cfg.get("headway_max_seconds")
            if hmin is not None or hmax is not None:
                parts.append(f"headway_{self._to_metric_token(hmin)}_{self._to_metric_token(hmax)}")
            if len(parts) > 0:
                return self._safe_name("_".join(parts))

        return flow_slug

    def _is_detailed_logging_enabled(self) -> bool:
        return self.eval_log_mode in {"detailed", "both"}

    def _is_compact_logging_enabled(self) -> bool:
        return self.eval_log_mode in {"compact", "both"}

    def _ensure_compact_metric_section(self, section_key: str) -> None:
        if section_key in self._compact_metric_sections or not self.sim_config.get("wandb", False):
            return

        step_metric = f"eval_compact/{section_key}/setpoint_idx"
        wandb.define_metric(step_metric)
        wandb.define_metric(f"eval_compact/{section_key}/*", step_metric=step_metric)
        self._compact_metric_sections.add(section_key)

    def _log_compact_setpoint(
        self,
        section_key: str,
        setpoint_idx: int,
        summary: dict[str, float],
        weights: np.ndarray | None = None,
        flow_overrides: dict[str, Any] | None = None,
    ) -> None:
        if not self.sim_config.get("wandb", False):
            return

        self._ensure_compact_metric_section(section_key)
        payload: dict[str, float] = {
            f"eval_compact/{section_key}/setpoint_idx": float(setpoint_idx),
        }

        normalized_summary = self._add_bus_faster_gap(self._normalize_numeric_metrics(summary))
        for key, value in normalized_summary.items():
            payload[f"eval_compact/{section_key}/{self._safe_name(key)}"] = float(value)

        if weights is not None:
            for idx, weight in enumerate(np.asarray(weights, dtype=np.float64).flatten()):
                payload[f"eval_compact/{section_key}/weight_{idx}"] = float(weight)

        flow_overrides = flow_overrides or {}
        if "iz_bus_multiplier" in flow_overrides:
            payload[f"eval_compact/{section_key}/bus_multiplier"] = float(
                flow_overrides["iz_bus_multiplier"]
            )

        timetable_cfg = flow_overrides.get("iz_bus_timetable_config")
        if isinstance(timetable_cfg, dict):
            if timetable_cfg.get("route_count") is not None:
                payload[f"eval_compact/{section_key}/route_count"] = float(
                    timetable_cfg["route_count"]
                )
            if timetable_cfg.get("headway_min_seconds") is not None:
                payload[f"eval_compact/{section_key}/headway_min_seconds"] = float(
                    timetable_cfg["headway_min_seconds"]
                )
            if timetable_cfg.get("headway_max_seconds") is not None:
                payload[f"eval_compact/{section_key}/headway_max_seconds"] = float(
                    timetable_cfg["headway_max_seconds"]
                )

        log_to_wandb(payload)

    @contextmanager
    def _temporary_sim_config(self, sim_config: dict[str, Any]):
        prev = self.sim_config
        self.sim_config = sim_config
        try:
            yield
        finally:
            self.sim_config = prev

    def _get_flow_overrides(self, flow_cfg: dict[str, Any]) -> dict[str, Any]:
        if "sim_config_overrides" in flow_cfg:
            nested = flow_cfg["sim_config_overrides"]
            if not isinstance(nested, dict):
                raise ValueError("'sim_config_overrides' inside eval_bus_flows must be a dict.")
            return deepcopy(nested)
        return {k: deepcopy(v) for k, v in flow_cfg.items() if k not in self._FLOW_META_KEYS}

    def _build_env_for_flow(
        self,
        flow_name: str,
        flow_overrides: dict[str, Any],
        stable_baselines_model_override: str | Callable | None = None,
        multi_objective_override: bool | None = None,
    ):
        flow_slug = self._safe_name(flow_name)
        flow_sim_config = self._merge_dicts(self.eval_base_sim_config, flow_overrides)
        flow_log_dir = os.path.join(str(self.sim_config["log_dir"]), "eval_bus_flow", flow_slug)
        os.makedirs(flow_log_dir, exist_ok=True)
        flow_sim_config["log_dir"] = flow_log_dir
        flow_sim_config["experiment_name"] = (
            f"{self.eval_base_sim_config.get('experiment_name', self.name)}_{flow_slug}"
        )

        with self._temporary_sim_config(flow_sim_config):
            if self.experiment_type == "intersection_zoo" and "scenarios" not in self.sim_config:
                self.sim_config["scenarios"] = self._init_intersection_zoo_scenarios()
            elif self.experiment_type != "intersection_zoo":
                self._add_sim_files_to_sim_config_dict()

            env_input = self._get_env_inputs()
            env_input["eval_mode"] = True
            if self.experiment_type == "intersection_zoo" and "eval_start_time" not in env_input:
                env_input["eval_start_time"] = 0
            use_multi_objective = (
                self.model_format == "morl_checkpoint"
                if multi_objective_override is None
                else bool(multi_objective_override)
            )
            if use_multi_objective:
                env_input["multi_objective"] = True

            previous_model_type = self.stable_baselines_model
            if use_multi_objective:
                self.stable_baselines_model = "PPO"
            elif stable_baselines_model_override is not None:
                # _get_env checks this value to decide if ActionMasker should wrap the env.
                self.stable_baselines_model = (
                    "MaskablePPO"
                    if self._is_maskable_model(stable_baselines_model_override)
                    else "PPO"
                )
            try:
                env = self._get_env(**env_input)
                ensure_env_spec_id(
                    env, default_id=str(flow_sim_config.get("experiment_name", flow_slug))
                )
            finally:
                self.stable_baselines_model = previous_model_type
        return env

    def _build_morl_model_for_env(self, env) -> Any:
        assert self.model_path is not None, "model_path is required for MORL checkpoint evaluation."
        model_class = load_object(self.morl_algorithm_path)
        ctor_kwargs = dict(self.morl_algorithm_kwargs)

        # Eval should stay lightweight and avoid creating nested MORL logging runs.
        ctor_kwargs["log"] = bool(
            self.sim_config.get("morl_eval_enable_algorithm_logging", False)
        ) and bool(self.sim_config.get("wandb", False))

        # Infer MORLD population size from checkpoint if not provided.
        if (
            getattr(model_class, "__name__", "") == "MORLD"
            and "pop_size" not in ctor_kwargs
            and self.model_path.lower().endswith(".tar")
        ):
            inferred_pop_size = self._infer_morld_population_size(self.model_path)
            if inferred_pop_size is not None:
                ctor_kwargs["pop_size"] = inferred_pop_size

        for env_key in ("env", "train_env", "environment"):
            if env_key in inspect.signature(model_class).parameters and env_key not in ctor_kwargs:
                ctor_kwargs[env_key] = env

        # Resolve callable strings (e.g. "module:function") to actual callables.
        ctor_sig = inspect.signature(model_class)
        for key, val in list(ctor_kwargs.items()):
            if isinstance(val, str) and ":" in val and key in ctor_sig.parameters:
                try:
                    ctor_kwargs[key] = load_object(val)
                except Exception:
                    pass

        ctor_kwargs = filter_kwargs(model_class, ctor_kwargs)
        model = model_class(**ctor_kwargs)

        if not hasattr(model, "load"):
            raise ValueError(
                f"Configured morl model class '{self.morl_algorithm_path}' does not expose a load(path, ...) method."
            )

        load_kwargs = dict(self.morl_load_kwargs)
        load_kwargs = filter_kwargs(model.load, load_kwargs)
        # Some algorithms (e.g. EUPG) have `path` as the second kwarg, not first positional.
        load_params = list(inspect.signature(model.load).parameters.keys())
        if len(load_params) >= 1 and load_params[0] != "path" and "path" in load_params:
            load_kwargs["path"] = self.model_path
            model.load(**load_kwargs)
        else:
            model.load(self.model_path, **load_kwargs)
        return model

    def _get_morl_eval_candidates(self, model: Any) -> list[tuple[str, Any, np.ndarray]]:
        available: list[tuple[str, Any]] = []
        selected: list[tuple[str, Any, np.ndarray]] = []

        sources = [str(s) for s in self.morl_eval_sources]
        for source in sources:
            if source == "archive":
                archive = getattr(model, "archive", None)
                individuals = getattr(archive, "individuals", None)
                if isinstance(individuals, list):
                    available.extend(
                        [(f"archive_{idx}", policy) for idx, policy in enumerate(individuals)]
                    )
            elif source == "population":
                population = getattr(model, "population", None)
                if isinstance(population, list):
                    available.extend(
                        [(f"population_{idx}", policy) for idx, policy in enumerate(population)]
                    )
            elif source == "model":
                # Population-based algorithms (MORLD) don't expose eval() on the
                # top-level model.  Fall back to population + archive members.
                population = getattr(model, "population", None)
                if (
                    isinstance(population, list)
                    and len(population) > 0
                    and not hasattr(model, "eval")
                ):
                    available.extend([(f"population_{idx}", p) for idx, p in enumerate(population)])
                    arch = getattr(model, "archive", None)
                    arch_indiv = getattr(arch, "individuals", None)
                    if isinstance(arch_indiv, list):
                        available.extend(
                            [(f"archive_{idx}", p) for idx, p in enumerate(arch_indiv)]
                        )
                else:
                    for idx, weights in enumerate(self._resolve_model_eval_weights(model)):
                        selected.append((f"model_{idx}", model, weights))
            else:
                raise ValueError(
                    f"Unsupported morl_eval_sources entry '{source}'. Use archive|population|model."
                )
        if len(available) == 0 and len(selected) == 0:
            raise ValueError(
                "No MORL candidates found for evaluation. "
                "Check morl_eval_sources and whether the checkpoint has archive/population entries. "
                "For single-policy checkpoints, use morl_eval_sources: [model]."
            )
        for default_name, policy in available:
            policy_id = getattr(policy, "id", None)
            if self.morl_eval_policy_ids is not None and policy_id not in self.morl_eval_policy_ids:
                continue
            wrapped = getattr(policy, "wrapped", policy)
            weights = np.asarray(getattr(policy, "weights", []), dtype=np.float64)
            policy_name = f"{default_name}_id{policy_id}" if policy_id is not None else default_name
            selected.append((policy_name, wrapped, weights))

        if self.morl_eval_limit is not None:
            selected = selected[: max(0, int(self.morl_eval_limit))]
        if len(selected) == 0:
            raise ValueError(
                "No MORL candidates selected after filtering. "
                "Check morl_eval_policy_ids/morl_eval_limit settings."
            )
        return selected

    def _resolve_model_eval_weights(self, model: Any) -> list[np.ndarray]:
        reward_dim = int(getattr(model, "reward_dim", 0))
        if reward_dim <= 0:
            raise ValueError(
                "Could not infer reward_dim from MORL model. "
                "Set 'morl_eval_weights' explicitly in the eval config."
            )

        if self.morl_eval_weights is None:
            return [np.ones(reward_dim, dtype=np.float64) / float(reward_dim)]

        normalized: list[np.ndarray] = []
        for idx, weight in enumerate(self.morl_eval_weights):
            w = np.asarray(weight, dtype=np.float64).flatten()
            if w.shape[0] != reward_dim:
                raise ValueError(
                    f"morl_eval_weights[{idx}] has dim={w.shape[0]} but reward_dim={reward_dim}."
                )
            if np.any(w < 0.0):
                raise ValueError(f"morl_eval_weights[{idx}] contains negative entries.")
            total = float(np.sum(w))
            if total <= 0.0:
                raise ValueError(f"morl_eval_weights[{idx}] sums to 0.")
            normalized.append(w / total)
        return normalized

    def _predict_sb3_action_with_model(
        self,
        model: Any,
        stable_baselines_model: str | Callable | None,
        obs: np.ndarray,
        env: Any,
    ) -> Any:
        obs_np = np.asarray(obs)
        if self._is_maskable_model(stable_baselines_model):
            model_action_n = getattr(getattr(model, "action_space", None), "n", None)
            env_action_n = getattr(getattr(env, "action_space", None), "n", None)
            if (
                isinstance(model_action_n, int)
                and isinstance(env_action_n, int)
                and model_action_n != env_action_n
            ):
                # Allow cross-topology evals where env/model action counts differ.
                action, _ = model.predict(obs_np, deterministic=self.deterministic)
                return action

            masks = experiment.mask_fn(env)
            try:
                action, _ = model.predict(
                    obs_np, deterministic=self.deterministic, action_masks=masks
                )
                return action
            except (TypeError, RuntimeError, ValueError):
                # Fallback for non-maskable models loaded under this setting.
                pass
        action, _ = model.predict(obs_np, deterministic=self.deterministic)
        return action

    def _log_eval_step(
        self,
        env,
        prefix: str,
        episode_push_history: dict[str, list[float]],
        last_push: dict[str, float],
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            current_push = {f"ts/{k}": v for k, v in env.get_numerical_info_dict().items()}
            current_push.update(
                {f"ts_bus/{k}": v for k, v in env.get_numerical_info_dict_bus().items()}
            )
            current_push.update(
                {f"rewards/{k}": v for k, v in env.get_all_reward_metrics().items()}
            )

        prefixed_push = {f"{prefix}{k}": v for k, v in current_push.items()}
        changed_push = {k: v for k, v in prefixed_push.items() if last_push.get(k) != v}
        if self.sim_config.get("wandb", False) and len(changed_push) > 0:
            log_to_wandb(changed_push)

        for k, v in current_push.items():
            episode_push_history[f"episode/{k.split('/')[1]}"].append(v)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for k, v in env.get_mean_accumulated_waiting_time_per_lane().items():
                episode_push_history[f"episode_lane/{k}"].append(v)

        last_push.clear()
        last_push.update(prefixed_push)

    def _evaluate_policy(
        self,
        env,
        eval_prefix: str,
        action_fn: Callable[[np.ndarray, np.ndarray], Any],
        emit_detailed_logs: bool = True,
    ) -> dict[str, float]:
        eval_episode_push_history: dict[str, list[float]] = defaultdict(list)
        eval_last_push: dict[str, float] = {}
        episode_returns: list[float] = []
        episode_infos: list[dict[str, Any]] = []

        reward_dim = 1
        try:
            reward_dim = int(env.unwrapped.reward_space.shape[0])  # type: ignore[attr-defined]
        except Exception:
            reward_dim = 1

        for ep in range(self.eval_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            accrued_reward = np.zeros(reward_dim, dtype=np.float64)

            while not (terminated or truncated):
                action = action_fn(np.asarray(obs), accrued_reward)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += self._reward_to_scalar(reward)

                if isinstance(reward, np.ndarray):
                    accrued_reward += reward.astype(np.float64)

                if self.eval_log_on_step and emit_detailed_logs:
                    self._log_eval_step(
                        env=env,
                        prefix=eval_prefix,
                        episode_push_history=eval_episode_push_history,
                        last_push=eval_last_push,
                    )

            episode_returns.append(episode_return)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                episode_info = env.get_episode_info()
            episode_info = self._normalize_episode_info(episode_info)
            episode_infos.append(episode_info)

            if self.sim_config.get("wandb", False) and emit_detailed_logs:
                log_to_wandb({f"{eval_prefix}episode_info/{k}": v for k, v in episode_info.items()})

                if self.save_eval_trajectories:
                    save_trajectory(
                        trajectories=env.trajectory,
                        ep=ep + 1,
                        prefix=eval_prefix.replace("/", "_"),
                    )

            eval_episode_push_history = defaultdict(list)

        summary = {
            "episode_return": float(np.mean(episode_returns)),
            "episode_return_min": float(np.min(episode_returns)),
            "episode_return_max": float(np.max(episode_returns)),
            "n_eval_episodes": float(self.eval_episodes),
        }
        summary.update(self._extract_numeric_episode_info(episode_infos))
        return summary

    def _evaluate_candidate_for_flow(
        self,
        env: Any,
        flow_slug: str,
        compact_section_key: str,
        flow_overrides: dict[str, Any],
        candidate: _EvalPolicyCandidate,
        compact_setpoint_idx_by_section: dict[str, int],
    ) -> dict[str, Any]:
        policy_slug = self._safe_name(candidate.policy_name)
        prefix = f"eval/{flow_slug}/{policy_slug}/"
        setpoint_idx = int(compact_setpoint_idx_by_section[compact_section_key])
        compact_setpoint_idx_by_section[compact_section_key] += 1

        summary = self._evaluate_policy(
            env=env,
            eval_prefix=prefix,
            action_fn=candidate.action_fn_factory(env),
            emit_detailed_logs=self._is_detailed_logging_enabled(),
        )
        summary = self._add_bus_faster_gap(self._normalize_numeric_metrics(summary))

        if self.sim_config.get("wandb", False) and self._is_detailed_logging_enabled():
            log_to_wandb({f"{prefix}summary/{k}": v for k, v in summary.items()})
        if self._is_compact_logging_enabled():
            self._log_compact_setpoint(
                section_key=compact_section_key,
                setpoint_idx=setpoint_idx,
                summary=summary,
                weights=candidate.weights,
                flow_overrides=flow_overrides,
            )

        row: dict[str, Any] = {
            "flow_name": flow_slug,
            "policy_name": policy_slug,
            "policy_source": candidate.policy_source,
            "model_type": candidate.metadata.get("model_type", candidate.policy_source),
            "compact_section": compact_section_key,
            "setpoint_idx": setpoint_idx,
            "weights": self._weights_to_string(candidate.weights),
            "route_seed": self.sim_config.get(
                "iz_bus_route_seed", self.sim_config.get("iz_bus_gen_seed")
            ),
            "timing_seed": self.sim_config.get(
                "iz_bus_timing_seed", self.sim_config.get("iz_seed")
            ),
            "seed": self.sim_config.get("iz_bus_gen_seed", self.sim_config.get("iz_seed")),
            **summary,
        }
        for key, value in candidate.metadata.items():
            if key not in row:
                row[key] = value
        return row

    def run(self) -> list[dict[str, Any]]:
        summary_rows: list[dict[str, Any]] = []
        compact_setpoint_idx_by_section: dict[str, int] = defaultdict(int)

        try:
            for flow_idx, flow_cfg in enumerate(self.bus_flow_tests):
                flow_name = str(flow_cfg.get("name", f"flow_{flow_idx}"))
                flow_slug = self._safe_name(flow_name)
                flow_overrides = self._get_flow_overrides(flow_cfg)
                compact_section_key = self._compact_section_key(flow_slug, flow_overrides)

                if self._policy_provider.share_env_across_candidates:
                    env = self._build_env_for_flow(
                        flow_name=flow_slug,
                        flow_overrides=flow_overrides,
                        multi_objective_override=(self.model_format == "morl_checkpoint"),
                    )
                    try:
                        candidates = self._policy_provider.get_candidates(env=env)
                        for candidate in candidates:
                            summary_rows.append(
                                self._evaluate_candidate_for_flow(
                                    env=env,
                                    flow_slug=flow_slug,
                                    compact_section_key=compact_section_key,
                                    flow_overrides=flow_overrides,
                                    candidate=candidate,
                                    compact_setpoint_idx_by_section=compact_setpoint_idx_by_section,
                                )
                            )
                    finally:
                        self._close_env(env)
                else:
                    candidates = self._policy_provider.get_candidates(env=None)
                    for candidate in candidates:
                        candidate_flow_overrides = self._merge_dicts(
                            flow_overrides, candidate.sim_config_overrides
                        )
                        env_flow_name = f"{flow_slug}_{self._safe_name(candidate.policy_name)}"
                        env = self._build_env_for_flow(
                            flow_name=env_flow_name,
                            flow_overrides=candidate_flow_overrides,
                            stable_baselines_model_override=candidate.stable_baselines_model,
                            multi_objective_override=False,
                        )
                        try:
                            summary_rows.append(
                                self._evaluate_candidate_for_flow(
                                    env=env,
                                    flow_slug=flow_slug,
                                    compact_section_key=compact_section_key,
                                    flow_overrides=candidate_flow_overrides,
                                    candidate=candidate,
                                    compact_setpoint_idx_by_section=compact_setpoint_idx_by_section,
                                )
                            )
                        finally:
                            self._close_env(env)

            if self.sim_config.get("wandb", False) and len(summary_rows) > 0:
                columns = sorted({k for row in summary_rows for k in row.keys()})
                table = wandb.Table(columns=columns)
                for row in summary_rows:
                    table.add_data(*[row.get(column) for column in columns])
                wandb.log({"eval/summary_table": table})
            self._write_rows_csv(summary_rows)
            return summary_rows

        finally:
            if self.sim_config.get("wandb", False) and self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Multi-seed evaluation orchestration
    # -------------------------------------------------------------------------

    def _log_aggregated_seed_results(
        self,
        seeds: list[int],
        seed_results: list[dict[str, Any]],
        aggregated_rows: list[dict[str, Any]],
    ) -> None:
        """Log aggregated seed results to a separate W&B run."""
        from morl_tsp import config as sumo_config

        if len(aggregated_rows) == 0:
            return
        if not bool(self.sim_config.get("wandb", False)):
            return

        base_name = str(self.sim_config.get("experiment_name", self.name))
        run_name = f"{base_name}_seed_mean"
        group = str(self.group)
        tags = list(dict.fromkeys(list(self.wandb_tags or []) + ["seed_mean"]))
        seed_run_ids = [str(res["run_id"]) for res in seed_results if res.get("run_id")]
        provenance = build_run_provenance(
            output_root=str(self.sim_config.get("log_dir", "")),
            config_path=self.sim_config.get("yaml_path"),
            model_path=self.sim_config.get("model_path"),
            extra={"eval_aggregation": "mean_across_seed_runs"},
        )

        wandb_config = {
            "name": run_name,
            "seed_list": list(seeds),
            "seed_run_count": len(seed_results),
            "seed_run_ids": seed_run_ids,
            "source_run_name": base_name,
            "eval_aggregation": "mean_across_seed_runs",
            "machine": provenance.get("machine", {}),
            "experiment_provenance_summary": provenance_summary(provenance),
        }

        run = wandb.init(
            project=sumo_config.WANDB_PROJECT_NAME,
            dir=f"{sumo_config.ROOT_PATH}/wandb/{group}/",
            config=wandb_config,
            sync_tensorboard=False,
            monitor_gym=False,
            save_code=True,
            name=run_name,
            group=group,
            tags=tags,
        )

        metric_sections = sorted(
            {str(row.get("compact_section", "global")) for row in aggregated_rows}
        )
        for section in metric_sections:
            step_metric = f"eval_compact_mean/{section}/setpoint_idx"
            wandb.define_metric(step_metric)
            wandb.define_metric(f"eval_compact_mean/{section}/*", step_metric=step_metric)

        for row in aggregated_rows:
            section = str(row.get("compact_section", "global"))
            setpoint_idx = float(row.get("setpoint_idx", 0))
            payload: dict[str, float] = {
                f"eval_compact_mean/{section}/setpoint_idx": setpoint_idx,
            }
            for key, value in row.items():
                if key in {"setpoint_idx", "compact_section"}:
                    continue
                if isinstance(value, (int, float, np.floating)):
                    payload[f"eval_compact_mean/{section}/{safe_name(str(key))}"] = float(value)
            wandb.log(payload)

        columns = sorted({k for row in aggregated_rows for k in row.keys()})
        table = wandb.Table(columns=columns)
        for row in aggregated_rows:
            table.add_data(*[row.get(column) for column in columns])
        wandb.log({"eval_compact_mean/summary_table": table})
        run.finish()

    def _format_rows_for_csv(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.rows_formatter:
            return rows
        formatter = load_object(self.rows_formatter)
        try:
            formatted = formatter(rows=rows, experiment=self)
        except TypeError:
            formatted = formatter(rows)
        if not isinstance(formatted, list):
            raise ValueError("rows_formatter must return list[dict].")
        return formatted

    def _write_rows_csv(self, rows: list[dict[str, Any]]) -> None:
        if not self.rows_csv or len(rows) == 0:
            return
        formatted_rows = self._format_rows_for_csv(rows)
        if len(formatted_rows) == 0:
            return
        path = Path(self.rows_csv)
        if not path.is_absolute():
            path = Path(str(self.sim_config.get("log_dir", "."))) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        if bool(self.sim_config.get("rows_csv_append", False)):
            formatted_rows = self._merge_existing_rows_csv(path, formatted_rows)
        columns = sorted({key for row in formatted_rows for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(formatted_rows)
        if self.sim_config.get("wandb", False) and self.wandb_run is not None:
            try:
                self.wandb_run.save(str(path))
            except Exception as exc:
                warnings.warn(f"Failed to save rows_csv to W&B: {exc}", stacklevel=2)

    def _merge_existing_rows_csv(
        self, path: Path, new_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not path.exists():
            return new_rows
        with path.open("r", encoding="utf-8", newline="") as file:
            existing_rows = list(csv.DictReader(file))
        if len(existing_rows) == 0:
            return new_rows

        key_columns_raw = self.sim_config.get(
            "rows_csv_key_columns",
            ["model_type", "name", "route_seed", "timing_seed", "weight_bus"],
        )
        if not isinstance(key_columns_raw, list) or len(key_columns_raw) == 0:
            return [*existing_rows, *new_rows]
        key_columns = [str(column) for column in key_columns_raw]

        merged: dict[tuple[str, ...], dict[str, Any]] = {}
        order: list[tuple[str, ...]] = []
        for row in [*existing_rows, *new_rows]:
            key = tuple(str(row.get(column, "")) for column in key_columns)
            if key not in merged:
                order.append(key)
            merged[key] = row
        return [merged[key] for key in order]

    # -------------------------------------------------------------------------
    # Static and class helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_maskable_model(stable_baselines_model: str | Callable | None) -> bool:
        if isinstance(stable_baselines_model, str):
            return stable_baselines_model == "MaskablePPO"
        return getattr(stable_baselines_model, "__name__", "") == "MaskablePPO"

    @staticmethod
    def _safe_name(name: str) -> str:
        return safe_name(name, fallback="flow")

    @staticmethod
    def _to_metric_token(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    @staticmethod
    def _numeric_summary(summary: dict[str, Any]) -> dict[str, float]:
        numeric: dict[str, float] = {}
        for key, value in summary.items():
            if isinstance(value, (int, float, np.floating)):
                numeric[key] = float(value)
        return numeric

    @staticmethod
    def _normalize_metric_name(name: str) -> str | None:
        lowered = name.lower()
        if "_std" in lowered or lowered.endswith("std"):
            return None
        normalized = str(name)
        while normalized.endswith("_mean"):
            normalized = normalized[:-5]
        return normalized

    @classmethod
    def _normalize_numeric_metrics(cls, metrics: dict[str, Any]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, value in cls._numeric_summary(metrics).items():
            normalized_key = cls._normalize_metric_name(key)
            if normalized_key is None:
                continue
            normalized[normalized_key] = float(value)
        return normalized

    @classmethod
    def _add_bus_faster_gap(cls, metrics: dict[str, float]) -> dict[str, float]:
        with_gap = dict(metrics)
        bus_key = cls._normalize_metric_name("Bus_crossing_time_mean")
        car_key = cls._normalize_metric_name("Car_crossing_time_mean")
        if bus_key in with_gap and car_key in with_gap:
            with_gap["Bus_faster_crossing_time_gap"] = float(with_gap[car_key] - with_gap[bus_key])
        return with_gap

    @classmethod
    def _normalize_episode_info(cls, episode_info: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in episode_info.items():
            normalized_key = cls._normalize_metric_name(key)
            if normalized_key is None:
                continue
            normalized[normalized_key] = value
        numeric_view = cls._normalize_numeric_metrics(normalized)
        normalized_with_gap = cls._add_bus_faster_gap(numeric_view)
        for key, value in normalized_with_gap.items():
            normalized[key] = value
        return normalized

    @staticmethod
    def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        return deep_merge_dicts(base, overrides)

    @staticmethod
    def _infer_morld_population_size(model_path: str) -> int | None:
        try:
            import torch
        except Exception:
            return None

        try:
            params = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return None

        indices: list[int] = []
        for key in params.keys():
            if not key.startswith("population_policy_"):
                continue
            suffix = key.replace("population_policy_", "", 1)
            if suffix.isdigit():
                indices.append(int(suffix))
        if len(indices) == 0:
            return None
        return max(indices) + 1

    @staticmethod
    def _predict_morl_action(
        policy: Any,
        obs: np.ndarray,
        weights: np.ndarray | None,
        accrued_reward: np.ndarray,
    ) -> Any:
        if not hasattr(policy, "eval"):
            raise ValueError(
                f"Policy object of type '{type(policy).__name__}' does not expose an eval(...) method."
            )

        eval_fn = policy.eval
        params = set(inspect.signature(eval_fn).parameters.keys())
        obs_np = np.asarray(obs)

        if "w" in params and weights is not None:
            return eval_fn(obs_np, w=weights)
        if "weights" in params and weights is not None:
            return eval_fn(obs_np, weights=weights)
        if "accrued_reward" in params:
            return eval_fn(obs_np, accrued_reward=accrued_reward)
        return eval_fn(obs_np)

    @staticmethod
    def _reward_to_scalar(reward: Any) -> float:
        if isinstance(reward, np.ndarray):
            return float(np.sum(reward))
        if isinstance(reward, (list, tuple)):
            return float(np.sum(np.asarray(reward, dtype=np.float64)))
        return float(reward)

    @staticmethod
    def _extract_numeric_episode_info(episode_infos: list[dict[str, Any]]) -> dict[str, float]:
        summary: dict[str, float] = {}
        keys = {
            k
            for info in episode_infos
            for k, v in info.items()
            if isinstance(v, (int, float, np.floating))
        }
        for key in sorted(keys):
            values = [
                float(info[key])
                for info in episode_infos
                if isinstance(info.get(key), (int, float, np.floating))
            ]
            if len(values) == 0:
                continue
            summary[key] = float(np.mean(values))
        return summary

    @staticmethod
    def _weights_to_string(weights: np.ndarray | None) -> str:
        if weights is None:
            return ""
        return np.array2string(
            np.asarray(weights, dtype=np.float64).flatten(), precision=4, separator=","
        )

    @staticmethod
    def _close_env(env: Any) -> None:
        try:
            env.close()
        except Exception:
            pass
        try:
            env.cleanup()
        except Exception:
            pass

    @staticmethod
    def _apply_bus_generation_seed(config_override: dict[str, Any], seed: int) -> dict[str, Any]:
        """Apply bus generation seed to config and nested flow configs."""
        return morl_eval_experiment._apply_route_timing_seed(
            config_override, route_seed=seed, timing_seed=seed
        )

    @staticmethod
    def _apply_route_timing_seed(
        config_override: dict[str, Any], route_seed: int, timing_seed: int
    ) -> dict[str, Any]:
        """Apply route and timing seeds to config and nested flow configs.

        Route/timing seed pairs are part of the generic YAML eval surface so
        paper evals can reproduce exact route generation identities without
        ITSC-only mutation code.
        """
        cfg = deepcopy(config_override)
        route_seed_int = int(route_seed)
        timing_seed_int = int(timing_seed)

        cfg["iz_bus_gen_seed"] = route_seed_int
        cfg["iz_bus_route_seed"] = route_seed_int
        cfg["iz_bus_timing_seed"] = timing_seed_int
        cfg["iz_seed"] = timing_seed_int

        timetable_cfg = dict(cfg.get("iz_bus_timetable_config", {}))
        timetable_cfg["bus_seed"] = route_seed_int
        timetable_cfg["bus_route_seed"] = route_seed_int
        timetable_cfg["bus_timing_seed"] = timing_seed_int
        cfg["iz_bus_timetable_config"] = timetable_cfg

        flows = cfg.get("eval_bus_flows")
        if isinstance(flows, list):
            for flow in flows:
                if not isinstance(flow, dict):
                    continue
                nested = flow.get("sim_config_overrides")
                if isinstance(nested, dict):
                    nested_timetable_cfg = nested.get("iz_bus_timetable_config")
                    if isinstance(nested_timetable_cfg, dict):
                        updated = dict(nested_timetable_cfg)
                        updated["bus_seed"] = route_seed_int
                        updated["bus_route_seed"] = route_seed_int
                        updated["bus_timing_seed"] = timing_seed_int
                        nested["iz_bus_timetable_config"] = updated
                    nested["iz_bus_gen_seed"] = route_seed_int
                    nested["iz_bus_route_seed"] = route_seed_int
                    nested["iz_bus_timing_seed"] = timing_seed_int
                    nested["iz_seed"] = timing_seed_int
                    continue
                flow["iz_bus_gen_seed"] = route_seed_int
                flow["iz_bus_route_seed"] = route_seed_int
                flow["iz_bus_timing_seed"] = timing_seed_int
                flow["iz_seed"] = timing_seed_int
        return cfg

    @staticmethod
    def _row_identity_key(row: dict[str, Any]) -> tuple[Any, ...]:
        """Return identity keys for grouping evaluation rows."""
        identity_keys = [
            "flow_name",
            "policy_name",
            "policy_source",
            "compact_section",
            "setpoint_idx",
            "weights",
            "model_path",
        ]
        return tuple((k, row.get(k)) for k in identity_keys)

    @classmethod
    def aggregate_seed_results(cls, seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate results from multiple seed runs by computing means."""
        grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for seed_result in seed_results:
            for row in seed_result.get("rows", []):
                grouped_rows[cls._row_identity_key(row)].append(row)

        aggregated: list[dict[str, Any]] = []
        for grouped in grouped_rows.values():
            base = {
                k: v
                for k, v in grouped[0].items()
                if not isinstance(v, (int, float, np.floating)) and not k.startswith("_")
            }

            setpoint_idx = grouped[0].get("setpoint_idx")
            if isinstance(setpoint_idx, (int, float, np.floating)):
                base["setpoint_idx"] = int(setpoint_idx)

            numeric_keys = sorted(
                {
                    k
                    for row in grouped
                    for k, v in row.items()
                    if isinstance(v, (int, float, np.floating))
                    and k
                    not in {
                        "setpoint_idx",
                        "seed",
                        "route_seed",
                        "timing_seed",
                        "_seed",
                        "_route_seed",
                        "_timing_seed",
                    }
                }
            )
            for key in numeric_keys:
                values = [
                    float(row[key])
                    for row in grouped
                    if isinstance(row.get(key), (int, float, np.floating))
                ]
                if len(values) == 0:
                    continue
                base[key] = float(np.mean(values))
            base["n_seed_runs"] = float(len(grouped))
            aggregated.append(base)

        return sorted(
            aggregated,
            key=lambda row: (
                str(row.get("compact_section", "")),
                int(row.get("setpoint_idx", 0)),
                str(row.get("flow_name", "")),
                str(row.get("policy_name", "")),
            ),
        )
