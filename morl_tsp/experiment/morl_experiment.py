# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

import csv
import functools
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import yaml  # type: ignore[import-untyped]
from gymnasium.vector import VectorEnv, VectorWrapper

import wandb
from morl_tsp import config
from morl_tsp.experiment.callbacks import log_to_wandb, push_episode_history, save_trajectory
from morl_tsp.experiment.experiment import experiment
from morl_tsp.experiment.utils import (
    ensure_env_spec_id,
    extract_env_history_from_candidate,
    filter_kwargs,
    hypervolume_2d_maximize,
    load_object,
    nondominated_points_minimize,
    pareto_hv_2d_minimize,
    pareto_nondominated_mask,
    to_reward_vector,
)
from morl_tsp.util.backend import LIBSUMO

try:
    from mo_gymnasium.wrappers.vector import (  # type: ignore
        MORecordEpisodeStatistics,
        MOSyncVectorEnv,
    )
except ImportError as exc:
    raise ImportError(
        "n_vec_envs > 1 for MORL requires mo-gymnasium vector wrappers. "
        "Install with: pip install mo-gymnasium"
    ) from exc


class train_morl_experiment(experiment):
    """
    Dedicated MORL experiment runner that keeps vector rewards from SumoEnvironment.

    Required config:
    - `morl_algorithm`: import path for algorithm class/function
      (e.g. "package.module:ClassName")

    Optional config:
    - `morl_algorithm_kwargs`: kwargs passed to algorithm constructor
    - `morl_train_method`: method to call on algorithm ("train" or "learn"), default tries train then learn
    - `morl_train_kwargs`: kwargs passed to the selected train method
    - `morl_enable_algorithm_logging`: if True, keep morl-baselines internal W&B logs. Default True.
    - `morl_checkpoint_freq_steps`: periodic checkpoint interval in train steps (0 disables)
    - `morl_checkpoint_upload_to_wandb`: upload periodic checkpoints as W&B artifacts
    - `morl_checkpoint_select_best_by_hv`: select best checkpoint by Pareto HV score (default True)
    - `morl_checkpoint_hv_metric`: pareto_hv | pareto_hv_nondom (default pareto_hv_nondom)
    - `morl_checkpoint_hv_episode_window`: number of recent episodes to score (default 32)
    - `morl_checkpoint_hv_ref_point`: explicit HV reference point [r0, r1] (optional)
    - `morl_checkpoint_hv_nondominated_bonus`: nondominated bonus coeff for pareto_hv_nondom (default 0.01)
    """

    def __init__(
        self,
        name: str = "morl_experiment",
        group: str = "morl",
        experiment_type: Literal["intersection_zoo", None] = None,
        base_config: dict | None = None,
        config_overwrite: dict | None = None,
        callbacks: Sequence[Literal["wandb", "optuna_prune"]] | None = None,
        hyperparameters: dict | None = None,
        name_suffix: str = "",
        wandb_tags: list[str] | None = None,
    ) -> None:

        morl_hyperparameters: dict[str, Any] = {} if hyperparameters is None else dict(hyperparameters)

        super().__init__(
            name=name,
            group=group,
            experiment_type=experiment_type,
            base_config=base_config,
            config_overwrite=config_overwrite,
            callbacks=callbacks,
            hyperparameters=hyperparameters,
            name_suffix=name_suffix,
            wandb_tags=wandb_tags,
        )
        assert "morl_algorithm" in self.sim_config, "Missing required config key 'morl_algorithm'."

        morl_defaults = config.EXP_MORL_DEFAULTS

        # ---------------------------------------------------------------------
        # MORL configuration
        # ---------------------------------------------------------------------

        self.morl_algorithm_path: str = self.sim_config["morl_algorithm"]
        self.morl_algorithm_kwargs: dict[str, Any] = dict(self.sim_config.get("morl_algorithm_kwargs", {}))
        self.morl_algorithm_kwargs.update(morl_hyperparameters)
        self.morl_algorithm_kwargs.setdefault("project_name", "morl_tsp")
        self.morl_algorithm_kwargs.setdefault(
            "experiment_name", str(self.sim_config.get("experiment_name", self.name))
        )
        self.morl_train_method: str | None = self.sim_config.get("morl_train_method")
        self.morl_train_kwargs: dict[str, Any] = self.sim_config.get("morl_train_kwargs", {})
        self.n_vec_envs: int = max(1, int(self.sim_config.get("n_vec_envs", 1)))
        self.morl_enable_algorithm_logging: bool = bool(self.sim_config.get("morl_enable_algorithm_logging", morl_defaults["morl_enable_algorithm_logging"]))
        self.morl_save_model: bool = bool(self.sim_config.get("morl_save_model", morl_defaults["morl_save_model"]))
        self.morl_checkpoint_freq_steps: int = max(0, int(self.sim_config.get("morl_checkpoint_freq_steps", morl_defaults["morl_checkpoint_freq_steps"])))
        self.morl_checkpoint_upload_to_wandb: bool = bool(self.sim_config.get("morl_checkpoint_upload_to_wandb", morl_defaults["morl_checkpoint_upload_to_wandb"]))
        self.morl_checkpoint_select_best_by_hv: bool = bool(self.sim_config.get("morl_checkpoint_select_best_by_hv", morl_defaults["morl_checkpoint_select_best_by_hv"]))
        hv_metric = str(self.sim_config.get("morl_checkpoint_hv_metric", morl_defaults["morl_checkpoint_hv_metric"])).strip().lower()
        if hv_metric not in {"pareto_hv", "pareto_hypervolume", "pareto_hv_nondom"}:
            warnings.warn(
                f"Unsupported morl_checkpoint_hv_metric='{hv_metric}'. Falling back to 'pareto_hv_nondom'.",
                stacklevel=2,
            )
            hv_metric = "pareto_hv_nondom"
        self.morl_checkpoint_hv_metric: str = hv_metric
        self.morl_checkpoint_hv_episode_window: int = max(0, int(self.sim_config.get("morl_checkpoint_hv_episode_window", self.sim_config.get("hpt_pareto_episode_window", morl_defaults["morl_checkpoint_hv_episode_window"]))))
        self.morl_checkpoint_hv_nondominated_bonus: float = float(self.sim_config.get("morl_checkpoint_hv_nondominated_bonus", self.sim_config.get("hpt_pareto_nondominated_bonus", morl_defaults["morl_checkpoint_hv_nondominated_bonus"])))
        configured_ref = self.sim_config.get("morl_checkpoint_hv_ref_point", self.sim_config.get("hpt_pareto_ref_point", self.sim_config.get("morl_ref_point")))
        self.morl_checkpoint_hv_ref_point: list[float] | tuple[float, ...] | None = configured_ref if isinstance(configured_ref, (list, tuple)) else None
        self.morl_weight_sampling_bus_schedule: list[dict[str, float | int | str | None]] = self._parse_weight_sampling_schedule(self.sim_config.get("morl_weight_sampling_bus_schedule"))
        self._next_checkpoint_step: int | None = self.morl_checkpoint_freq_steps if self.morl_checkpoint_freq_steps > 0 else None
        self._checkpoint_replay_buffer: Any | None = None
        self._checkpoint_original_add: Any | None = None
        self._checkpoint_wrapped_add: Any | None = None
        self._checkpoint_history_offset: int = 0
        self._checkpoint_episode_reward_sum: dict[int, np.ndarray] = {}
        self._checkpoint_episode_reward_count: dict[int, int] = {}
        self._best_checkpoint_hv_score: float | None = None
        self._best_checkpoint_hv_step: int | None = None
        self._best_checkpoint_hv_path: str | None = None
        self._best_checkpoint_reason: str | None = None
        self.morl_periodic_eval_enabled: bool = bool(self.sim_config.get("morl_periodic_eval", morl_defaults["morl_periodic_eval"]))
        self.morl_periodic_eval_every_steps: int = max(0, int(self.sim_config.get("morl_periodic_eval_every_steps", morl_defaults["morl_periodic_eval_every_steps"])))
        periodic_seeds_raw = self.sim_config.get("morl_periodic_eval_bus_generation_seeds", [int(self.sim_config.get("iz_bus_gen_seed", self.sim_config.get("iz_seed", 420)))])
        if isinstance(periodic_seeds_raw, list) and len(periodic_seeds_raw) > 0:
            self.morl_periodic_eval_bus_generation_seeds = [int(seed) for seed in periodic_seeds_raw]
        else:
            self.morl_periodic_eval_bus_generation_seeds = [int(self.sim_config.get("iz_bus_gen_seed", self.sim_config.get("iz_seed", 420)))]
        self.morl_periodic_eval_seed_workers: int = max(1, int(self.sim_config.get("morl_periodic_eval_seed_workers", min(len(self.morl_periodic_eval_bus_generation_seeds), max(1, int(os.cpu_count() or 1))))))
        self.morl_periodic_eval_episodes: int = max(1, int(self.sim_config.get("morl_periodic_eval_episodes", morl_defaults["morl_periodic_eval_episodes"])))
        self.morl_periodic_eval_weights: list[list[float]] = self._resolve_periodic_eval_weights(self.sim_config.get("morl_periodic_eval_weights"))
        self.morl_periodic_eval_use_best_hv_checkpoint: bool = bool(self.sim_config.get("morl_periodic_eval_use_best_hv_checkpoint", morl_defaults["morl_periodic_eval_use_best_hv_checkpoint"]))
        self.morl_periodic_eval_eval_log_on_step: bool = bool(self.sim_config.get("morl_periodic_eval_eval_log_on_step", morl_defaults["morl_periodic_eval_eval_log_on_step"]))
        self.morl_periodic_eval_on_final: bool = bool(self.sim_config.get("morl_periodic_eval_on_final", morl_defaults["morl_periodic_eval_on_final"]))
        self.morl_periodic_eval_reuse_training_scenarios: bool = bool(self.sim_config.get("morl_periodic_eval_reuse_training_scenarios", morl_defaults["morl_periodic_eval_reuse_training_scenarios"]))
        self._next_periodic_eval_step: int | None = self.morl_periodic_eval_every_steps if self.morl_periodic_eval_every_steps > 0 else None

        if not self.morl_save_model:
            self.morl_checkpoint_select_best_by_hv = False

        if (
            self.morl_periodic_eval_enabled
            and self.morl_periodic_eval_every_steps <= 0
            and self.morl_checkpoint_freq_steps > 0
        ):
            self.morl_periodic_eval_every_steps = self.morl_checkpoint_freq_steps
            self._next_periodic_eval_step = self.morl_periodic_eval_every_steps

        if (
            self.morl_periodic_eval_enabled
            and self.morl_periodic_eval_every_steps > 0
            and self.morl_checkpoint_freq_steps <= 0
        ):
            self.morl_checkpoint_freq_steps = self.morl_periodic_eval_every_steps
            self._next_checkpoint_step = self.morl_checkpoint_freq_steps

        # ---------------------------------------------------------------------
        # Environment and model setup
        # ---------------------------------------------------------------------

        self.env_input = self._get_env_inputs()
        self.env_input["multi_objective"] = True
        self.env_input["add_reward_weights_to_obs"] = self.sim_config.get(
            "add_reward_weights_to_obs", False
        )  # Only enable for algorithms that augment obs with weights (e.g. Envelope, GPI-PD).
        self.env = self._build_train_env()

        if not self.morl_enable_algorithm_logging:
            self.morl_algorithm_kwargs["log"] = False

        self.model = self._build_model()

    # -------------------------------------------------------------------------
    # Environment building
    # -------------------------------------------------------------------------

    def _build_train_env(self) -> gym.Env:
        """No vector env"""
        if self.n_vec_envs <= 1:
            base_env = self._make_env_fn(rank=0)()
        else:
            """Vector env"""
            env_fns = [self._make_env_fn(i) for i in range(self.n_vec_envs)]
            base_env = MORecordEpisodeStatistics(MOSyncVectorEnv(env_fns))  # type: ignore

        log_every_n_steps = max(1, int(self.sim_config.get("morl_wandb_log_every_n_steps", config.EXP_MORL_DEFAULTS["morl_wandb_log_every_n_steps"])))
        save_trajectories = bool(self.sim_config.get("morl_save_trajectories", config.EXP_MORL_DEFAULTS["morl_save_trajectories"]))
        if "wandb" in self.callbacks:
            if self.n_vec_envs > 1:
                return MORLVecWandbLoggingWrapper(
                    base_env,
                    log_every_n_steps=log_every_n_steps,
                    save_trajectories=save_trajectories,
                )  # type: ignore
            return MORLWandbLoggingWrapper(
                base_env, log_every_n_steps=log_every_n_steps, save_trajectories=save_trajectories
            )  # type: ignore

        """No logging wrapper"""
        return base_env  # type: ignore

    def _make_env_fn(self, rank: int):
        def _init():
            env_kwargs = dict(self.env_input)
            if isinstance(env_kwargs.get("random_seed"), int):
                env_kwargs["random_seed"] = env_kwargs["random_seed"] + rank
            if self.n_vec_envs > 1:
                if env_kwargs.get("out_history_name") is not None:
                    env_kwargs["out_history_name"] = f"{env_kwargs['out_history_name']}_env{rank}"
            env = self._get_env(**env_kwargs)
            default_id = str(self.sim_config.get("experiment_name", "morl_tsp_morl_env"))
            ensure_env_spec_id(env, default_id)
            return env

        return _init

    def _build_eval_env(self) -> gym.Env:
        """Build a separate evaluation environment for algorithms that require one (e.g. MORLD, PCN).

        The eval env is a plain (non-vectorized, non-logging-wrapped) copy of
        the training environment so that algorithms can evaluate policies
        without interfering with the active training simulation.
        """
        eval_env_kwargs = dict(self.env_input)
        # Use a different seed to avoid identical episode sequences
        if isinstance(eval_env_kwargs.get("random_seed"), int):
            eval_env_kwargs["random_seed"] = eval_env_kwargs["random_seed"] + 9999
        # Do NOT set eval_mode=True here: eval_mode uses eval_start_time
        # (default 27000) which may exceed the route file duration, resulting
        # in zero vehicles and zero rewards.  The MORL eval_env should mirror
        # training conditions so that policy_evaluation_mo() sees realistic
        # traffic.
        eval_env_kwargs["eval_mode"] = False
        eval_env = self._get_env(**eval_env_kwargs)
        default_id = str(self.sim_config.get("experiment_name", "morl_tsp_morl_env"))
        ensure_env_spec_id(eval_env, default_id)
        return eval_env

    def _get_primary_env(self) -> gym.Env:
        base_env = self.env.unwrapped
        envs = getattr(base_env, "envs", None)
        if isinstance(envs, (list, tuple)) and len(envs) > 0:
            primary = envs[0]
            return primary.unwrapped if hasattr(primary, "unwrapped") else primary
        return base_env

    # -------------------------------------------------------------------------
    # Model building and training
    # -------------------------------------------------------------------------

    def _build_model(self) -> Any:
        try:
            model_class = load_object(self.morl_algorithm_path)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] == "morl_baselines":
                raise ImportError(
                    "morl_baselines is required for train_morl_experiment. Install with: "
                    "pip install morl-baselines mo-gymnasium"
                ) from exc
            raise
        ctor_kwargs = dict(self.morl_algorithm_kwargs)

        if self.n_vec_envs > 1 and getattr(model_class, "__name__", "") == "MORLD":
            raise ValueError(
                "MORLD from morl-baselines currently expects a single gym.Env. "
                "Set n_vec_envs=1 or choose a MORL algorithm that supports vectorized environments."
            )

        if getattr(model_class, "__name__", "") == "MORLD":
            policy_name = ctor_kwargs.get("policy_name")
            action_space = getattr(self.env, "single_action_space", self.env.action_space)
            is_discrete = isinstance(action_space, gym.spaces.Discrete)
            is_continuous = isinstance(action_space, gym.spaces.Box)

            if policy_name is None:
                if is_discrete:
                    ctor_kwargs["policy_name"] = "MOSACDiscrete"
                elif is_continuous:
                    ctor_kwargs["policy_name"] = "MOSAC"
            elif is_discrete and policy_name == "MOSAC":
                raise ValueError(
                    "MORLD policy mismatch: env.action_space is Discrete but policy_name='MOSAC' "
                    "(continuous-only). Set policy_name='MOSACDiscrete'."
                )
            elif is_continuous and policy_name == "MOSACDiscrete":
                raise ValueError(
                    "MORLD policy mismatch: env.action_space is Box but policy_name='MOSACDiscrete' "
                    "(discrete-only). Set policy_name='MOSAC'."
                )

        # Resolve callable kwargs specified as import paths (e.g. scalarization).
        ctor_sig = inspect.signature(model_class)
        for key, val in list(ctor_kwargs.items()):
            if isinstance(val, str) and ":" in val and key in ctor_sig.parameters:
                try:
                    ctor_kwargs[key] = load_object(val)
                except Exception:
                    pass  # leave as string if resolution fails

        # Common constructor names used by MORL libraries.
        for env_key in ("env", "train_env", "environment"):
            if env_key in ctor_sig.parameters and env_key not in ctor_kwargs:
                ctor_kwargs[env_key] = self.env

        ctor_kwargs = filter_kwargs(model_class, ctor_kwargs)
        return model_class(**ctor_kwargs)

    @staticmethod
    def _parse_weight_sampling_schedule(
        raw_schedule: Any,
    ) -> list[dict[str, float | int | str | None]]:
        if not isinstance(raw_schedule, list):
            return []

        schedule: list[dict[str, float | int | str | None]] = []
        for idx, phase in enumerate(raw_schedule):
            if not isinstance(phase, dict):
                warnings.warn(
                    f"Ignoring morl_weight_sampling_bus_schedule[{idx}] (expected dict).",
                    stacklevel=2,
                )
                continue

            min_bus = phase.get("min_bus_weight", phase.get("min", None))
            max_bus = phase.get("max_bus_weight", phase.get("max", None))
            until_step = phase.get("until_step")

            if min_bus is None or max_bus is None:
                warnings.warn(
                    f"Ignoring morl_weight_sampling_bus_schedule[{idx}] due to missing "
                    "'min_bus_weight'/'max_bus_weight'.",
                    stacklevel=2,
                )
                continue

            try:
                min_bus_f = float(min_bus)
                max_bus_f = float(max_bus)
            except Exception:
                warnings.warn(
                    f"Ignoring morl_weight_sampling_bus_schedule[{idx}] due to non-numeric bounds.",
                    stacklevel=2,
                )
                continue

            min_bus_f = max(0.0, min(1.0, min_bus_f))
            max_bus_f = max(0.0, min(1.0, max_bus_f))
            if max_bus_f < min_bus_f:
                min_bus_f, max_bus_f = max_bus_f, min_bus_f

            until_step_i: int | None
            if until_step is None:
                until_step_i = None
            else:
                try:
                    until_step_i = max(0, int(until_step))
                except Exception:
                    warnings.warn(
                        f"Ignoring invalid until_step in morl_weight_sampling_bus_schedule[{idx}]: {until_step}",
                        stacklevel=2,
                    )
                    until_step_i = None

            sampling_raw = (
                str(phase.get("sampling", phase.get("distribution", "uniform"))).strip().lower()
            )
            if sampling_raw not in {"uniform", "beta"}:
                warnings.warn(
                    f"Unknown sampling mode in morl_weight_sampling_bus_schedule[{idx}]: "
                    f"{sampling_raw}. Falling back to 'uniform'.",
                    stacklevel=2,
                )
                sampling_raw = "uniform"

            beta_alpha = phase.get("beta_alpha")
            beta_beta = phase.get("beta_beta")
            beta_alpha_f: float | None = None
            beta_beta_f: float | None = None
            if beta_alpha is not None:
                try:
                    beta_alpha_f = float(beta_alpha)
                except Exception:
                    warnings.warn(
                        f"Ignoring non-numeric beta_alpha in morl_weight_sampling_bus_schedule[{idx}]: {beta_alpha}",
                        stacklevel=2,
                    )
            if beta_beta is not None:
                try:
                    beta_beta_f = float(beta_beta)
                except Exception:
                    warnings.warn(
                        f"Ignoring non-numeric beta_beta in morl_weight_sampling_bus_schedule[{idx}]: {beta_beta}",
                        stacklevel=2,
                    )

            if sampling_raw == "beta":
                if (
                    beta_alpha_f is None
                    or beta_beta_f is None
                    or beta_alpha_f <= 0.0
                    or beta_beta_f <= 0.0
                ):
                    warnings.warn(
                        "morl_weight_sampling_bus_schedule phase requested 'beta' sampling but "
                        "beta_alpha/beta_beta are missing or invalid. Falling back to 'uniform'.",
                        stacklevel=2,
                    )
                    sampling_raw = "uniform"

            phase_dict: dict[str, float | int | str | None] = {
                "min_bus_weight": float(min_bus_f),
                "max_bus_weight": float(max_bus_f),
                "until_step": until_step_i,
                "sampling": sampling_raw,
            }
            if beta_alpha_f is not None:
                phase_dict["beta_alpha"] = float(beta_alpha_f)
            if beta_beta_f is not None:
                phase_dict["beta_beta"] = float(beta_beta_f)

            schedule.append(phase_dict)

        # Deterministic ordering by step boundary; terminal phase (None) goes last.
        schedule.sort(key=lambda p: (p["until_step"] is None, int(p["until_step"] or 0)))
        return schedule

    def _resolve_weight_sampling_phase_for_step(
        self, step: int
    ) -> dict[str, float | int | str | None]:
        if len(self.morl_weight_sampling_bus_schedule) == 0:
            return {
                "min_bus_weight": 0.0,
                "max_bus_weight": 1.0,
                "until_step": None,
                "sampling": "uniform",
            }

        for phase in self.morl_weight_sampling_bus_schedule:
            until_step = phase.get("until_step")
            if until_step is None or step <= int(until_step):
                return phase

        return self.morl_weight_sampling_bus_schedule[-1]

    def _resolve_bus_weight_bounds_for_step(self, step: int) -> tuple[float, float]:
        phase = self._resolve_weight_sampling_phase_for_step(step)
        min_bus = self._to_float_or_none(phase.get("min_bus_weight"))
        max_bus = self._to_float_or_none(phase.get("max_bus_weight"))
        if min_bus is None or max_bus is None:
            raise ValueError(f"Invalid staged MORL weight bounds: {phase}")
        return min_bus, max_bus

    def _configure_staged_weight_sampling(self, train_kwargs: dict[str, Any]):
        if len(self.morl_weight_sampling_bus_schedule) == 0:
            return None

        if train_kwargs.get("weight") is not None:
            warnings.warn(
                "morl_weight_sampling_bus_schedule is ignored because train() received a fixed 'weight' argument.",
                stacklevel=2,
            )
            return None

        model_type_name = type(self.model).__name__
        if model_type_name != "Envelope":
            warnings.warn(
                "morl_weight_sampling_bus_schedule currently supports Envelope only. "
                f"Found model type: {model_type_name}. Schedule disabled.",
                stacklevel=2,
            )
            return None

        model_module = inspect.getmodule(type(self.model))
        if model_module is None or not hasattr(model_module, "random_weights"):
            warnings.warn(
                "Could not locate Envelope module random_weights; staged weight sampling disabled.",
                stacklevel=2,
            )
            return None

        original_random_weights = model_module.random_weights  # type: ignore[attr-defined]
        print(
            "[morl] Enabled staged Envelope weight sampling schedule: "
            f"{self.morl_weight_sampling_bus_schedule}"
        )

        def _patched_random_weights(*args: Any, **kwargs: Any):
            reward_dim_raw: Any | None = None
            if "reward_dim" in kwargs:
                reward_dim_raw = kwargs["reward_dim"]
            elif "dim" in kwargs:
                reward_dim_raw = kwargs["dim"]
            elif len(args) > 0:
                reward_dim_raw = args[0]

            if reward_dim_raw is None:
                return original_random_weights(*args, **kwargs)

            n_raw: Any = kwargs.get("n", args[1] if len(args) > 1 else 1)
            rng = kwargs.get("rng", args[3] if len(args) > 3 else None)

            try:
                reward_dim_i = int(reward_dim_raw)
                n_i = max(1, int(n_raw))
            except Exception:
                return original_random_weights(*args, **kwargs)

            # Keep original behavior for non-2D objectives.
            if reward_dim_i != 2:
                return original_random_weights(*args, **kwargs)

            current_step = int(getattr(self.model, "global_step", 0))
            phase = self._resolve_weight_sampling_phase_for_step(current_step)
            min_bus = self._to_float_or_none(phase.get("min_bus_weight"))
            max_bus = self._to_float_or_none(phase.get("max_bus_weight"))
            if min_bus is None or max_bus is None:
                return original_random_weights(*args, **kwargs)
            local_rng = rng if rng is not None else np.random.default_rng()
            span = max_bus - min_bus
            sampling = str(phase.get("sampling", "uniform")).strip().lower()
            if span <= 1e-12:
                bus_weights = np.full((n_i, 1), fill_value=min_bus, dtype=np.float64)
            elif sampling == "beta":
                beta_alpha = self._to_float_or_none(phase.get("beta_alpha", 2.0)) or 2.0
                beta_beta = self._to_float_or_none(phase.get("beta_beta", 2.0)) or 2.0
                raw = local_rng.beta(beta_alpha, beta_beta, size=(n_i, 1))
                bus_weights = min_bus + span * raw
            else:
                bus_weights = local_rng.uniform(min_bus, max_bus, size=(n_i, 1))
            car_weights = 1.0 - bus_weights
            weights = np.concatenate([bus_weights, car_weights], axis=1).astype(np.float64)
            if n_i == 1:
                return weights[0]
            return weights

        model_module.random_weights = _patched_random_weights  # type: ignore[attr-defined]

        def _cleanup():
            try:
                model_module.random_weights = original_random_weights  # type: ignore[attr-defined]
            except Exception:
                pass

        return _cleanup

    def _resolve_periodic_eval_weights(self, raw_weights: Any) -> list[list[float]]:
        if not isinstance(raw_weights, list) or len(raw_weights) == 0:
            return [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]

        normalized: list[list[float]] = []
        for idx, weight in enumerate(raw_weights):
            if not isinstance(weight, (list, tuple)):
                warnings.warn(
                    f"Ignoring morl_periodic_eval_weights[{idx}] (expected list/tuple, got {type(weight)}).",
                    stacklevel=2,
                )
                continue
            w = np.asarray(weight, dtype=np.float64).flatten()
            if w.shape[0] == 0:
                continue
            if np.any(w < 0.0):
                warnings.warn(
                    f"Ignoring morl_periodic_eval_weights[{idx}] due to negative entries.",
                    stacklevel=2,
                )
                continue
            weight_sum = float(np.sum(w))
            if weight_sum <= 0.0:
                warnings.warn(
                    f"Ignoring morl_periodic_eval_weights[{idx}] because sum is 0.",
                    stacklevel=2,
                )
                continue
            normalized.append((w / weight_sum).tolist())

        if len(normalized) == 0:
            return [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        return normalized

    def _should_run_periodic_eval(self, step: int) -> bool:
        if not self.morl_periodic_eval_enabled:
            return False
        if self._next_periodic_eval_step is None:
            return False
        if step < self._next_periodic_eval_step:
            return False
        self._next_periodic_eval_step += self.morl_periodic_eval_every_steps
        return True

    def _select_checkpoint_for_periodic_eval(self, saved_paths: list[str]) -> str | None:
        if self.morl_periodic_eval_use_best_hv_checkpoint and self._best_checkpoint_hv_path:
            if os.path.exists(self._best_checkpoint_hv_path):
                return self._best_checkpoint_hv_path
        return self._select_preferred_checkpoint_path(saved_paths)

    def _build_periodic_eval_payload(
        self, checkpoint_path: str, seed: int, step: int
    ) -> dict[str, Any]:
        eval_log_dir = os.path.join(
            self.sim_config["log_dir"], "periodic_eval", f"step_{step}", f"seed_{seed}"
        )
        flow_overrides: dict[str, Any] = {}
        if not self.morl_periodic_eval_reuse_training_scenarios:
            timetable_cfg = dict(self.sim_config.get("iz_bus_timetable_config", {}))
            timetable_cfg["bus_seed"] = int(seed)
            flow_overrides = {
                "iz_bus_gen_seed": int(seed),
                "iz_bus_timetable_config": timetable_cfg,
            }

        base_config = dict(self.sim_config)
        base_config["wandb"] = False
        base_config["log_dir"] = eval_log_dir
        # Keep periodic eval deterministic per seed while preserving train settings.
        base_config["random_seed"] = int(seed)
        base_config["eval_episodes"] = int(self.morl_periodic_eval_episodes)
        base_config["eval_log_on_step"] = bool(self.morl_periodic_eval_eval_log_on_step)
        base_config["save_eval_trajectories"] = False
        base_config["model_path"] = checkpoint_path
        base_config["model_format"] = "morl_checkpoint"
        base_config["morl_eval_sources"] = ["model"]
        base_config["morl_eval_weights"] = self.morl_periodic_eval_weights
        base_config["morl_eval_limit"] = len(self.morl_periodic_eval_weights)
        base_config["reuse_base_scenarios"] = bool(self.morl_periodic_eval_reuse_training_scenarios)
        morl_algorithm_kwargs = dict(base_config.get("morl_algorithm_kwargs", {}))
        morl_algorithm_kwargs["log"] = False
        base_config["morl_algorithm_kwargs"] = morl_algorithm_kwargs
        base_config["eval_bus_flows"] = [
            {
                "name": f"periodic_seed_{seed}",
                "sim_config_overrides": flow_overrides,
            }
        ]

        # Remove train-only keys from the eval payload to keep it lean.
        for key in [
            "morl_periodic_eval",
            "morl_periodic_eval_every_steps",
            "morl_periodic_eval_episodes",
            "morl_periodic_eval_bus_generation_seeds",
            "morl_periodic_eval_seed_workers",
            "morl_periodic_eval_weights",
            "morl_periodic_eval_use_best_hv_checkpoint",
            "morl_periodic_eval_eval_log_on_step",
            "morl_periodic_eval_on_final",
            "morl_periodic_eval_reuse_training_scenarios",
            "morl_train_kwargs",
            "morl_train_method",
            "morl_checkpoint_freq_steps",
            "morl_checkpoint_upload_to_wandb",
        ]:
            base_config.pop(key, None)

        return {
            "seed": int(seed),
            "step": int(step),
            "init_kwargs": {
                "name": f"{self.sim_config.get('experiment_name', 'morl')}_periodic_eval_step{step}_seed{seed}",
                "group": self.group,
                "experiment_type": self.experiment_type,
                "base_config": base_config,
                "config_overwrite": {},
                "callbacks": [],
                "stable_baselines_model": "PPO",
                "wandb_tags": ["periodic_eval", f"seed_{seed}", f"step_{step}"],
            },
        }

    def _build_periodic_eval_yaml_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        init_kwargs = dict(payload.get("init_kwargs", {}))
        cfg_override = dict(init_kwargs.get("config_overwrite", {}))
        cfg_override["name"] = str(
            init_kwargs.get("name", cfg_override.get("name", "periodic_eval"))
        )

        return {
            "group": str(init_kwargs.get("group", self.group)),
            "callbacks": list(init_kwargs.get("callbacks", [])),
            "experiment_type": init_kwargs.get("experiment_type", self.experiment_type),
            "workers": 1,
            "n_parallel": 1,
            "stable_baselines_model": init_kwargs.get("stable_baselines_model", "PPO"),
            "config_overwrite": [cfg_override],
            "hyperparameters": {},
            "base_config": dict(init_kwargs.get("base_config", {})),
        }

    def _run_periodic_eval_subprocess(self, payload: dict[str, Any]) -> dict[str, Any]:
        seed = int(payload["seed"])
        step = int(payload["step"])
        yaml_data = self._build_periodic_eval_yaml_config(payload)

        with tempfile.TemporaryDirectory(
            prefix=f"morl_periodic_eval_step{step}_seed{seed}_"
        ) as tmp_dir:
            yaml_path = os.path.join(tmp_dir, "periodic_eval.yaml")
            output_json_path = os.path.join(tmp_dir, "periodic_eval_result.json")
            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(yaml_data, file, sort_keys=False)

            cmd = [
                sys.executable,
                f"{config.ROOT_PATH}/morl_tsp/experiment/run_morl_eval_experiment.py",
                "--yaml_path",
                yaml_path,
                "--experiment_idx",
                "0",
                "--output_json",
                output_json_path,
            ]

            env = os.environ.copy()
            env.setdefault("MORL_TSP_ROOT_PATH", config.ROOT_PATH)
            backend_override = self.sim_config.get("morl_periodic_eval_traci_backend")
            if isinstance(backend_override, str) and len(backend_override.strip()) > 0:
                backend = backend_override.strip().lower()
            elif (
                isinstance(env.get("MORL_TSP_TRACI_BACKEND"), str)
                and len(str(env["MORL_TSP_TRACI_BACKEND"]).strip()) > 0
            ):
                backend = str(env["MORL_TSP_TRACI_BACKEND"]).strip().lower()
            else:
                use_libsumo = bool(self.sim_config.get("morl_periodic_eval_use_libsumo", False))
                backend = "libsumo" if use_libsumo else "libtraci"
            env["MORL_TSP_TRACI_BACKEND"] = backend
            env["MORL_TSP_USE_LIBSUMO"] = "1" if backend == "libsumo" else "0"
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                details = stderr if len(stderr) > 0 else stdout
                raise RuntimeError(
                    f"Periodic eval subprocess failed for step={step}, seed={seed} (exit={exc.returncode}).\n{details}"
                ) from exc

            with open(output_json_path, encoding="utf-8") as file:
                result_payload = json.load(file)
            rows = result_payload.get("rows", [])
            if not isinstance(rows, list):
                rows = []

        return {
            "seed": seed,
            "step": step,
            "rows": rows,
            "summary": self._summarize_periodic_eval_rows(rows),
        }

    def _persist_periodic_eval_outputs(
        self,
        step: int,
        seed_results: list[dict[str, Any]],
    ) -> tuple[str | None, str | None, int]:
        step_dir = Path(self.sim_config["log_dir"]) / "periodic_eval" / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        flattened_rows: list[dict[str, Any]] = []
        for seed_result in seed_results:
            seed = int(seed_result.get("seed", -1))
            rows = seed_result.get("rows", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                enriched = dict(row)
                enriched["periodic_eval_seed"] = seed
                enriched["periodic_eval_step"] = int(step)
                flattened_rows.append(enriched)

        csv_path: str | None = None
        if len(flattened_rows) > 0:
            csv_path = str(step_dir / "rows.csv")
            preferred = [
                "periodic_eval_step",
                "periodic_eval_seed",
                "flow_name",
                "policy_name",
                "setpoint_idx",
                "weights",
                "Bus_crossing_time",
                "Car_crossing_time",
                "Bus_faster_crossing_time_gap",
                "episode_return",
            ]
            all_columns = sorted({key for row in flattened_rows for key in row.keys()})
            fieldnames = preferred + [key for key in all_columns if key not in preferred]
            with open(csv_path, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in flattened_rows:
                    writer.writerow(row)

        plot_path = self._save_periodic_eval_plot(step_dir=step_dir, rows=flattened_rows)
        return csv_path, plot_path, len(flattened_rows)

    def _save_periodic_eval_plot(self, step_dir: Path, rows: list[dict[str, Any]]) -> str | None:
        points: list[tuple[float, float, float | None, int]] = []
        for row in rows:
            bus = self._to_float_or_none(row.get("Bus_crossing_time"))
            car = self._to_float_or_none(row.get("Car_crossing_time"))
            if bus is None or car is None:
                continue
            weight_bus = self._extract_weight_bus(row)
            seed = int(row.get("periodic_eval_seed", -1))
            points.append((bus, car, weight_bus, seed))

        if len(points) == 0:
            return None

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            warnings.warn(
                f"Skipping periodic eval Pareto plot (matplotlib unavailable): {exc}",
                stacklevel=2,
            )
            return None

        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)

        cars = np.asarray([car for _, car, _, _ in points], dtype=np.float64)
        buses = np.asarray([bus for bus, _, _, _ in points], dtype=np.float64)
        weight_values = np.asarray(
            [weight if weight is not None else np.nan for _, _, weight, _ in points],
            dtype=np.float64,
        )
        use_weight_coloring = np.isfinite(weight_values).any()

        if use_weight_coloring:
            scatter = ax.scatter(
                cars,
                buses,
                c=weight_values,
                cmap="viridis",
                s=34,
                alpha=0.9,
                edgecolors="none",
                label="Periodic eval points",
            )
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label("Bus weight (w_bus)")
        else:
            for seed in sorted({seed for _, _, _, seed in points}):
                seed_points = [(bus, car) for bus, car, _, row_seed in points if row_seed == seed]
                ax.scatter(
                    [car for _, car in seed_points],
                    [bus for bus, _ in seed_points],
                    s=34,
                    alpha=0.9,
                    edgecolors="none",
                    label=f"Seed {seed}",
                )

        nondom = nondominated_points_minimize([(bus, car) for bus, car, _, _ in points])
        if len(nondom) >= 2:
            front = sorted([(car, bus) for bus, car in nondom], key=lambda item: item[0])
            ax.plot(
                [car for car, _ in front],
                [bus for _, bus in front],
                color="#0f172a",
                linewidth=1.3,
                alpha=0.85,
                label="Nondominated front",
                zorder=3,
            )

        ax.set_title("MORL periodic eval Pareto front")
        ax.set_xlabel("Car crossing time")
        ax.set_ylabel("Bus crossing time")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

        plot_path = step_dir / "pareto_front_periodic_eval.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        return str(plot_path)

    def _log_periodic_eval_results(self, step: int, seed_results: list[dict[str, Any]]) -> None:
        if len(seed_results) == 0:
            return

        csv_path, plot_path, row_count = self._persist_periodic_eval_outputs(
            step=step,
            seed_results=seed_results,
        )

        if not self.sim_config.get("wandb", False):
            return

        seed_hv_values = [
            value
            for value in (
                self._to_float_or_none(result.get("summary", {}).get("pareto_hv_auto_ref"))
                for result in seed_results
            )
            if value is not None
        ]
        seed_episode_return_values = [
            value
            for value in (
                self._to_float_or_none(result.get("summary", {}).get("episode_return_mean"))
                for result in seed_results
            )
            if value is not None
        ]

        aggregated_points: list[tuple[float, float]] = []
        for result in seed_results:
            rows = result.get("rows", [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                bus = self._to_float_or_none(row.get("Bus_crossing_time"))
                car = self._to_float_or_none(row.get("Car_crossing_time"))
                if bus is None or car is None:
                    continue
                aggregated_points.append((bus, car))

        payload: dict[str, Any] = {
            "periodic_eval/checkpoint_step": float(step),
            "periodic_eval/seed_count": float(len(seed_results)),
            "periodic_eval/row_count": float(row_count),
        }
        if len(seed_hv_values) > 0:
            payload["periodic_eval/pareto_hv_auto_ref_mean_over_seeds"] = float(
                np.mean(seed_hv_values)
            )
            payload["periodic_eval/pareto_hv_auto_ref_min_over_seeds"] = float(
                np.min(seed_hv_values)
            )
        if len(seed_episode_return_values) > 0:
            payload["periodic_eval/episode_return_mean_over_seeds"] = float(
                np.mean(seed_episode_return_values)
            )
        if len(aggregated_points) > 0:
            payload["periodic_eval/pareto_hv_auto_ref_all_points"] = float(
                pareto_hv_2d_minimize(aggregated_points)
            )
            payload["periodic_eval/pareto_nondominated_count_all_points"] = float(
                len(nondominated_points_minimize(aggregated_points))
            )
        if plot_path is not None and os.path.exists(plot_path):
            payload["periodic_eval/pareto_front_plot"] = wandb.Image(plot_path)

        log_to_wandb(payload)
        try:
            if csv_path is not None and os.path.exists(csv_path):
                wandb.save(csv_path, policy="now")
            if plot_path is not None and os.path.exists(plot_path):
                wandb.save(plot_path, policy="now")
        except Exception as exc:
            warnings.warn(f"Failed to upload periodic eval artifacts to W&B: {exc}", stacklevel=2)

    def _run_periodic_eval_for_checkpoint(self, checkpoint_path: str, step: int) -> None:
        seeds = [int(seed) for seed in self.morl_periodic_eval_bus_generation_seeds]
        payloads = [
            self._build_periodic_eval_payload(checkpoint_path=checkpoint_path, seed=seed, step=step)
            for seed in seeds
        ]
        if len(payloads) == 0:
            return

        workers = max(1, min(self.morl_periodic_eval_seed_workers, len(payloads)))
        seed_results: list[dict[str, Any]] = []
        try:
            if workers == 1 or len(payloads) == 1:
                for payload in payloads:
                    seed_results.append(self._run_periodic_eval_subprocess(payload))
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(self._run_periodic_eval_subprocess, payload)
                        for payload in payloads
                    ]
                    for future in as_completed(futures):
                        seed_results.append(future.result())
        except Exception as exc:
            warnings.warn(
                f"Periodic MORL eval failed at step={step}: {exc}\n{traceback.format_exc()}",
                stacklevel=2,
            )
            return

        self._log_periodic_eval_results(step=step, seed_results=seed_results)

    """
    Main experiment loop
    """

    def run(self) -> None:
        original_close_wandb = None
        checkpoint_cleanup = None
        weight_sampling_cleanup = None
        eval_env = None
        try:
            if self.sim_config.get("wandb", False):
                try:
                    wandb.define_metric("periodic_eval/checkpoint_step")
                    wandb.define_metric(
                        "periodic_eval/*", step_metric="periodic_eval/checkpoint_step"
                    )
                except Exception:
                    pass

            train_fn = self._get_train_callable()
            train_kwargs = dict(self.morl_train_kwargs)
            self._configure_model_save_defaults()
            checkpoint_cleanup = self._configure_periodic_checkpointing()
            if "total_timesteps" in self.sim_config and "total_timesteps" not in train_kwargs:
                train_kwargs["total_timesteps"] = self.sim_config["total_timesteps"]

            # libsumo supports one active simulation per process. Passing a live
            # eval_env to morl-baselines train() can spawn a second SUMO env in
            # the same process and crash training. Force-disable it here.
            if LIBSUMO and train_kwargs.get("eval_env") is not None:
                warnings.warn(
                    "Disabling morl_train_kwargs.eval_env because LIBSUMO=True "
                    "and concurrent train/eval SUMO environments are not supported "
                    "in the same process.",
                    stacklevel=2,
                )
                train_kwargs["eval_env"] = None

            # Auto-build eval_env for algorithms that require it (e.g. MORLD, PCN)
            # when using a non-libsumo backend that supports multiple connections.
            train_sig = inspect.signature(train_fn)
            if (
                "eval_env" in train_sig.parameters
                and train_kwargs.get("eval_env") is None
                and not LIBSUMO
                and bool(
                    self.sim_config.get(
                        "morl_auto_build_eval_env",
                        config.EXP_MORL_DEFAULTS["morl_auto_build_eval_env"],
                    )
                )
            ):
                eval_env = self._build_eval_env()
                train_kwargs["eval_env"] = eval_env

            # Auto-inject ref_point from config for algorithms that need it
            # (e.g. MORLD, PCN require ref_point as a train kwarg).
            if "ref_point" in train_sig.parameters and "ref_point" not in train_kwargs:
                configured_ref = self.sim_config.get("morl_ref_point")
                if isinstance(configured_ref, (list, tuple)):
                    train_kwargs["ref_point"] = np.asarray(configured_ref, dtype=np.float64)

            # Keep the parent experiment run active until this method returns.
            if self.sim_config.get("wandb", False) and hasattr(self.model, "close_wandb"):
                original_close_wandb = self.model.close_wandb
                self.model.close_wandb = lambda: None

            # Some MORL algorithms (e.g. PCN) expect numpy arrays for
            # vector-valued kwargs and may call `.tolist()` internally.
            # YAML provides python lists, so normalize here.
            if "max_return" in train_kwargs and isinstance(train_kwargs["max_return"], list):
                train_kwargs["max_return"] = np.asarray(
                    train_kwargs["max_return"], dtype=np.float64
                )

            weight_sampling_cleanup = self._configure_staged_weight_sampling(
                train_kwargs=train_kwargs
            )
            train_kwargs = filter_kwargs(train_fn, train_kwargs)
            train_fn(**train_kwargs)

            if self.morl_save_model:
                if callable(checkpoint_cleanup):
                    checkpoint_cleanup()
                    checkpoint_cleanup = None
                final_paths = self._save_model_checkpoint(
                    upload_to_wandb=self.sim_config.get("wandb", False),
                    reason="final",
                )
                if self.morl_periodic_eval_enabled and self.morl_periodic_eval_on_final:
                    final_checkpoint = self._select_checkpoint_for_periodic_eval(final_paths)
                    if final_checkpoint is not None:
                        final_step = int(
                            getattr(
                                self.model, "global_step", self.sim_config.get("total_timesteps", 0)
                            )
                        )
                        self._run_periodic_eval_for_checkpoint(
                            checkpoint_path=final_checkpoint,
                            step=final_step,
                        )
        finally:
            if eval_env is not None:
                try:
                    eval_env.close()
                except Exception:
                    pass
            if callable(checkpoint_cleanup):
                try:
                    checkpoint_cleanup()
                except Exception:
                    pass
            if callable(weight_sampling_cleanup):
                try:
                    weight_sampling_cleanup()
                except Exception:
                    pass
            if original_close_wandb is not None:
                try:
                    self.model.close_wandb = original_close_wandb
                except Exception:
                    pass
            try:
                self.env.close()
            except Exception:
                pass
            try:
                self.env.cleanup()  # type: ignore
            except Exception:
                pass
            if self.sim_config.get("wandb", False) and self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass

    def _get_train_callable(self):
        if self.morl_train_method is not None:
            if not hasattr(self.model, self.morl_train_method):
                raise ValueError(
                    f"Configured morl_train_method='{self.morl_train_method}' "
                    f"not found on model of type {type(self.model).__name__}."
                )
            return getattr(self.model, self.morl_train_method)

        if hasattr(self.model, "train"):
            return self.model.train
        if hasattr(self.model, "learn"):
            return self.model.learn
        raise ValueError(
            f"Model of type {type(self.model).__name__} has neither 'train' nor 'learn' method."
        )

    def _save_model_checkpoint(
        self, upload_to_wandb: bool, reason: str = "checkpoint"
    ) -> list[str]:
        save_fn = getattr(self.model, "save", None)
        if not callable(save_fn):
            warnings.warn(
                f"Model type {type(self.model).__name__} has no save() method. "
                "Skipping MORL model artifact logging.",
                stacklevel=2,
            )
            return []

        model_dir = self._get_model_save_dir()
        os.makedirs(model_dir, exist_ok=True)

        exp_name = str(self.sim_config.get("experiment_name", "morl_model"))
        step = int(getattr(self.model, "global_step", self.sim_config.get("total_timesteps", 0)))
        filename = (
            f"{exp_name}_step{step}" if reason == "final" else f"{exp_name}_{reason}_step{step}"
        )
        model_exts = (".tar", ".pt", ".pth", ".zip")

        pre_existing: dict[str, tuple[float, int]] = {}
        for f in os.listdir(model_dir):
            if not f.endswith(model_exts):
                continue
            path = os.path.join(model_dir, f)
            try:
                pre_existing[path] = (os.path.getmtime(path), os.path.getsize(path))
            except OSError:
                continue

        save_sig = inspect.signature(save_fn)
        save_kwargs: dict[str, Any] = {}
        if "save_dir" in save_sig.parameters:
            save_kwargs["save_dir"] = model_dir
        if "filename" in save_sig.parameters:
            save_kwargs["filename"] = filename
        if "save_replay_buffer" in save_sig.parameters:
            save_kwargs["save_replay_buffer"] = False

        saved_paths: list[str] = []
        try:
            if len(save_kwargs) > 0:
                save_fn(**save_kwargs)
                for ext in model_exts:
                    p = os.path.join(model_dir, f"{filename}{ext}")
                    if os.path.exists(p):
                        saved_paths.append(p)
            else:
                direct_path = os.path.join(model_dir, f"{filename}.pt")
                save_fn(direct_path)
                if os.path.exists(direct_path):
                    saved_paths.append(direct_path)
        except Exception as exc:
            warnings.warn(f"Failed to save MORL model via save(): {exc}", stacklevel=2)
            return []

        if len(saved_paths) == 0:
            changed_paths: list[str] = []
            for f in os.listdir(model_dir):
                if not f.endswith(model_exts):
                    continue
                path = os.path.join(model_dir, f)
                try:
                    sig = (os.path.getmtime(path), os.path.getsize(path))
                except OSError:
                    continue
                if pre_existing.get(path) != sig:
                    changed_paths.append(path)

            if len(changed_paths) > 0:
                saved_paths = changed_paths
            else:
                # Last-resort fallback to any model file in model_dir.
                for f in os.listdir(model_dir):
                    if f.endswith(model_exts):
                        saved_paths.append(os.path.join(model_dir, f))

            if len(saved_paths) == 0:
                warnings.warn(
                    f"save() completed but no model files were found in {model_dir}. "
                    "Skipping W&B model artifact logging.",
                    stacklevel=2,
                )
                return []

        unique_paths = self._preserve_versioned_checkpoint_files(saved_paths, model_dir, filename)
        print(f"Saved MORL model ({reason}) with {len(unique_paths)} file(s): {unique_paths}")

        checkpoint_hv_record = self._score_checkpoint_and_update_best_alias(
            model_dir=model_dir,
            exp_name=exp_name,
            step=step,
            reason=reason,
            checkpoint_paths=unique_paths,
            upload_to_wandb=upload_to_wandb,
        )

        if self.sim_config.get("wandb", False):
            payload: dict[str, Any] = {
                f"model/{reason}_save_step": step,
                f"model/{reason}_num_files": len(unique_paths),
            }
            if checkpoint_hv_record is not None:
                payload.update(
                    {
                        f"model/{reason}_hv_score": float(checkpoint_hv_record["pareto_score"]),
                        f"model/{reason}_hv": float(checkpoint_hv_record["pareto_hv"]),
                        f"model/{reason}_hv_nondominated_points": float(
                            checkpoint_hv_record["pareto_nondominated_points"]
                        ),
                        f"model/{reason}_hv_episode_count": float(
                            checkpoint_hv_record["pareto_episode_count"]
                        ),
                        "model/checkpoint_hv_score": float(checkpoint_hv_record["pareto_score"]),
                        "model/checkpoint_hv": float(checkpoint_hv_record["pareto_hv"]),
                        "model/checkpoint_hv_step": float(step),
                    }
                )
                if bool(checkpoint_hv_record.get("is_new_best", False)):
                    payload.update(
                        {
                            "model/best_hv_score": float(checkpoint_hv_record["pareto_score"]),
                            "model/best_hv_step": float(step),
                        }
                    )
            log_to_wandb(payload)

        if not upload_to_wandb:
            return unique_paths

        try:
            artifact = wandb.Artifact(
                name=f"{exp_name}-model-{reason}-{step}",
                type="model",
            )
            for path in unique_paths:
                artifact.add_file(path)
            wandb.log_artifact(artifact)
        except Exception as exc:
            warnings.warn(
                f"Model saved locally but failed to upload W&B artifact: {exc}",
                stacklevel=2,
            )
        return unique_paths

    def _score_checkpoint_and_update_best_alias(
        self,
        model_dir: str,
        exp_name: str,
        step: int,
        reason: str,
        checkpoint_paths: list[str],
        upload_to_wandb: bool,
    ) -> dict[str, Any] | None:
        if not self.morl_checkpoint_select_best_by_hv:
            return None

        checkpoint_score = self._compute_checkpoint_hv_score()
        if checkpoint_score is None:
            return None

        record: dict[str, Any] = {
            "step": int(step),
            "reason": str(reason),
            **checkpoint_score,
        }
        score = float(record["pareto_score"])

        is_new_best = (
            self._best_checkpoint_hv_score is None
            or score > self._best_checkpoint_hv_score + 1e-12
            or (
                abs(score - self._best_checkpoint_hv_score) <= 1e-12
                and (self._best_checkpoint_hv_step is None or step >= self._best_checkpoint_hv_step)
            )
        )
        record["is_new_best"] = bool(is_new_best)

        if is_new_best:
            best_alias_path = self._update_best_checkpoint_alias(
                model_dir=model_dir,
                exp_name=exp_name,
                checkpoint_paths=checkpoint_paths,
                step=step,
                reason=reason,
                score=score,
                score_details=checkpoint_score,
                upload_to_wandb=upload_to_wandb,
            )
            if best_alias_path is not None:
                self._best_checkpoint_hv_score = score
                self._best_checkpoint_hv_step = int(step)
                self._best_checkpoint_hv_path = best_alias_path
                self._best_checkpoint_reason = str(reason)
                record["best_checkpoint_path"] = best_alias_path

        self._append_checkpoint_hv_record(
            model_dir=model_dir,
            exp_name=exp_name,
            record=record,
        )
        return record

    def _compute_checkpoint_hv_score(self) -> dict[str, float | int | str] | None:
        self._update_checkpoint_reward_cache()

        episode_ids = sorted(self._checkpoint_episode_reward_sum.keys())
        if len(episode_ids) == 0:
            return None

        episode_mean_vectors: list[np.ndarray] = []
        for episode_id in episode_ids:
            vector_sum = self._checkpoint_episode_reward_sum.get(int(episode_id))
            vector_count = int(self._checkpoint_episode_reward_count.get(int(episode_id), 0))
            if vector_sum is None or vector_count <= 0:
                continue
            episode_mean_vectors.append(vector_sum / float(vector_count))

        if len(episode_mean_vectors) == 0:
            return None

        reward_dim = min(int(v.shape[0]) for v in episode_mean_vectors)
        if reward_dim <= 0:
            return None
        episode_vectors = np.stack([v[:reward_dim] for v in episode_mean_vectors], axis=0)

        if (
            self.morl_checkpoint_hv_episode_window > 0
            and episode_vectors.shape[0] > self.morl_checkpoint_hv_episode_window
        ):
            episode_vectors = episode_vectors[-self.morl_checkpoint_hv_episode_window :, :]

        if episode_vectors.shape[1] != 2:
            fallback_score = float(np.mean(np.sum(episode_vectors, axis=1)))
            return {
                "pareto_metric_name": self.morl_checkpoint_hv_metric,
                "pareto_metric_fallback": "mean_scalar_reward",
                "pareto_reward_dim": int(episode_vectors.shape[1]),
                "pareto_episode_count": int(episode_vectors.shape[0]),
                "pareto_nondominated_points": 0,
                "pareto_hv": 0.0,
                "pareto_score": fallback_score,
            }

        ref_point = self._resolve_checkpoint_hv_ref_point(
            reward_dim=2, episode_vectors=episode_vectors
        )
        nondominated_mask_arr = pareto_nondominated_mask(episode_vectors)
        nondominated_points = episode_vectors[nondominated_mask_arr]
        hv = hypervolume_2d_maximize(points=nondominated_points, ref_point=ref_point)

        score = float(hv)
        if self.morl_checkpoint_hv_metric == "pareto_hv_nondom":
            score += self.morl_checkpoint_hv_nondominated_bonus * (
                float(nondominated_points.shape[0]) / float(max(1, episode_vectors.shape[0]))
            )

        return {
            "pareto_metric_name": self.morl_checkpoint_hv_metric,
            "pareto_reward_dim": int(episode_vectors.shape[1]),
            "pareto_episode_count": int(episode_vectors.shape[0]),
            "pareto_nondominated_points": int(nondominated_points.shape[0]),
            "pareto_hv": float(hv),
            "pareto_ref_point_0": float(ref_point[0]),
            "pareto_ref_point_1": float(ref_point[1]),
            "pareto_score": float(score),
        }

    def _update_checkpoint_reward_cache(self) -> None:
        env_history = self._extract_env_history()
        if len(env_history) == 0:
            return

        if self._checkpoint_history_offset > len(env_history):
            self._checkpoint_history_offset = 0
            self._checkpoint_episode_reward_sum = {}
            self._checkpoint_episode_reward_count = {}

        new_entries = env_history[self._checkpoint_history_offset :]
        if len(new_entries) == 0:
            return

        for entry in new_entries:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue

            step_reward = entry[1]
            reward_vector = to_reward_vector(step_reward)
            if reward_vector is None:
                continue

            episode_number = entry[5] if len(entry) > 5 else 0
            try:
                episode_idx = int(episode_number)
            except (TypeError, ValueError):
                episode_idx = 0

            existing = self._checkpoint_episode_reward_sum.get(episode_idx)
            if existing is None:
                self._checkpoint_episode_reward_sum[episode_idx] = reward_vector.copy()
            elif existing.shape == reward_vector.shape:
                existing += reward_vector
            else:
                min_dim = min(existing.shape[0], reward_vector.shape[0])
                if min_dim > 0:
                    existing[:min_dim] += reward_vector[:min_dim]
                    self._checkpoint_episode_reward_sum[episode_idx] = existing[:min_dim]

            self._checkpoint_episode_reward_count[episode_idx] = (
                int(self._checkpoint_episode_reward_count.get(episode_idx, 0)) + 1
            )

        self._checkpoint_history_offset = len(env_history)

    def _extract_env_history(self) -> list[Any]:
        for candidate in (self.env, getattr(self, "_get_primary_env", lambda: None)()):
            env_history = extract_env_history_from_candidate(candidate)
            if env_history is not None:
                return env_history
        return []

    def _resolve_checkpoint_hv_ref_point(
        self, reward_dim: int, episode_vectors: np.ndarray
    ) -> np.ndarray:
        configured_ref = self.morl_checkpoint_hv_ref_point
        if (
            isinstance(configured_ref, (list, tuple))
            and len(configured_ref) >= reward_dim
            and all(isinstance(v, (int, float)) for v in configured_ref[:reward_dim])
        ):
            return np.asarray(configured_ref[:reward_dim], dtype=np.float64)

        min_values = np.min(episode_vectors, axis=0)
        margin = np.maximum(0.05 * np.maximum(np.abs(min_values), 1.0), 1e-3)
        return (min_values - margin).astype(np.float64)

    def _update_best_checkpoint_alias(
        self,
        model_dir: str,
        exp_name: str,
        checkpoint_paths: list[str],
        step: int,
        reason: str,
        score: float,
        score_details: dict[str, Any],
        upload_to_wandb: bool,
    ) -> str | None:
        source_checkpoint = self._select_preferred_checkpoint_path(checkpoint_paths)
        if source_checkpoint is None:
            return None

        source_ext = os.path.splitext(source_checkpoint)[1]
        if source_ext not in {".tar", ".pt", ".pth", ".zip"}:
            source_ext = ".tar"
        alias_path = os.path.join(model_dir, f"{exp_name}_best_hv{source_ext}")
        shutil.copy2(source_checkpoint, alias_path)

        metadata_path = os.path.join(model_dir, f"{exp_name}_best_hv.json")
        metadata_payload: dict[str, Any] = {
            "experiment_name": exp_name,
            "metric_name": self.morl_checkpoint_hv_metric,
            "score": float(score),
            "step": int(step),
            "reason": str(reason),
            "source_checkpoint": source_checkpoint,
            "best_checkpoint_path": alias_path,
            "available_checkpoint_paths": list(checkpoint_paths),
            **score_details,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2, sort_keys=True)

        if self.sim_config.get("wandb", False) and upload_to_wandb:
            try:
                artifact = wandb.Artifact(
                    name=f"{exp_name}-model-best-hv",
                    type="model",
                    metadata={
                        "metric_name": self.morl_checkpoint_hv_metric,
                        "step": int(step),
                        "score": float(score),
                    },
                )
                artifact.add_file(alias_path)
                artifact.add_file(metadata_path)
                wandb.log_artifact(artifact)
            except Exception as exc:
                warnings.warn(
                    f"Failed to upload best-HV MORL checkpoint artifact: {exc}",
                    stacklevel=2,
                )

        return alias_path

    @staticmethod
    def _select_preferred_checkpoint_path(checkpoint_paths: list[str]) -> str | None:
        if len(checkpoint_paths) == 0:
            return None
        ext_priority = {
            ".tar": 0,
            ".pth": 1,
            ".pt": 2,
            ".zip": 3,
        }
        return sorted(
            checkpoint_paths,
            key=lambda path: (
                ext_priority.get(os.path.splitext(path)[1], 99),
                os.path.basename(path),
            ),
        )[0]

    def _append_checkpoint_hv_record(
        self, model_dir: str, exp_name: str, record: dict[str, Any]
    ) -> None:
        path = os.path.join(model_dir, f"{exp_name}_checkpoint_hv_scores.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            warnings.warn(
                f"Failed to persist checkpoint HV score record: {exc}",
                stacklevel=2,
            )

    def _preserve_versioned_checkpoint_files(
        self,
        saved_paths: list[str],
        model_dir: str,
        filename: str,
    ) -> list[str]:
        """Guarantee each checkpoint has a step-tagged filename.

        Some MORL save() implementations ignore the requested filename and always
        overwrite a fixed output path. We keep a versioned copy so every checkpoint
        is preserved.
        """
        kept_paths: list[str] = []
        for src in sorted(set(saved_paths)):
            ext = os.path.splitext(src)[1]
            if ext not in (".tar", ".pt", ".pth", ".zip"):
                ext = ".pt"
            target = os.path.join(model_dir, f"{filename}{ext}")

            if os.path.abspath(src) == os.path.abspath(target):
                kept_paths.append(src)
                continue

            final_target = target
            suffix = 1
            while os.path.exists(final_target):
                if os.path.abspath(src) == os.path.abspath(final_target):
                    break
                final_target = os.path.join(model_dir, f"{filename}_v{suffix}{ext}")
                suffix += 1

            if os.path.abspath(src) != os.path.abspath(final_target):
                shutil.copy2(src, final_target)
            kept_paths.append(final_target)

        return sorted(set(kept_paths))

    def _configure_periodic_checkpointing(self):
        if self.morl_checkpoint_freq_steps <= 0:
            return None

        replay_buffer = getattr(self.model, "replay_buffer", None)
        add_fn = getattr(replay_buffer, "add", None) if replay_buffer is not None else None
        if not callable(add_fn):
            # Fallback: hook into env.step() to track steps for MORLD/PCN
            env = getattr(self.model, "env", None)
            step_fn = getattr(env, "step", None) if env is not None else None
            if not callable(step_fn):
                warnings.warn(
                    "morl_checkpoint_freq_steps is set but neither model.replay_buffer.add "
                    "nor model.env.step is available. Periodic checkpoints are disabled.",
                    stacklevel=2,
                )
                return None

            original_step = step_fn

            def _step_with_checkpoint(*args, **kwargs):
                out = original_step(*args, **kwargs)
                self._maybe_save_periodic_checkpoint()
                return out

            self._checkpoint_replay_buffer = None
            self._checkpoint_original_add = original_step
            self._checkpoint_wrapped_add = _step_with_checkpoint
            env.step = _step_with_checkpoint

            def _cleanup():
                try:
                    env.step = original_step
                except Exception:
                    pass
                self._checkpoint_original_add = None
                self._checkpoint_wrapped_add = None

            return _cleanup

        original_add = add_fn

        def _add_with_checkpoint(*args, **kwargs):
            out = original_add(*args, **kwargs)
            self._maybe_save_periodic_checkpoint()
            return out

        self._checkpoint_replay_buffer = replay_buffer
        self._checkpoint_original_add = original_add
        self._checkpoint_wrapped_add = _add_with_checkpoint
        replay_buffer.add = _add_with_checkpoint

        def _cleanup():
            try:
                replay_buffer.add = original_add
            except Exception:
                pass
            self._checkpoint_replay_buffer = None
            self._checkpoint_original_add = None
            self._checkpoint_wrapped_add = None

        return _cleanup

    def _maybe_save_periodic_checkpoint(self) -> None:
        if self._next_checkpoint_step is None:
            return

        step = int(getattr(self.model, "global_step", 0))
        if step < self._next_checkpoint_step:
            return

        saved_paths = self._save_checkpoint_with_hook_temporarily_disabled(
            upload_to_wandb=self.morl_checkpoint_upload_to_wandb
            and self.sim_config.get("wandb", False),
            reason="periodic",
        )
        if self._should_run_periodic_eval(step=step):
            checkpoint_path = self._select_checkpoint_for_periodic_eval(saved_paths)
            if checkpoint_path is not None:
                self._run_periodic_eval_for_checkpoint(checkpoint_path=checkpoint_path, step=step)
        self._next_checkpoint_step += self.morl_checkpoint_freq_steps

    def _save_checkpoint_with_hook_temporarily_disabled(
        self, upload_to_wandb: bool, reason: str
    ) -> list[str]:
        replay_buffer = self._checkpoint_replay_buffer
        original_add = self._checkpoint_original_add
        wrapped_add = self._checkpoint_wrapped_add
        hook_disabled = False

        if (
            replay_buffer is not None
            and callable(original_add)
            and callable(wrapped_add)
            and getattr(replay_buffer, "add", None) is wrapped_add
        ):
            replay_buffer.add = original_add
            hook_disabled = True
        elif replay_buffer is None and callable(original_add) and callable(wrapped_add):
            # env.step hook: temporarily restore original step
            env = getattr(self.model, "env", None)
            if env is not None and getattr(env, "step", None) is wrapped_add:
                env.step = original_add
                hook_disabled = True

        try:
            return self._save_model_checkpoint(upload_to_wandb=upload_to_wandb, reason=reason)
        finally:
            if hook_disabled:
                if replay_buffer is not None:
                    replay_buffer.add = wrapped_add
                else:
                    env = getattr(self.model, "env", None)
                    if env is not None:
                        env.step = wrapped_add

    def _get_model_save_dir(self) -> str:
        model_output_dir = self.sim_config.get("model_output_dir")
        if isinstance(model_output_dir, str) and len(model_output_dir.strip()) > 0:
            output_dir = Path(model_output_dir)
            if not output_dir.is_absolute():
                output_dir = Path(config.ROOT_PATH) / output_dir
            return str(output_dir)
        if self.sim_config.get("wandb", False) and getattr(self, "wandb_run", None) is not None:
            run_dir = getattr(self.wandb_run, "dir", None)
            if isinstance(run_dir, str) and len(run_dir) > 0:
                return os.path.join(run_dir, "models")
        return os.path.join(self.sim_config["log_dir"], "models")

    def _configure_model_save_defaults(self) -> None:
        save_fn = getattr(self.model, "save", None)
        if not callable(save_fn):
            return

        try:
            save_sig = inspect.signature(save_fn)
        except (TypeError, ValueError):
            return

        if "save_dir" not in save_sig.parameters:
            return

        model_dir = self._get_model_save_dir()
        os.makedirs(model_dir, exist_ok=True)
        original_save = save_fn

        def _save_with_default_dir(*args, **kwargs):
            save_kwargs = dict(kwargs)
            if "save_dir" not in save_kwargs:
                # Inject default only when caller doesn't provide one
                bound = None
                try:
                    bound = inspect.signature(original_save).bind_partial(*args, **save_kwargs)
                except TypeError:
                    pass
                if bound is None or "save_dir" not in (bound.arguments or {}):
                    save_kwargs["save_dir"] = model_dir
            return original_save(*args, **save_kwargs)

        _save_with_default_dir.__wrapped__ = original_save  # type: ignore[attr-defined]
        functools.update_wrapper(_save_with_default_dir, original_save)
        self.model.save = _save_with_default_dir

    # -------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        try:
            parsed = float(value)
            if np.isfinite(parsed):
                return parsed
        except Exception:
            return None
        return None

    @staticmethod
    def _summarize_periodic_eval_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
        if len(rows) == 0:
            return {}

        episode_returns = [
            value
            for value in (
                train_morl_experiment._to_float_or_none(row.get("episode_return")) for row in rows
            )
            if value is not None
        ]
        bus_vals = [
            value
            for value in (
                train_morl_experiment._to_float_or_none(row.get("Bus_crossing_time"))
                for row in rows
            )
            if value is not None
        ]
        car_vals = [
            value
            for value in (
                train_morl_experiment._to_float_or_none(row.get("Car_crossing_time"))
                for row in rows
            )
            if value is not None
        ]
        gap_vals = [
            value
            for value in (
                train_morl_experiment._to_float_or_none(row.get("Bus_faster_crossing_time_gap"))
                for row in rows
            )
            if value is not None
        ]

        points = list(zip(bus_vals, car_vals, strict=True))
        summary: dict[str, float] = {
            "row_count": float(len(rows)),
            "pareto_hv_auto_ref": pareto_hv_2d_minimize(points)
            if len(points) > 0
            else float("nan"),
            "pareto_nondominated_count": float(len(nondominated_points_minimize(points)))
            if len(points) > 0
            else 0.0,
        }
        if len(episode_returns) > 0:
            summary["episode_return_mean"] = float(np.mean(episode_returns))
        if len(bus_vals) > 0:
            summary["bus_crossing_time_mean"] = float(np.mean(bus_vals))
        if len(car_vals) > 0:
            summary["car_crossing_time_mean"] = float(np.mean(car_vals))
        if len(gap_vals) > 0:
            summary["bus_faster_crossing_time_gap_mean"] = float(np.mean(gap_vals))
        return summary

    @staticmethod
    def _extract_weight_bus(row: dict[str, Any]) -> float | None:
        raw = row.get("weights")
        if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
            return train_morl_experiment._to_float_or_none(raw[0])
        if not isinstance(raw, str):
            return None
        raw = raw.strip()
        if len(raw) == 0:
            return None

        parsed: Any
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = json.loads(raw.replace("'", '"'))
            except Exception:
                parsed = [
                    train_morl_experiment._to_float_or_none(v)
                    for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                ]
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return train_morl_experiment._to_float_or_none(parsed[0])
        return None


"""
Logging wrappers
"""


class _MORLLoggingMixin:
    def _init_morl_logging(
        self,
        train_prefix: str = "",
        log_every_n_steps: int = 1,
        save_trajectories: bool = True,
    ) -> None:
        self.train_prefix = train_prefix
        self.last_push: dict[str, Any] = {}
        self.train_episode_push_history: dict[str, list[float]] = defaultdict(list)
        self.trajectories: list[Any] = []
        self.train_episode_info: dict[str, Any] = {}
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.save_trajectories = bool(save_trajectories)
        self._log_step_counter = 0

    def _get_primary_env(self) -> Any:
        """
        Return the first concrete env for metrics/trajectory access.
        Mirrors SB3 callback behavior that reads index 0 in VecEnv.
        """
        base_env = self.env.unwrapped  # type: ignore
        envs = getattr(base_env, "envs", None)
        if isinstance(envs, (list, tuple)) and len(envs) > 0:
            primary = envs[0]
            return primary.unwrapped if hasattr(primary, "unwrapped") else primary
        return base_env

    def _log_train_ts(self) -> None:
        self._log_step_counter += 1
        if self._log_step_counter % self.log_every_n_steps != 0:
            return

        env = self._get_primary_env()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            current_push = {
                f"{self.train_prefix}ts/{k}": v for k, v in env.get_numerical_info_dict().items()
            }  # type: ignore
            current_push.update(
                {
                    f"{self.train_prefix}ts_bus/{k}": v
                    for k, v in env.get_numerical_info_dict_bus().items()
                }
            )  # type: ignore
            current_push.update(
                {
                    f"{self.train_prefix}rewards/{k}": v
                    for k, v in env.get_all_reward_metrics().items()
                }
            )  # type: ignore

        push_on_change = {k: v for k, v in current_push.items() if v != self.last_push.get(k, None)}
        if len(push_on_change) > 0:
            log_to_wandb(push_on_change)

        for k, v in current_push.items():
            self.train_episode_push_history[f"{self.train_prefix}episode/{k.split('/')[1]}"].append(
                v
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for k, v in env.get_mean_accumulated_waiting_time_per_lane().items():  # type: ignore
                self.train_episode_push_history[f"{self.train_prefix}episode_lane/{k}"].append(v)

        episode_key = f"{self.train_prefix}ts/episode"
        if self.last_push and self.last_push.get(episode_key) != current_push.get(episode_key):
            push_episode_history(
                episode_push_history=self.train_episode_push_history,
                episode_info=self.train_episode_info,
                prefix=self.train_prefix,
            )
            if self.save_trajectories:
                save_trajectory(
                    trajectories=self.trajectories,
                    ep=current_push[episode_key],
                    prefix=self.train_prefix,
                )
            self.train_episode_push_history = defaultdict(list)

        self.last_push = current_push
        self.trajectories = getattr(env, "trajectory", [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.train_episode_info = env.get_episode_info()  # type: ignore


class MORLWandbLoggingWrapper(gym.Wrapper, _MORLLoggingMixin):
    """
    Step-based W&B logging wrapper for MORL algorithms that do not use SB3 callbacks.
    """

    def __init__(
        self,
        env: gym.Env,
        train_prefix: str = "",
        log_every_n_steps: int = 1,
        save_trajectories: bool = True,
    ):
        gym.Wrapper.__init__(self, env)
        self._init_morl_logging(
            train_prefix=train_prefix,
            log_every_n_steps=log_every_n_steps,
            save_trajectories=save_trajectories,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._log_train_ts()
        return obs, reward, terminated, truncated, info


class MORLVecWandbLoggingWrapper(VectorWrapper, _MORLLoggingMixin):
    """
    Vector-env variant of MORL W&B logging wrapper.
    """

    def __init__(
        self,
        env: VectorEnv,
        train_prefix: str = "",
        log_every_n_steps: int = 1,
        save_trajectories: bool = True,
    ):
        VectorWrapper.__init__(self, env)
        self._init_morl_logging(
            train_prefix=train_prefix,
            log_every_n_steps=log_every_n_steps,
            save_trajectories=save_trajectories,
        )

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self._log_train_ts()
        return obs, rewards, terminations, truncations, infos
