# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

import gc
import hashlib
import json
import math
import multiprocessing as mp
import multiprocessing.pool as mp_pool
import os
import pickle
import re
import time
import traceback
import warnings
from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objs as go
import scipy.stats as stats
import wandb.sdk
from optuna.storages import RDBStorage
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from plotly.subplots import make_subplots

import wandb
from morl_tsp import config
from morl_tsp.experiment.experiment import train_experiment
from morl_tsp.experiment.morl_experiment import train_morl_experiment
from morl_tsp.experiment.utils import (
    extract_env_history_from_candidate,
    hypervolume_2d_maximize,
    pareto_nondominated_mask,
    resolve_optuna_connection_string,
    resolve_pareto_ref_point,
    to_reward_vector,
)


class _NonDaemonProcess(mp.Process):
    @property
    def daemon(self) -> bool:
        return False

    @daemon.setter
    def daemon(self, value: bool) -> None:
        pass


class _NonDaemonPool(mp_pool.Pool):
    def Process(self, *args: Any, **kwargs: Any) -> mp.Process:  # noqa: N802
        proc = super().Process(*args, **kwargs)
        proc.__class__ = _NonDaemonProcess
        return proc


class HPExperiment:
    _PARETO_OBJECTIVE_METRICS = {"pareto_hv", "pareto_hypervolume", "pareto_hv_nondom"}

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        base_config: dict,
        hp_tuning_config: dict,
        agent_tuning_config: dict,
        env_seeds: list[int] | tuple[int, ...],
        experiment_type: Literal["intersection_zoo", None] = None,
        callbacks: list[Literal["wandb", "optuna_prune"]] | None = None,
        stable_baselines_model: Literal["PPO", "A2C", "DQN", "MaskablePPO"] = "PPO",
        experiment_runner: Literal["auto", "sb3", "morl"] = "auto",
        objective_metric: str = "reward",
        group: str = "default_hpt_group",
        agent_seed: int = 27429,  # this seed is for the agent to produce deterministic results, it is not used for the environment
        worker: int = 0,  # index of the worker running this experiment
        n_direction: Literal[
            "maximize", "minimize"
        ] = "maximize",  # direction of optimization for the objective function, can be "maximize" or "minimize"
        n_trials: int = 20,  # total number of trials to run
        n_startup_trials: int = 5,  # number of full trials to run before pruning starts
        n_warmup_steps: int = 5000,  # number of steps to run before pruning starts
        db_study_name: str = "default_study",  # this name is used to store the study in the database
        db_load_if_exists: bool = True,  # this is needed for optuna to load the study if it already exists in the database
        db_connection_string: str | None = None,
        default_log_params: tuple[str, ...] = ("learning_rate", "ent_coef"),
    ):
        assert db_study_name is not None, "Database study name must be provided for Optuna."
        assert db_load_if_exists is not None, "Database load_if_exists must be provided for Optuna."
        callbacks = ["wandb"] if callbacks is None else list(callbacks)
        db_connection_string = resolve_optuna_connection_string(
            db_study_name=db_study_name,
            db_connection_string=db_connection_string,
        )

        self.base_config = base_config
        self.env_seeds = env_seeds

        self.experiment_type = experiment_type
        self.hp_tuning_config = hp_tuning_config
        self.agent_tuning_config = agent_tuning_config
        self.callbacks = callbacks
        self.stable_baselines_model = stable_baselines_model
        self.experiment_runner = self._resolve_experiment_runner(experiment_runner)
        self.objective_metric = objective_metric
        self.group = group
        self.agent_seed = agent_seed

        self.worker = worker

        self.n_direction = n_direction
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

        self.db_connection_string = db_connection_string
        self.db_study_name = db_study_name
        self.db_load_if_exists = db_load_if_exists

        self.default_log_params = default_log_params

    # -------------------------------------------------------------------------
    # Trial execution
    # -------------------------------------------------------------------------

    def _resolve_experiment_runner(
        self, experiment_runner: Literal["auto", "sb3", "morl"]
    ) -> Literal["sb3", "morl"]:
        if experiment_runner in {"sb3", "morl"}:
            return cast(Literal["sb3", "morl"], experiment_runner)
        if "morl_algorithm" in self.base_config:
            return "morl"
        return "sb3"

    def objective(self, trial: optuna.Trial) -> float:

        trial.set_user_attr("agent_seed", self.agent_seed)
        trial.set_user_attr("worker", self.worker)
        trial.set_user_attr("env_seeds", list(self.env_seeds))
        trial.set_user_attr("objective_metric", self.objective_metric)
        trial.set_user_attr("experiment_runner", self.experiment_runner)

        sampled_model_params = {
            param: self._suggest_parameters(trial=trial, parameter_name=param, value_range=val)
            for param, val in self.hp_tuning_config.items()
        }
        if self.experiment_runner == "sb3":
            sampled_model_params = self.apply_special_hyperparams(sampled_model_params)
        print(f"[trial {trial.number}] params={sampled_model_params}")

        sampled_agent_params = {
            param: self._suggest_parameters(trial=trial, parameter_name=param, value_range=val)
            for param, val in self.agent_tuning_config.items()
        }

        trial_hyperparameters, trial_overrides = self._build_trial_parameters(
            sampled_model_params=sampled_model_params,
            sampled_agent_params=sampled_agent_params,
        )

        objective_values_per_seed_session: dict[int, float] = {}
        episode_stats_per_seed_session: dict[int, dict[str, float]] = {}
        use_pareto_objective = self._uses_pareto_objective()

        for seed_index, seed in enumerate(self.env_seeds):
            trial.set_user_attr("current_env_seed", int(seed))
            trial.set_user_attr("seed_index", int(seed_index))

            sim_config_with_trial = deepcopy(self.base_config)
            sim_config_with_trial["random_seed"] = int(seed)

            exp = self._build_trial_experiment(
                trial=trial,
                seed=int(seed),
                trial_hyperparameters=trial_hyperparameters,
                trial_overrides=trial_overrides,
                sim_config_with_trial=sim_config_with_trial,
            )

            try:
                if hasattr(exp, "wandb_run") and exp.wandb_run is not None:
                    trial.set_user_attr("last_wandb_run_id", exp.wandb_run.id)
                    trial.set_user_attr("last_wandb_run_name", exp.wandb_run.name)
            except Exception:
                pass

            exp.run()

            env_history = self._extract_env_history(exp)
            if len(env_history) == 0:
                raise ValueError("No environment history recorded during the experiment.")

            rewards_per_episode = defaultdict(list)
            episode_reward_vector_sum: dict[int, np.ndarray] = {}
            episode_reward_vector_count: defaultdict[int, int] = defaultdict(int)

            ep_stat_sum: defaultdict[int, defaultdict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            ep_stat_count: defaultdict[int, defaultdict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )

            for entry in env_history:
                step_reward = entry[1] if len(entry) > 1 else None
                episode_number = entry[5] if len(entry) > 5 else 0
                step_info = entry[6] if len(entry) > 6 else {}

                episode_idx = int(episode_number)

                if not use_pareto_objective:
                    step_objective = self._extract_objective_step_value(
                        step_reward=step_reward, step_info=step_info
                    )
                    if step_objective is not None:
                        rewards_per_episode[episode_idx].append(float(step_objective))

                reward_vector = to_reward_vector(step_reward)
                if reward_vector is not None:
                    if episode_idx not in episode_reward_vector_sum:
                        episode_reward_vector_sum[episode_idx] = reward_vector.copy()
                    else:
                        existing = episode_reward_vector_sum[episode_idx]
                        if existing.shape == reward_vector.shape:
                            existing += reward_vector
                        else:
                            min_dim = min(existing.shape[0], reward_vector.shape[0])
                            if min_dim > 0:
                                existing[:min_dim] += reward_vector[:min_dim]
                    episode_reward_vector_count[episode_idx] += 1

                if isinstance(step_info, dict):
                    for k, v in step_info.items():
                        if isinstance(v, (int, float)):
                            ep_stat_sum[episode_idx][k] += float(v)
                            ep_stat_count[episode_idx][k] += 1

            if use_pareto_objective:
                seed_objective, seed_objective_details = self._compute_pareto_seed_objective(
                    episode_reward_vector_sum=episode_reward_vector_sum,
                    episode_reward_vector_count=episode_reward_vector_count,
                )
                objective_values_per_seed_session[seed] = float(seed_objective)
                for detail_key, detail_value in seed_objective_details.items():
                    trial.set_user_attr(f"seed_{seed}_{detail_key}", detail_value)
            else:
                if not rewards_per_episode:
                    raise ValueError(
                        f"No objective values were collected during the experiment for metric '{self.objective_metric}'."
                    )

                mean_rewards_per_episode = {
                    ep: np.mean(rewards) for ep, rewards in rewards_per_episode.items()
                }
                objective_values_per_seed_session[seed] = float(
                    np.mean(list(mean_rewards_per_episode.values()))
                )

            # per-episode means
            ep_stat_mean: defaultdict[int, dict[str, float]] = defaultdict(dict)
            for ep, sums in ep_stat_sum.items():
                for k, s in sums.items():
                    c = ep_stat_count[ep][k]
                    if c > 0:
                        ep_stat_mean[ep][k] = s / c

            # per-seed stats: equal weight per episode
            seed_stat_sum: defaultdict[str, float] = defaultdict(float)
            seed_stat_count: defaultdict[str, int] = defaultdict(int)

            for _ep, episode_stats in ep_stat_mean.items():
                for k, v in episode_stats.items():
                    seed_stat_sum[k] += v
                    seed_stat_count[k] += 1

            episode_stats_per_seed_session[seed] = {
                k: seed_stat_sum[k] / seed_stat_count[k]
                for k in seed_stat_sum
                if seed_stat_count[k] > 0
            }

        if not objective_values_per_seed_session:
            raise ValueError("No seed runs produced objective values.")

        trial.set_user_attr("agent_seed", self.agent_seed)
        trial.set_user_attr("worker", self.worker)

        self.analyze_rewards(list(objective_values_per_seed_session.values()), trial)

        average_objective_per_seed = float(
            np.mean(list(objective_values_per_seed_session.values()))
        )

        average_stats_across_seeds: defaultdict[str, float] = defaultdict(float)
        counts_across_seeds: defaultdict[str, int] = defaultdict(int)

        for _seed, seed_stats in episode_stats_per_seed_session.items():
            for stat_key, stat_value in seed_stats.items():
                average_stats_across_seeds[stat_key] += stat_value
                counts_across_seeds[stat_key] += 1

        for stat_key in average_stats_across_seeds:
            avg_stat = average_stats_across_seeds[stat_key] / counts_across_seeds[stat_key]
            # Keep *_vs_reward for backwards compatibility with existing dashboards.
            trial.set_user_attr(f"{stat_key}_vs_reward", (avg_stat, average_objective_per_seed))
            trial.set_user_attr(f"{stat_key}_vs_objective", (avg_stat, average_objective_per_seed))

        return average_objective_per_seed

    def _build_trial_parameters(
        self,
        sampled_model_params: dict[str, Any],
        sampled_agent_params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        trial_hyperparameters: dict[str, Any] = {}
        trial_overrides: dict[str, Any] = {}

        if self.experiment_runner == "sb3":
            trial_hyperparameters = dict(sampled_model_params)
            trial_hyperparameters["seed"] = self.agent_seed
        else:
            for parameter_name, value in sampled_model_params.items():
                self._apply_sampled_param_to_overrides(
                    overrides=trial_overrides,
                    parameter_name=parameter_name,
                    value=value,
                    default_prefix="morl_algorithm_kwargs",
                )
            if not self._has_nested_key(trial_overrides, "morl_algorithm_kwargs.seed"):
                self._set_nested_value(
                    trial_overrides, "morl_algorithm_kwargs.seed", self.agent_seed
                )

        for parameter_name, value in sampled_agent_params.items():
            self._apply_sampled_param_to_overrides(
                overrides=trial_overrides,
                parameter_name=parameter_name,
                value=value,
            )

        return trial_hyperparameters, trial_overrides

    def _build_trial_experiment(
        self,
        trial: optuna.Trial,
        seed: int,
        trial_hyperparameters: dict[str, Any],
        trial_overrides: dict[str, Any],
        sim_config_with_trial: dict[str, Any],
    ) -> train_experiment | train_morl_experiment:
        callbacks_for_trial = list(self.callbacks)
        if self.experiment_runner == "morl":
            sim_config_with_trial["morl_auto_eval_env"] = False
            morl_train_kwargs = sim_config_with_trial.get("morl_train_kwargs")
            if isinstance(morl_train_kwargs, dict):
                morl_train_kwargs["eval_env"] = None

        common_kwargs = {
            "name": f"{self.group}_t{trial.number}_w{self.worker}_s{seed}",
            "group": self.group,
            "base_config": sim_config_with_trial,
            "config_overwrite": deepcopy(trial_overrides),
            "callbacks": callbacks_for_trial,
            "experiment_type": self.experiment_type,
            "wandb_tags": [
                "optuna",
                "hyperparameter_tuning",
                "individual_hpt_trial",
                f"runner_{self.experiment_runner}",
            ],
        }

        if self.experiment_runner == "morl":
            return train_morl_experiment(
                hyperparameters=trial_hyperparameters,
                **common_kwargs,  # type: ignore[arg-type]
            )

        return train_experiment(
            hyperparameters=trial_hyperparameters,
            stable_baselines_model=self.stable_baselines_model,
            optuna_trial=trial,
            **common_kwargs,  # type: ignore[arg-type]
        )

    # -------------------------------------------------------------------------
    # Objective aggregation
    # -------------------------------------------------------------------------

    def _apply_sampled_param_to_overrides(
        self,
        overrides: dict[str, Any],
        parameter_name: str,
        value: Any,
        default_prefix: str | None = None,
    ) -> None:
        target_path = parameter_name
        if (
            "." not in parameter_name
            and isinstance(default_prefix, str)
            and len(default_prefix) > 0
        ):
            target_path = f"{default_prefix}.{parameter_name}"
        self._set_nested_value(overrides, target_path, value)

    def _extract_env_history(self, exp: Any) -> list[Any]:
        final_snapshot = getattr(exp, "final_env_history_snapshot", None)
        if isinstance(final_snapshot, list) and len(final_snapshot) > 0:
            return final_snapshot

        final_snapshot_path = getattr(exp, "final_env_history_path", None)
        if isinstance(final_snapshot_path, str) and len(final_snapshot_path) > 0:
            try:
                with open(final_snapshot_path, "rb") as handle:
                    file_snapshot = pickle.load(handle)
                if isinstance(file_snapshot, list) and len(file_snapshot) > 0:
                    return file_snapshot
            except Exception:
                pass

        for candidate in (getattr(exp, "train_env", None), getattr(exp, "env", None)):
            env_history = extract_env_history_from_candidate(candidate)
            if env_history is not None:
                return env_history

        get_primary_env = getattr(exp, "_get_primary_env", None)
        if callable(get_primary_env):
            try:
                env_history = extract_env_history_from_candidate(get_primary_env())
                if env_history is not None:
                    return env_history
            except Exception:
                pass

        return []

    def _extract_objective_step_value(self, step_reward: Any, step_info: Any) -> float | None:
        metric = str(self.objective_metric)
        if metric in {"reward", "reward_sum"}:
            return self._to_scalar_reward(step_reward)

        reward_component = re.match(r"^reward\[(\d+)\]$", metric)
        if reward_component is not None:
            return self._to_reward_component(step_reward, int(reward_component.group(1)))

        if isinstance(step_info, dict):
            step_value = step_info.get(metric)
            if isinstance(step_value, (int, float, np.floating)):
                return float(step_value)

        return self._to_scalar_reward(step_reward)

    def _uses_pareto_objective(self) -> bool:
        return str(self.objective_metric).strip().lower() in self._PARETO_OBJECTIVE_METRICS

    def _resolve_pareto_ref_point(self, reward_dim: int, episode_vectors: np.ndarray) -> np.ndarray:
        return resolve_pareto_ref_point(
            reward_dim=reward_dim,
            episode_vectors=episode_vectors,
            configured_ref=self.base_config.get("hpt_pareto_ref_point"),
            morl_ref=self.base_config.get("morl_ref_point"),
        )

    def _compute_pareto_seed_objective(
        self,
        episode_reward_vector_sum: dict[int, np.ndarray],
        episode_reward_vector_count: dict[int, int],
    ) -> tuple[float, dict[str, float | int | str]]:
        episode_ids = sorted(episode_reward_vector_sum.keys())
        if len(episode_ids) == 0:
            raise ValueError("No reward vectors were collected for Pareto objective computation.")

        episode_mean_vectors: list[np.ndarray] = []
        for episode_id in episode_ids:
            vector_sum = episode_reward_vector_sum.get(int(episode_id))
            vector_count = int(episode_reward_vector_count.get(int(episode_id), 0))
            if vector_sum is None or vector_count <= 0:
                continue
            episode_mean_vectors.append(vector_sum / float(vector_count))

        if len(episode_mean_vectors) == 0:
            raise ValueError(
                "No per-episode reward vectors were available for Pareto objective computation."
            )

        reward_dim = min(int(v.shape[0]) for v in episode_mean_vectors)
        if reward_dim <= 0:
            raise ValueError(
                "Reward vectors have invalid dimensionality for Pareto objective computation."
            )
        episode_vectors = np.stack([v[:reward_dim] for v in episode_mean_vectors], axis=0)

        episode_window = int(self.base_config.get("hpt_pareto_episode_window", 32))
        if episode_window > 0 and episode_vectors.shape[0] > episode_window:
            episode_vectors = episode_vectors[-episode_window:, :]

        metric_name = str(self.objective_metric).strip().lower()
        if episode_vectors.shape[1] != 2:
            fallback_score = float(np.mean(np.sum(episode_vectors, axis=1)))
            return fallback_score, {
                "pareto_metric_name": metric_name,
                "pareto_metric_fallback": "mean_scalar_reward",
                "pareto_reward_dim": int(episode_vectors.shape[1]),
                "pareto_episode_count": int(episode_vectors.shape[0]),
                "pareto_nondominated_points": 0,
                "pareto_hv": 0.0,
                "pareto_score": fallback_score,
            }

        ref_point = self._resolve_pareto_ref_point(reward_dim=2, episode_vectors=episode_vectors)
        nondominated_mask_arr = pareto_nondominated_mask(episode_vectors)
        nondominated_points = episode_vectors[nondominated_mask_arr]
        hv = hypervolume_2d_maximize(points=nondominated_points, ref_point=ref_point)

        score = float(hv)
        if metric_name == "pareto_hv_nondom":
            bonus_coeff = float(self.base_config.get("hpt_pareto_nondominated_bonus", 0.01))
            score += bonus_coeff * (
                float(nondominated_points.shape[0]) / float(max(1, episode_vectors.shape[0]))
            )

        return score, {
            "pareto_metric_name": metric_name,
            "pareto_reward_dim": int(episode_vectors.shape[1]),
            "pareto_episode_count": int(episode_vectors.shape[0]),
            "pareto_nondominated_points": int(nondominated_points.shape[0]),
            "pareto_hv": float(hv),
            "pareto_ref_point_0": float(ref_point[0]),
            "pareto_ref_point_1": float(ref_point[1]),
            "pareto_score": float(score),
        }

    # -------------------------------------------------------------------------
    # Hyperparameter sampling
    # -------------------------------------------------------------------------

    def _suggest_parameters(
        self, trial: optuna.Trial, parameter_name: str, value_range: dict | list | tuple
    ):

        assert isinstance(value_range, (dict, list, tuple)), (
            f"Unsupported hyperparameter format for {parameter_name}: {type(value_range)}"
        )

        # Style A: dict spec {low, high, log?, step?}
        if isinstance(value_range, dict):
            # Explicit categorical
            if "choices" in value_range:
                choices = value_range["choices"]
                if not isinstance(choices, (list, tuple)) or len(choices) == 0:
                    raise ValueError(f"{parameter_name}: 'choices' must be a non-empty list/tuple.")
                return trial.suggest_categorical(parameter_name, list(choices))

            # Explicit range
            if "low" in value_range and "high" in value_range:
                low = value_range["low"]
                high = value_range["high"]
                log = bool(value_range.get("log", parameter_name in self.default_log_params))
                forced_type = value_range.get("type", None)

                # Force type if requested
                if forced_type == "int":
                    step = int(value_range.get("step", 1))
                    return trial.suggest_int(
                        parameter_name, int(low), int(high), step=step, log=log
                    )
                if forced_type == "float":
                    return trial.suggest_float(parameter_name, float(low), float(high), log=log)

                # Infer int vs float
                if isinstance(low, int) and isinstance(high, int):
                    step = int(value_range.get("step", 1))
                    return trial.suggest_int(
                        parameter_name, int(low), int(high), step=step, log=log
                    )

                return trial.suggest_float(parameter_name, float(low), float(high), log=log)

            raise ValueError(
                f"{parameter_name}: dict spec must contain either ('choices') or ('low' and 'high'). Got keys={list(value_range.keys())}"
            )

        # ---- Sequence specs (YAML lists) ----
        if isinstance(value_range, (list, tuple)):
            if len(value_range) == 0:
                raise ValueError(f"{parameter_name}: value_range cannot be empty.")

            # Treat length-2 numeric as a range by default
            if len(value_range) == 2 and all(isinstance(v, (int, float)) for v in value_range):
                low, high = value_range
                log = parameter_name in self.default_log_params

                # int vs float
                if isinstance(low, int) and isinstance(high, int):
                    return trial.suggest_int(parameter_name, int(low), int(high), log=log)
                return trial.suggest_float(parameter_name, float(low), float(high), log=log)

            # Otherwise categorical
            return trial.suggest_categorical(parameter_name, list(value_range))

    def apply_special_hyperparams(self, hparams: dict) -> dict:
        """
        Mutates/normalizes sampled hyperparameters so they match SB3 constructor kwargs.
        Call this after sampling and before model creation.
        """
        hparams = dict(hparams)  # shallow copy

        net_arch_spec = hparams.pop("policy_kwargs_net_arch", None)
        if net_arch_spec is not None:
            parsed = self.parse_policy_kwargs_net_arch(net_arch_spec)

            # Merge into existing policy_kwargs if present
            existing_pk = hparams.get("policy_kwargs")
            if existing_pk is None:
                existing_pk = {}
            if not isinstance(existing_pk, dict):
                raise TypeError(
                    f"policy_kwargs must be a dict when provided, got {type(existing_pk)}"
                )

            # If you prefer strictness, replace this overwrite with an error.
            existing_pk = dict(existing_pk)
            existing_pk.update(parsed)  # type: ignore # adds/overwrites 'net_arch'
            hparams["policy_kwargs"] = existing_pk

        return hparams

    # -------------------------------------------------------------------------
    # Static and class helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _split_nested_key(parameter_name: str) -> list[str]:
        return [part for part in str(parameter_name).split(".") if len(part) > 0]

    @classmethod
    def _set_nested_value(cls, payload: dict[str, Any], parameter_name: str, value: Any) -> None:
        keys = cls._split_nested_key(parameter_name)
        if len(keys) == 0:
            raise ValueError(f"Invalid parameter path '{parameter_name}'.")
        current = payload
        for key in keys[:-1]:
            next_value = current.get(key)
            if not isinstance(next_value, dict):
                next_value = {}
                current[key] = next_value
            current = next_value
        current[keys[-1]] = value

    @classmethod
    def _has_nested_key(cls, payload: dict[str, Any], parameter_name: str) -> bool:
        keys = cls._split_nested_key(parameter_name)
        if len(keys) == 0:
            return False
        current: Any = payload
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        return True

    @staticmethod
    def _to_scalar_reward(step_reward: Any) -> float | None:
        if step_reward is None:
            return None
        reward_array = np.asarray(step_reward, dtype=np.float64)
        if reward_array.size == 0:
            return None
        return float(np.sum(reward_array))

    @staticmethod
    def _to_reward_component(step_reward: Any, reward_idx: int) -> float | None:
        if step_reward is None:
            return None
        reward_array = np.asarray(step_reward, dtype=np.float64).reshape(-1)
        if reward_array.size <= reward_idx:
            return None
        return float(reward_array[reward_idx])

    @staticmethod
    def analyze_rewards(rewards: list, trial: optuna.Trial):
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        std_reward = np.std(rewards)
        iqr_reward = stats.iqr(rewards)
        skewness = stats.skew(rewards)

        trial.set_user_attr("mean_reward_across_seeds", mean_reward)
        trial.set_user_attr("median_reward_across_seeds", median_reward)
        trial.set_user_attr("std_reward_across_seeds", std_reward)
        trial.set_user_attr("iqr_reward_across_seeds", iqr_reward)
        trial.set_user_attr("skewness_reward_across_seeds", skewness)

        print(f"Trial {trial.number} -> Rewards across seeds:")
        print(f"Mean: {mean_reward:.3f}")
        print(f"Median: {median_reward:.3f}")
        print(f"Std Dev: {std_reward:.3f}")
        print(f"IQR: {iqr_reward:.3f}")
        print(f"Skewness: {skewness:.3f}")
        print(f"Rewards: {rewards}")

    @staticmethod
    def parse_policy_kwargs_net_arch(spec: Any) -> dict | None:
        """
        Converts a compact string like:
        'pi_64x64_vf_64x64'
        into Stable-Baselines3 policy_kwargs:
        {'net_arch': {'pi': [64, 64], 'vf': [64, 64]}}

        Returns None if spec is None.
        """
        if spec is None:
            return None
        if isinstance(spec, dict):
            return {"net_arch": spec}
        if isinstance(spec, (list, tuple)):
            return {"net_arch": list(spec)}
        if not isinstance(spec, str):
            raise TypeError(f"policy_kwargs_net_arch must be str|dict|list|None, got {type(spec)}")

        s = spec.strip()
        m = re.compile(r"^pi_(?P<pi>\d+(?:x\d+)*)_vf_(?P<vf>\d+(?:x\d+)*)$").match(s)
        if not m:
            raise ValueError(
                f"Invalid policy_kwargs_net_arch='{spec}'. Expected format like "
                f"'pi_64x64_vf_64x64' or 'pi_128x128_vf_128x128'."
            )

        def _layers(x: str) -> list[int]:
            return [int(v) for v in x.split("x") if v]

        return {"net_arch": {"pi": _layers(m.group("pi")), "vf": _layers(m.group("vf"))}}


class ParallelHPExperiments:
    def __init__(
        self,
        hp_tuning_config: dict,
        agent_tuning_config: dict,
        base_config: dict,
        db_study_name: str,
        env_seeds: list[int] | tuple[int, ...],
        experiment_type: Literal["intersection_zoo", None] = None,
        callbacks: list[Literal["wandb", "optuna_prune"]] | None = None,
        stable_baselines_model: Literal["PPO", "A2C", "DQN", "MaskablePPO"] = "PPO",
        experiment_runner: Literal["auto", "sb3", "morl"] = "auto",
        objective_metric: str = "reward",
        n_direction: Literal["maximize", "minimize"] = "maximize",
        group: str = "",
        n_trials: int = 20,
        n_workers: int = 1,
        n_startup_trials: int = 5,
        n_warmup_steps: int = 5000,
        agent_seed: int = 27429,
        db_load_if_exists: bool = True,
        db_connection_string: str | None = None,
    ):

        assert isinstance(env_seeds, (list, tuple)), "env_seeds must be a list or tuple."
        callbacks = ["wandb"] if callbacks is None else list(callbacks)
        db_connection_string = resolve_optuna_connection_string(
            db_study_name=db_study_name,
            db_connection_string=db_connection_string,
        )

        self.hp_tuning_config = hp_tuning_config
        self.agent_tuning_config = agent_tuning_config
        self.base_config = base_config
        self.db_study_name = db_study_name
        self.env_seeds = env_seeds

        self.experiment_type = experiment_type
        self.callbacks = callbacks
        self.stable_baselines_model = stable_baselines_model
        self.experiment_runner = experiment_runner
        self.objective_metric = objective_metric
        self.n_direction = n_direction
        self.group = group if group != "" else db_study_name

        self.n_trials = n_trials
        self.n_workers = n_workers
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

        self.agent_seed = agent_seed

        self.db_load_if_exists = db_load_if_exists
        self.db_connection_string = db_connection_string

    def run(self):
        # Delay the parent W&B run until after forked workers finish. Creating a
        # W&B run before forking can leave child trial processes with inherited
        # W&B service state and broken online auth.
        self.wandb_run = None
        self._ensure_study_initialized()

        base_hp_args = {
            "base_config": self.base_config,
            "env_seeds": self.env_seeds,
            "hp_tuning_config": self.hp_tuning_config,
            "agent_tuning_config": self.agent_tuning_config,
            "experiment_type": self.experiment_type,
            "callbacks": self.callbacks,
            "stable_baselines_model": self.stable_baselines_model,
            "experiment_runner": self.experiment_runner,
            "objective_metric": self.objective_metric,
            "n_direction": self.n_direction,
            "group": self.group,
            "agent_seed": self.agent_seed,
            "db_connection_string": self.db_connection_string,
            "db_study_name": self.db_study_name,
            "db_load_if_exists": self.db_load_if_exists,
            "n_warmup_steps": self.n_warmup_steps,
            "n_startup_trials": self.n_startup_trials,
            "n_trials": 1,  # one trial per task
        }

        status_step = 0
        poll_seconds = 10
        with _NonDaemonPool(processes=self.n_workers, maxtasksperchild=1) as pool:
            async_res = pool.map_async(
                partial(self._run_one_trial, base_hp_args=base_hp_args),
                range(self.n_trials),
            )

            # Poll until done
            while not async_res.ready():
                if self.wandb_run is not None:
                    try:
                        did_log = self._log_live_hpt_status(step=status_step)
                        if did_log:
                            status_step += 1
                    except Exception as e:
                        print(f"[summary wandb] status logging failed: {repr(e)}")
                        print(traceback.format_exc())

                # Only here to surface worker exceptions; ignore expected timeouts
                try:
                    async_res.get(timeout=0.0)
                except mp.TimeoutError:
                    pass

                time.sleep(poll_seconds)

            async_res.get()

        self.wandb_run = self._init_wandb()
        return self.log_results_to_wandb()

    def _ensure_study_initialized(self) -> None:
        storage = RDBStorage(
            url=self.db_connection_string,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 180},
        )
        optuna.create_study(
            storage=storage,
            study_name=self.db_study_name,
            direction=self.n_direction,
            load_if_exists=self.db_load_if_exists,
        )

    @staticmethod
    def _run_one_trial(slot: int, base_hp_args: dict) -> None:
        # Make worker identity unique for naming/logging
        hp_args = dict(base_hp_args)
        hp_args["worker"] = os.getpid()

        hp_exp = HPExperiment(**hp_args)

        storage = RDBStorage(
            url=hp_exp.db_connection_string,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 180},
        )

        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=((abs(hash(hp_exp.db_study_name)) & 0xFFFFFFFF) ^ (slot + 0x9E3779B9))
            & 0xFFFFFFFF,  # random but deterministic per study + slot
            n_startup_trials=hp_exp.n_startup_trials,  # align startup behavior
        )

        study = optuna.load_study(
            storage=storage,
            study_name=hp_exp.db_study_name,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=hp_exp.n_startup_trials,
                n_warmup_steps=hp_exp.n_warmup_steps,
            ),
        )

        # Exactly one trial in this child process
        study.optimize(hp_exp.objective, n_trials=1, gc_after_trial=True)

        # Defensive cleanup (cheap; helps with Python-level objects)
        gc.collect()

    def log_results_to_wandb(self) -> optuna.Study:
        assert self.wandb_run is not None, "WandB run not initialized call method run before."

        storage = RDBStorage(
            url=self.db_connection_string,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 180},
        )

        # After all workers finish, load the full study once
        full_study = optuna.load_study(storage=storage, study_name=self.db_study_name)
        target_label = f"Objective ({self.objective_metric})"

        def _safe_build_plot(plot_name: str, build_fn):
            try:
                return build_fn()
            except Exception as exc:
                warnings.warn(f"Skipping plot '{plot_name}': {exc}", stacklevel=2)
                return None

        contour_plot = _safe_build_plot(
            "contour_plot",
            lambda: plot_contour(full_study, target_name=target_label),
        )
        hp_importances_plot = _safe_build_plot(
            "hp_importances_plot",
            lambda: plot_param_importances(full_study, target_name=target_label),
        )
        edf_plot = _safe_build_plot(
            "edf_plot",
            lambda: plot_edf(full_study, target_name=target_label),
        )
        optimization_history_plot = _safe_build_plot(
            "optimization_history_plot",
            lambda: plot_optimization_history(full_study, target_name=target_label),
        )
        parallel_coordinate_plot = _safe_build_plot(
            "parallel_coordinate_plot",
            lambda: plot_parallel_coordinate(full_study, target_name=target_label),
        )
        slice_plot = _safe_build_plot(
            "slice_plot",
            lambda: plot_slice(full_study, target_name=target_label),
        )
        user_attribute_plot = _safe_build_plot(
            "user_attribute_plot",
            lambda: self.plot_attr_vs_reward(
                full_study, target_name=target_label, colorscale="Viridis"
            ),
        )

        # Create full dataframe from ALL trials
        study_df = full_study.trials_dataframe()
        user_attrs_df = pd.DataFrame([trial.user_attrs for trial in full_study.trials])
        study_df = pd.concat([study_df, user_attrs_df], axis=1)
        study_df = self._sanitize_df_for_wandb(study_df)

        plot_map = {
            "contour_plot": contour_plot,
            "hp_importances_plot": hp_importances_plot,
            "edf_plot": edf_plot,
            "optimization_history_plot": optimization_history_plot,
            "parallel_coordinate_plot": parallel_coordinate_plot,
            "slice_plot": slice_plot,
            "user_attribute_plot": user_attribute_plot,
        }

        for plot_name, fig in plot_map.items():
            if fig is None:
                continue
            try:
                self.log_plotly_html_to_wandb(self.wandb_run, fig, plot_name)
            except Exception as exc:
                warnings.warn(f"Failed HTML logging for plot '{plot_name}': {exc}", stacklevel=2)

        wandb_table = wandb.Table(dataframe=study_df)
        wandb_payload: dict[str, Any] = {"optuna_trials_combined": wandb_table}
        for plot_name, fig in plot_map.items():
            if fig is not None:
                wandb_payload[plot_name] = fig
        self.wandb_run.log(wandb_payload)
        self.wandb_run.finish()

        return full_study

    def _init_wandb(self) -> wandb.sdk.wandb_run.Run:

        wandb_config = {
            "n_trials": self.n_trials,
            "n_startup_trials": self.n_startup_trials,
            "n_warmup_steps": self.n_warmup_steps,
            "agent_seed": self.agent_seed,
            "env_seed": self.env_seeds,
            "objective_metric": self.objective_metric,
            "experiment_runner": self.experiment_runner,
            "stable_baselines_model": self.stable_baselines_model,
        }

        wandb_config = wandb_config | self.hp_tuning_config | self.agent_tuning_config

        wandb_run = wandb.init(
            project=config.WANDB_PROJECT_NAME,
            name=f"{self.group}_table",
            group=self.group,
            reinit=True,
            config=wandb_config,
            tags=["optuna", "hyperparameter_tuning", "parallel", "summary"],
        )

        return wandb_run

    def _load_full_study(self) -> optuna.Study:
        storage = RDBStorage(
            url=self.db_connection_string,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 180},
        )
        return optuna.load_study(storage=storage, study_name=self.db_study_name)

    @staticmethod
    def _safe_cell(v):
        # Convert numpy scalars to python scalars
        if hasattr(v, "item") and callable(v.item):
            try:
                v = v.item()
            except Exception:
                pass

        # Normalize NaN/NaT
        try:
            if isinstance(v, float) and math.isnan(v):
                return None
        except Exception:
            pass

        # pandas Timestamp / numpy datetime64 -> string
        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
        if "numpy" in str(type(v)) and "datetime64" in str(type(v)):
            ts = pd.to_datetime(v, errors="coerce")
            return None if pd.isna(ts) else ts.isoformat()

        # dict/list/tuple -> JSON string
        if isinstance(v, (dict, list, tuple)):
            return json.dumps(v, sort_keys=True, default=str)

        return v

    def _trial_to_row(self, t):
        row = {
            "trial": int(t.number),
            "state": t.state.name,
            "value": None if t.value is None else float(t.value),
            "start": self._safe_cell(t.datetime_start),
            "end": self._safe_cell(t.datetime_complete),
        }

        # Select scalar params you care about for live view
        scalar_param_keys = [
            "n_steps",
            "batch_size",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "n_epochs",
            "clip_range",
            "max_grad_norm",
            "target_kl",
            "normalize_advantage",
            "clip_range_vf",
            "policy",
        ]
        for k in scalar_param_keys:
            if k in t.params:
                row[f"param/{k}"] = self._safe_cell(t.params[k])

        # policy_kwargs causes schema issues -> keep as JSON string
        if "policy_kwargs" in t.params:
            row["param/policy_kwargs_json"] = self._safe_cell(t.params["policy_kwargs"])

        # full params JSON for debugging
        row["params_json"] = self._safe_cell(t.params)

        # Useful user attrs that are scalar-ish
        scalar_attr_keys = [
            "agent_seed",
            "worker",
            "current_env_seed",
            "seed_index",
            "last_wandb_run_id",
            "last_wandb_run_name",
        ]
        for k in scalar_attr_keys:
            if k in (t.user_attrs or {}):
                row[f"attr/{k}"] = self._safe_cell(t.user_attrs[k])

        return row

    @staticmethod
    def _df_fingerprint(df: pd.DataFrame) -> str:
        """
        Stable fingerprint for change-detection.
        Assumes df already has stable column order and row order.
        """
        df2 = df.reset_index(drop=True).copy()
        df2 = df2.where(pd.notnull(df2), None)  # type: ignore
        s = df2.to_json(orient="split", date_format="iso", default_handler=str)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _same_value(a: Any, b: Any) -> bool:
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        return a == b

    def _log_live_hpt_status(self, step: int, max_rows: int = 50) -> bool:
        """
        Returns True if something was logged, False if omitted because nothing changed.
        """
        assert self.wandb_run is not None

        # Lazy-init caches (no need to touch __init__)
        last_scalars: dict[str, Any] = getattr(self, "_hpt_last_scalars", {})
        last_table_fp: dict[str, str] = getattr(self, "_hpt_last_table_fp", {})

        try:
            study = self._load_full_study()
        except KeyError as exc:
            # The first poll can happen before the worker creates the study record.
            if "Record does not exist" in str(exc):
                return False
            raise
        trials = study.trials

        # --- scalars ---
        state_counts = Counter([t.state.name for t in trials])
        scalars: dict[str, Any] = {
            "hpt/total_trials_seen": len(trials),
            "hpt/n_running": state_counts.get("RUNNING", 0),
            "hpt/n_complete": state_counts.get("COMPLETE", 0),
            "hpt/n_pruned": state_counts.get("PRUNED", 0),
            "hpt/n_fail": state_counts.get("FAIL", 0),
        }
        try:
            scalars["hpt/best_value_so_far"] = float(study.best_value)
        except Exception:
            pass

        delta: dict[str, Any] = {}
        for k, v in scalars.items():
            if (k not in last_scalars) or (not self._same_value(last_scalars[k], v)):
                delta[k] = v
                last_scalars[k] = v

        # --- tables (only when they actually change) ---
        if trials:
            df = pd.DataFrame([self._trial_to_row(t) for t in trials])

            # IMPORTANT: keep a stable schema across logs for W&B Tables
            scalar_param_keys = [
                "n_steps",
                "batch_size",
                "learning_rate",
                "gamma",
                "gae_lambda",
                "ent_coef",
                "vf_coef",
                "n_epochs",
                "clip_range",
                "max_grad_norm",
                "target_kl",
                "normalize_advantage",
                "clip_range_vf",
                "policy",
            ]
            scalar_attr_keys = [
                "agent_seed",
                "worker",
                "current_env_seed",
                "seed_index",
                "last_wandb_run_id",
                "last_wandb_run_name",
            ]
            expected_cols = (
                ["trial", "state", "value", "start", "end"]
                + [f"param/{k}" for k in scalar_param_keys]
                + ["param/policy_kwargs_json", "params_json"]
                + [f"attr/{k}" for k in scalar_attr_keys]
            )
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = None
            df = df[expected_cols]  # stable column order

            running_df = df[df["state"] == "RUNNING"].copy()
            running_df = (
                running_df.sort_values("start", ascending=False)
                .head(max_rows)
                .reset_index(drop=True)
            )
            running_df = self._sanitize_df_for_wandb(running_df)

            recent_df = df[df["state"].isin(["COMPLETE", "PRUNED", "FAIL"])].copy()
            recent_df = (
                recent_df.sort_values("end", ascending=False).head(max_rows).reset_index(drop=True)
            )
            recent_df = self._sanitize_df_for_wandb(recent_df)

            running_fp = self._df_fingerprint(running_df)
            if last_table_fp.get("hpt_tables/running_trials") != running_fp:
                delta["hpt_tables/running_trials"] = wandb.Table(dataframe=running_df)
                last_table_fp["hpt_tables/running_trials"] = running_fp

            recent_fp = self._df_fingerprint(recent_df)
            if last_table_fp.get("hpt_tables/recent_trials") != recent_fp:
                delta["hpt_tables/recent_trials"] = wandb.Table(dataframe=recent_df)
                last_table_fp["hpt_tables/recent_trials"] = recent_fp

        # Persist caches
        self._hpt_last_scalars = last_scalars
        self._hpt_last_table_fp = last_table_fp

        # Nothing changed -> omit the push entirely
        if not delta:
            return False

        self.wandb_run.log(delta, step=step)
        return True

    @staticmethod
    def _sanitize_df_for_wandb(df: pd.DataFrame) -> pd.DataFrame:
        sanitized = df.copy()
        for col in sanitized.columns:
            series = sanitized[col]
            if pd.api.types.is_timedelta64_dtype(series):
                sanitized[col] = series.dt.total_seconds()
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                sanitized[col] = series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").where(
                    series.notna(), None
                )
                continue
            if series.dtype == object:
                sanitized[col] = series.map(ParallelHPExperiments._sanitize_wandb_cell)
        return sanitized

    @staticmethod
    def _sanitize_wandb_cell(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, pd.Timedelta):
            return value.total_seconds()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (float, np.floating)) and pd.isna(value):
            return None
        return value

    @staticmethod
    def plot_attr_vs_reward(study, target_name="Objective Value", colorscale="Blues") -> go.Figure:
        """
        Creates a multi-subplot Plotly figure from user attributes ending in *_vs_reward.
        Each subplot shows a stat (X) vs. reward (Y) colored by trial index.

        Args:
            study (optuna.Study): The study containing trials.
            target_name (str): Label for Y-axis.
            colorscale (str): Plotly colorscale.

        Returns:
            go.Figure: A multi-subplot interactive figure.
        """
        user_attributes = defaultdict(list)

        for trial_index, trial in enumerate(study.trials):
            for key, value in trial.user_attrs.items():
                if (
                    key.endswith("_vs_reward")
                    and isinstance(value, (tuple, list))
                    and len(value) == 2
                ):
                    x, y = value
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        user_attributes[key].append((x, y, trial_index))

        if not user_attributes:
            return go.Figure(layout=go.Layout(title="No *_vs_reward attributes found"))

        num_user_attributes = len(user_attributes)
        fig = make_subplots(
            rows=1,
            cols=num_user_attributes,
            shared_yaxes=True,
            subplot_titles=[k.replace("_vs_reward", "") for k in user_attributes.keys()],
        )

        showscale_flag = True

        for col, (user_attribute_key, datapoints) in enumerate(user_attributes.items(), start=1):
            attribute_value, objective_value, trial_id = zip(*datapoints, strict=True)

            trace = go.Scatter(
                x=attribute_value,
                y=objective_value,
                mode="markers",
                name=user_attribute_key.replace("_vs_reward", ""),
                marker=dict(
                    size=8,
                    color=trial_id,
                    colorscale=colorscale,
                    reversescale=True,
                    showscale=showscale_flag,
                    colorbar=dict(title="Trial Index", x=1.0 + 0.05 * col)
                    if showscale_flag
                    else None,
                ),
                hovertemplate=(
                    f"{target_name}: %{{y}}<br>Trial #: %{{marker.color}}<extra></extra>"
                ),
                showlegend=False,
            )

            fig.add_trace(trace, row=1, col=col)
            fig.update_xaxes(
                title_text=user_attribute_key.replace("_vs_reward", ""), row=1, col=col
            )

            if col == 1:
                fig.update_yaxes(title_text=target_name, row=1, col=col)

            showscale_flag = False  # Show colorbar only once

        fig.update_layout(
            title="Attribute vs Reward Scatter Plots",
            hovermode="closest",
            height=900,
            width=320 * num_user_attributes,
            margin=dict(t=100),
        )

        return fig

    @staticmethod
    def log_plotly_html_to_wandb(wandb_run, fig, name: str, include_plotlyjs="cdn"):

        html_data = fig.to_html(
            full_html=False,  # usually better for embedding in W&B
            include_plotlyjs=include_plotlyjs,
        )

        wandb_run.log({name: wandb.Html(html_data)})
