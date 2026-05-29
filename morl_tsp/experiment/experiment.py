# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Experiment classes for RL training and evaluation with Stable Baselines3.

This module provides:
- experiment: Base class for SB3 experiments (inherits from BaseExperiment)
- train_experiment: Training experiments
- eval_experiemnt: Evaluation experiments (note: typo preserved for backward compatibility)
- parallel_experiments_yaml: Run multiple experiments from YAML config
"""

import inspect
import json
import os
import pickle
import re
import subprocess
import sys
import warnings
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import optuna
import yaml
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import wandb
from morl_tsp import SumoEnvironment, config
from morl_tsp.experiment.base import BaseExperiment
from morl_tsp.experiment.callbacks import get_callbacks, push_episode_history, save_trajectory
from morl_tsp.experiment.utils import extract_env_history_from_candidate

model_map: dict[str, type] = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "MaskablePPO": MaskablePPO,
}


class AttrPassActionMasker(ActionMasker):
    """ActionMasker wrapper that passes through attribute access to the underlying env."""

    def __getattr__(self, name):
        return getattr(self.env, name)


class experiment(BaseExperiment):
    """Base class for Stable Baselines 3 experiments.

    Extends BaseExperiment with SB3-specific functionality:
    - Model selection (PPO, A2C, DQN, MaskablePPO)
    - Hyperparameter handling
    - SumoEnvironment creation
    - Simulation file management
    """

    def __init__(
        self,
        name: str = "group_test1",
        group: str = "test",
        base_config: dict[str, Any] | None = None,
        config_overwrite: dict[str, Any] | None = None,
        experiment_type: Literal["intersection_zoo", None] = None,
        callbacks: Sequence[Literal["wandb", "optuna_prune"]] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        stable_baselines_model: Literal["PPO", "A2C", "DQN", "MaskablePPO"] | Callable = PPO,
        name_suffix: str = "",
        wandb_tags: list[str] | None = None,
    ) -> None:
        base_config = {} if base_config is None else dict(base_config)
        callbacks = ["wandb"] if callbacks is None else list(callbacks)
        base_config["wandb"] = "wandb" in callbacks

        # Initialize base class
        super().__init__(
            name=name,
            group=group,
            base_config=base_config,
            config_overwrite=config_overwrite,
            experiment_type=experiment_type,
            callbacks=callbacks,
            wandb_tags=wandb_tags,
        )

        # SB3-specific initialization
        hyperparameters = {} if hyperparameters is None else dict(hyperparameters)
        self.name_suffix = name_suffix
        self.hyperparameters = hyperparameters
        self.stable_baselines_model = stable_baselines_model
        self.modded_add_xml_file_sufix = config.modded_add_xml_file_sufix

        self._set_global_random_seed()

        # Validation
        assert "simulation" in self.sim_config or self.experiment_type == "intersection_zoo", "simulation must be in the sim_config unless experiment_type is set to intersection_zoo"
        assert experiment_type is None or experiment_type == "intersection_zoo", "experiment_type must be None or should use intersection_zoo datasets"
        assert "simulation" not in self.sim_config or self.sim_config["simulation"] in ["INGOLSTADT_TINY", "MULTIMODAL_SIMULATION_SMALL", "MULTIMODAL_SIMULATION", "TRENDS"], "simulation must be one of ['INGOLSTADT_TINY','MULTIMODAL_SIMULATION_SMALL','MULTIMODAL_SIMULATION','TRENDS']"

        # Set experiment name
        if "experiment_name" not in self.sim_config:
            self.sim_config["experiment_name"] = name + name_suffix
        else:
            self.sim_config["experiment_name"] = self.sim_config["experiment_name"] + name_suffix

        self._create_log_dir_and_add_to_sim_config()

        # Setup hyperparameters and wandb
        if "policy" not in self.hyperparameters:
            self.hyperparameters["policy"] = "MlpPolicy"
        if self.sim_config.get("wandb"):
            self.wandb_run = self._init_wandb_run(
                wandb_config=self.sim_config | self.hyperparameters,
                experiment_name=self.sim_config["experiment_name"],
                sync_tensorboard=True,
            )
            if self.hyperparameters is not None:
                self.hyperparameters["tensorboard_log"] = f"{self.sim_config['log_dir']}/runs/{self.wandb_run.id}"

        self._add_defaults_to_sim_config()

        # Setup scenarios or simulation files
        if experiment_type == "intersection_zoo":
            if "scenarios" not in self.sim_config:
                self.sim_config["scenarios"] = self._init_intersection_zoo_scenarios()
        else:
            self.sim_dir_path = getattr(config, self.sim_config["simulation"])
            self._add_sim_files_to_sim_config_dict()

        self._persist_experiment_config_snapshot()
        self._persist_generic_provenance()

    @staticmethod
    def _safe_model_filename(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"

    def _resolve_model_output_dir(self) -> Path | None:
        model_output_dir = self.sim_config.get("model_output_dir")
        if not isinstance(model_output_dir, str) or len(model_output_dir.strip()) == 0:
            return None
        output_dir = Path(model_output_dir)
        if not output_dir.is_absolute():
            output_dir = Path(config.ROOT_PATH) / output_dir
        return output_dir

    def _save_final_model_to_output_dir(self) -> str | None:
        output_dir = self._resolve_model_output_dir()
        if output_dir is None:
            return None

        save_fn = getattr(self.model, "save", None)
        if not callable(save_fn):
            warnings.warn(
                f"Model type {type(self.model).__name__} has no save() method. "
                "Skipping local model output.",
                stacklevel=2,
            )
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        filename = self.sim_config.get("model_output_name")
        if not isinstance(filename, str) or len(filename.strip()) == 0:
            filename = f"{self.group}_{self.sim_config.get('experiment_name', self.name)}"
        model_path = output_dir / self._safe_model_filename(filename)
        save_fn(str(model_path))
        saved_path = model_path.with_suffix(".zip")
        if not saved_path.exists() and model_path.exists():
            saved_path = model_path
        print(f"Saved final RL model to {saved_path}")
        return str(saved_path)

    def _add_sim_files_to_sim_config_dict(self):
        """
        Adds paths to the simulation files to the sim_config dict

        The simulation files are:#
            - net file
            - route files
            - additional files
            - config file
        """

        sim_files = os.listdir(self.sim_dir_path)

        # net file
        net_files = [s for s in sim_files if ".net.xml" in s]
        assert len(net_files) > 0, "No net files found in the simulation folder."
        assert len(net_files) == 1, "Multiple net files found in the simulation folder."
        net_file = self.sim_dir_path + net_files[0]
        if "net_file" not in self.sim_config:
            self.sim_config["net_file"] = net_file

        # route files
        if "route_file" not in self.sim_config:
            route_files = [self.sim_dir_path + s for s in sim_files if ".rou.xml" in s]
            self.sim_config["route_file"] = route_files

        # flow files
        if "flow_file" not in self.sim_config:
            flow_files = [self.sim_dir_path + s for s in sim_files if ".flow.xml" in s]
            self.sim_config["flow_file"] = ",".join(flow_files)

        # additional files
        if "additional_files" not in self.sim_config:
            additional_files = ",".join([self.sim_dir_path + s for s in sim_files if ".add.xml" in s and (s.replace(".add.xml", f"{self.modded_add_xml_file_sufix}.add.xml") not in sim_files or self.modded_add_xml_file_sufix in s)])
            self.sim_config["additional_files"] = additional_files

        # config file
        config_file_path = self.sim_dir_path + "parameters.json"
        if "config_file_path" not in self.sim_config:
            self.sim_config["config_file_path"] = config_file_path


    def _build_experiment_snapshot(self) -> dict[str, object]:
        model_name: str
        if isinstance(self.stable_baselines_model, str):
            model_name = self.stable_baselines_model
        else:
            model_name = getattr(
                self.stable_baselines_model, "__name__", str(self.stable_baselines_model)
            )

        return {
            "name": self.name,
            "group": self.group,
            "experiment_type": self.experiment_type,
            "callbacks": list(self.callbacks),
            "wandb_tags": list(self.wandb_tags),
            "name_suffix": self.name_suffix,
            "stable_baselines_model": model_name,
            "base_config_input": self.base_config,
            "config_overrides_input": self.config_overwrite,
            "resolved_hyperparameters": self.hyperparameters,
            "resolved_sim_config": self.sim_config,
        }

    def _persist_experiment_config_snapshot(self) -> None:
        snapshot = self._to_plain_data(self._build_experiment_snapshot())

        output_dirs: list[Path] = []
        log_dir = self.sim_config.get("log_dir")
        if isinstance(log_dir, str) and len(log_dir) > 0:
            output_dirs.append(Path(log_dir))

        wandb_run = getattr(self, "wandb_run", None)
        run_dir = getattr(wandb_run, "dir", None) if wandb_run is not None else None
        if isinstance(run_dir, str) and len(run_dir) > 0:
            output_dirs.append(Path(run_dir))

        for out_dir in output_dirs:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                json_path = out_dir / "resolved_experiment_config.json"
                yaml_path = out_dir / "resolved_experiment_config.yaml"
                json_path.write_text(
                    json.dumps(snapshot, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                yaml_path.write_text(
                    yaml.safe_dump(snapshot, sort_keys=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                warnings.warn(f"Failed to persist experiment config snapshot in {out_dir}: {exc}", stacklevel=2)

    def _get_env(self, **kwargs) -> Any:
        if self.stable_baselines_model == "MaskablePPO":
            return AttrPassActionMasker(SumoEnvironment(**kwargs), self.mask_fn)
        return SumoEnvironment(**kwargs)

    def get_env(self, **kwargs) -> Any:
        return self._get_env(**kwargs)

    def _get_env_inputs(self) -> Any:
        env_input = {k: v for k, v in self.sim_config.items() if k in inspect.signature(SumoEnvironment).parameters}
        env_input["out_history_name"] = f"{self.sim_config['log_dir']}/history/"
        return env_input

    def _get_model(self, env: Any) -> OnPolicyAlgorithm:
        """
        Given an stablebaselines model Callable, returns the model object with the hyperparameters set in the experiment object
        """
        if isinstance(self.stable_baselines_model, OnPolicyAlgorithm):
            model_factory: Any = self.stable_baselines_model
        elif isinstance(self.stable_baselines_model, str):
            model_factory = model_map.get(self.stable_baselines_model)
            if model_factory is None:
                raise ValueError(f"Unknown stable baselines model '{self.stable_baselines_model}'.")
        else:
            raise ValueError(
                f"Invalid model type: {type(self.stable_baselines_model)}. Must be a string or a callable."
            )

        if self.hyperparameters is None:
            return cast(OnPolicyAlgorithm, model_factory(env=env, policy="MlpPolicy", verbose=0))

        # make sure that the correct information was passed to the model
        invalid_keys = set(self.hyperparameters.keys()) - set(
            inspect.signature(cast(Callable[..., Any], model_factory)).parameters.keys()
        )
        assert not invalid_keys, f"Invalid keys found in the hyperparameters: {invalid_keys}"
        if "policy" not in self.hyperparameters:
            self.hyperparameters["policy"] = "MlpPolicy"
        return cast(OnPolicyAlgorithm, model_factory(env=env, verbose=0, **self.hyperparameters))

    @abstractmethod
    def run(self) -> Any:
        """
        Run the experiment
        """
        pass

    @staticmethod
    def mask_fn(env: Any) -> np.ndarray:
        n = env.action_space.n
        mask = np.zeros(n, dtype=bool)

        if env.traffic_signals[env.tls_id].can_change_phase:  # you define/maintain this flag
            mask[:] = True  # or restrict to allowed transitions
        else:
            mask[env.traffic_signals[env.tls_id].current_phase] = True

        return mask


class train_experiment(experiment):
    def __init__(
        self,
        name: str = "group_test1",  # name of the experiment
        group: str = "test",
        experiment_type: Literal["intersection_zoo", None ] = None,  # this is used to specify the type of experiment, currently only intersection_zoo and None - meaning default
        base_config: dict[str, Any] | None = None,  # experiemnts in the group all share this config
        config_overwrite: dict[str, Any] | None = None,  # these config values override the sim_config for each experiment in the group
        hyperparameters: dict[str, Any] | None = None,  # hyperparameters for the model - None gives default values
        stable_baselines_model: Literal["PPO", "A2C", "DQN", "MaskablePPO"] | Callable = PPO,  # stable baselines model object # type: ignore
        callbacks: list[Literal["wandb", "optuna_prune"]] | None = None,
        name_suffix: str = "",  # this can be used to add e.g. a number to the name of the run if an experiment with the same conmfig is run multiple times
        wandb_tags: list[str] | None = None,
        optuna_trial: None | optuna.Trial = None,  # if this is not None, the experiment will be run with optuna pruning
    ) -> None:

        super().__init__(
            name=name,
            group=group,
            experiment_type=experiment_type,
            base_config=base_config,
            config_overwrite=config_overwrite,
            stable_baselines_model=stable_baselines_model,
            hyperparameters=hyperparameters,
            callbacks=callbacks,
            name_suffix=name_suffix,
            wandb_tags=wandb_tags,
        )

        self.sim_config["wandb"] = "wandb" in self.callbacks
        assert self.sim_config["wandb"], "train_experiment currently requires wandb logging. Include 'wandb' in callbacks; current callback/env wiring relies on self.wandb_run being initialized."

        self.callbacks_list: list[Any] = []

        self.train_env_input = self._get_env_inputs()
        self.train_env = self._build_train_vec_env(self.train_env_input)

        if "optuna_prune" in self.callbacks:
            assert optuna_trial is not None, "optuna_trial must be set if 'optuna_prune' is in the callbacks"
            self.optuna_trial = optuna_trial

        """
        Init the model
        """
        self.model = self._get_model(self.train_env)

        # reinitializing the environment because the basealgorithm code inside the model resets the environment seed to the agent seed
        # this work around will ensure the environment uses the random seed from the sim_config
        train_env_again = self._build_train_vec_env(self.train_env_input)
        self.model.set_env(train_env_again)
        self.train_env.close()
        self.train_env = train_env_again

    def _get_env_inputs(self) -> dict[str, Any]:
        env_input = super()._get_env_inputs()
        return {k.replace("train_", "", 1): v for k, v in env_input.items() if "eval" not in k}

    def _make_env_fn(self, env_input: dict[str, Any], rank: int):
        use_masker = self.stable_baselines_model == "MaskablePPO"
        mask_fn = experiment.mask_fn

        def _init():
            env_kwargs = dict(env_input)
            if isinstance(env_kwargs.get("random_seed"), int):
                env_kwargs["random_seed"] = env_kwargs["random_seed"] + rank
            if use_masker:
                return AttrPassActionMasker(SumoEnvironment(**env_kwargs), mask_fn)
            return SumoEnvironment(**env_kwargs)

        return _init

    def _build_train_vec_env(self, env_input: dict[str, Any]) -> Any:
        n_vec_envs = int(self.sim_config.get("n_vec_envs", 1))
        if n_vec_envs <= 1:
            return self._get_env(**env_input)

        vec_env_type = self.sim_config.get("vec_env_type", "subproc")
        env_fns = [self._make_env_fn(env_input, i) for i in range(n_vec_envs)]
        if vec_env_type == "dummy":
            venv = DummyVecEnv(env_fns)  # type: ignore
        else:
            venv = SubprocVecEnv(env_fns)  # type: ignore
        return VecMonitor(venv)

    def run(self) -> None:

        callbacks = get_callbacks(
            callbacks=self.callbacks,
            optuna_trial=self.optuna_trial if "optuna_prune" in self.callbacks else None,
            config=self.sim_config,
            wandb_run=self.wandb_run,
        )

        try:
            self.model.learn(
                total_timesteps=self.sim_config["total_timesteps"],
                callback=callbacks,
                progress_bar=False,
            )
            self._save_final_model_to_output_dir()
        finally:
            try:
                final_env_history = extract_env_history_from_candidate(self.train_env)
                if isinstance(final_env_history, list) and len(final_env_history) > 0:
                    self.final_env_history_snapshot = list(final_env_history)
                    history_path = Path(str(self.sim_config["log_dir"])) / "final_env_history.pkl"
                    history_path.parent.mkdir(parents=True, exist_ok=True)
                    with history_path.open("wb") as handle:
                        pickle.dump(self.final_env_history_snapshot, handle)
                    self.final_env_history_path = str(history_path)
            except Exception:
                pass

            if self.sim_config.get("wandb", False) and self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass

            for env_name in ("train_env",):
                env = getattr(self, env_name, None)
                if env is None:
                    continue
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    env.cleanup()  # if your env implements it
                except Exception:
                    pass


class eval_experiemnt(experiment):
    def __init__(
        self,
        model_path: str | None = None,  # path to the model
        name: str = "group_test1",  # name of the experiment
        group: str = "test",
        experiment_type: Literal["intersection_zoo", None] = None,
        base_config: dict[str, Any] | None = None,  # experiemnts in the group all share this config
        config_overwrite: dict[str, Any] | None = None,  # these config values override the sim_config for each experiment in the group
        name_suffix: str = "",  # this can be used to add e.g. a number to the name of the run if an experiment with the same conmfig is run multiple times
        wandb_tags: list[str] | None = None,
    ) -> None:

        super().__init__(
            name=name,
            group=group,
            experiment_type=experiment_type,
            base_config=base_config,
            config_overwrite=config_overwrite,
            name_suffix=name_suffix,
            wandb_tags=wandb_tags,
        )

        assert "model_path" in self.sim_config or model_path is not None, "model_path must be in the sim_config or given as a parameter"
        assert "deterministic" in self.sim_config, "deterministic must be in the sim_config"

        self.env_input: dict[str, Any] = self._get_env_inputs()
        self.env: Any = self._get_env(**self.env_input)
        self.deterministic = bool(self.sim_config["deterministic"])

        self.episode_push_history: dict[str, list[float]] = defaultdict(list)
        self.last_push: dict = {}
        self.episode_info: dict[str, str | float | list[float]] = {}

        """
        Model related parameters
        """
        self.model_path: str | None = model_path if model_path is not None else self.sim_config["model_path"]
        if self.model_path is None:
            self.model: OnPolicyAlgorithm | None = None
        else:
            self.model = self._get_model_stable_baselines(self.model_path)

    @staticmethod
    def _is_controller_only_eval(model_path: str | None) -> bool:
        if model_path is None:
            return True
        return str(model_path).strip().lower() in {"fixed_ts", "actuated_ts"}

    def _get_model_stable_baselines(self, model_path: str) -> OnPolicyAlgorithm:
        if self._is_controller_only_eval(model_path):
            raise ValueError("Controller-only evals should not call _get_model_stable_baselines().")
        if not str(model_path).lower().endswith(".zip"):
            raise ValueError(
                f"Unsupported evaluation model path '{model_path}'. Only Stable Baselines '.zip' checkpoints are supported."
            )

        model_class = model_map.get(self.sim_config["stable_baselines_model"])
        if model_class is None:
            raise ValueError(
                f"Unknown stable baselines model '{self.sim_config['stable_baselines_model']}'."
            )
        return cast(Any, model_class).load(model_path, env=self.env)

    def _evaluate_model(self) -> None:
        """
        Runs the evaluation loop for the loaded model.
        Executes a specified number of episodes and logs rewards to Weights & Biases if enabled.
        """
        total_episodes = self.sim_config.get("episodes", 10)

        for _episode in range(total_episodes):
            obs, _ = self.env.reset()
            truncated = False
            episode_reward = 0.0

            while not truncated:
                obs = np.array(obs)
                action: int | np.ndarray | None
                # Use the stable baselines 'predict' method if available.
                if self.model is None:
                    if bool(self.sim_config.get("fixed_ts", False)) or bool(
                        self.sim_config.get("actuated_ts", False)
                    ):
                        action = None
                    else:
                        tls_fallback = self.env.tls_id if getattr(self.env, "tls_id", None) in self.env.traffic_signals else next(iter(self.env.traffic_signals.keys()))
                        action = int(getattr(self.env.traffic_signals[tls_fallback], "green_phase", 0))
                else:
                    assert isinstance(self.model, OnPolicyAlgorithm), "Model must be a stable baselines model."
                    predictor = cast(Any, self.model)
                    if self.sim_config.get("stable_baselines_model") == "MaskablePPO" and hasattr(
                        self.env, "action_masks"
                    ):
                        action_masks = self.env.action_masks()  # type: ignore[attr-defined]
                        action, _ = predictor.predict(
                            obs,
                            deterministic=self.deterministic,
                            action_masks=action_masks,
                        )
                    else:
                        action, _ = predictor.predict(obs, deterministic=self.deterministic)

                assert isinstance(action, (int, np.ndarray)) or action is None, f"action must be an int or None incase of a Fixed or actuated controller \ninstead action was {action} type: {type(action)}"
                obs, reward, _, truncated, _ = self.env.step(action)
                assert isinstance(reward, float), "reward must be an int or float"
                episode_reward += reward

                self._log_on_step(truncated=truncated)

    def _log_on_step(self, truncated: bool = False) -> None:
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            current_push: dict[str, Any] = {f"ts/{k}": v for k, v in self.env.get_numerical_info_dict().items()}
            current_push.update(
                {f"ts_bus/{k}": v for k, v in self.env.get_numerical_info_dict_bus().items()}
            )
            current_push.update(
                {f"ppd/{k}": v for k, v in self.env.get_all_reward_metrics().items()}
            )

        push_on_change = {k: v for k, v in current_push.items() if v != self.last_push.get(k, None)}
        if len(push_on_change) > 0:
            wandb.log(push_on_change)

        for k, v in current_push.items():
            self.episode_push_history[f"episode/{k.split('/')[1]}"].append(v)

        with warnings.catch_warnings(action="ignore", category=UserWarning):
            for k, v in self.env.get_mean_accumulated_waiting_time_per_lane().items():
                self.episode_push_history[f"episode_lane/{k}"].append(v)

        # check if the episode is done
        if truncated:
            with warnings.catch_warnings(action="ignore", category=UserWarning):
                episode_info = cast(dict[str, str | float | list[float]], self.env.get_episode_info())
                push_episode_history(
                    episode_push_history=self.episode_push_history,
                    episode_info=episode_info,
                )
                save_trajectory(
                    trajectories=self.env.trajectory,
                    ep=current_push["ts/episode"],
                )

            self.episode_push_history = defaultdict(list)

        self.last_push = current_push

        return None

    def run(self) -> None:
        try:
            self._evaluate_model()
        finally:
            if self.sim_config.get("wandb", False) and self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass
            try:
                self.env.close()
            except Exception:
                pass


class parallel_experiments_yaml:
    def __init__(
        self,
        yaml_path: str = config.ROOT_PATH + "/morl_tsp/experiment/yaml/PPO_experiment1.yaml",
        path_to_py_file: str = config.ROOT_PATH + "/morl_tsp/experiment/run_experiment.py",
        index_list: list[int] | None = None,
        debug: bool = False,
    ):

        self.yaml_path = yaml_path
        self.path_to_py_file = path_to_py_file
        self.index_list = index_list
        self.debug = debug

        with open(yaml_path) as file:
            self.yaml_experiment_input = yaml.load(file, Loader=yaml.FullLoader)
        # filters the number of experiments to run based on the index_list
        if index_list is not None:
            self.yaml_experiment_input["config_overwrite"] = [ex for idx, ex in enumerate(self.yaml_experiment_input["config_overwrite"]) if idx in index_list]

        self._assert_yaml_experiment_config()

        self.experiments = self._get_experiment_list()

    def _assert_yaml_experiment_config(self) -> None:
        d = self.yaml_experiment_input
        assert d["workers"] % d["n_parallel"] == 0, f"workers must be divisible by n_parallel ({d['workers']} % {d['n_parallel']} = {d['workers'] % d['n_parallel']} != 0)"
        assert "experiment_type" not in self.yaml_experiment_input or self.yaml_experiment_input["experiment_type"] is None or self.yaml_experiment_input["experiment_type"] == "intersection_zoo", "experiment_type must be None or should use intersection_zoo datasets"

    def _get_experiment_list(self) -> list[list[str]]:
        length = len(self.yaml_experiment_input["config_overwrite"])
        python_exe = sys.executable
        target_indices = self.index_list if self.index_list is not None else list(range(length))
        commands: list[list[str]] = []
        for i in target_indices:
            for j in range(self.yaml_experiment_input["n_parallel"]):
                commands.append([python_exe, self.path_to_py_file, "--nth_run", str(j), "--yaml_path", self.yaml_path, "--experiment_idx", str(i)])
        return commands

    def run_experiments(self):
        failures: list[tuple[list[str], Exception]] = []
        # Create a thread pool executor with a maximum number of workers
        with ThreadPoolExecutor(max_workers=self.yaml_experiment_input["workers"]) as executor:
            # A list to keep track of all submitted tasks
            future_to_experiment = {
                executor.submit(subprocess.run, experiment_str, check=True): experiment_str
                for experiment_str in self.experiments
            }

            # Wait for each task to complete
            for future in as_completed(future_to_experiment):
                i = future_to_experiment[future]
                try:
                    # This will raise an exception if the task had an error
                    future.result()
                except Exception as exc:
                    if self.debug:
                        raise (exc)
                    failures.append((i, exc))
                    print(f"Experiment {i} generated an exception: {exc}")
                else:
                    print(f"Experiment {i} completed successfully.")

        if failures:
            failure_lines = [f"{cmd}: {exc}" for cmd, exc in failures]
            raise RuntimeError(
                "One or more experiments failed:\n" + "\n".join(failure_lines)
            )


"""
YAML Structure

The YAML file is composed of several keys, each defining different aspects of the experimental setup:

    group (Required)
        Type: str
        Description: The name for the group of experiments to be run. This is used to group together related experiments for easier identification and management.
        Example: test_PPO_tiny

    simulation (Required)
        Type: Literal: INGOLSTADT_TINY,MULTIMODAL_SIMULATION_SMALL,MULTIMODAL_SIMULATION, TRENDS
        Description: The name of the simulation folder within the morl_tsp/simulations directory. This specifies which simulation environment to use.
        Example: INGOLSTADT_TINY

    stable_baselines_model (Required)
        Type: Literal: PPO,A2C,DQN
        Description: The name of the stable baselines model to use for training. This must be a valid model name compatible with Stable Baselines.
        Example: PPO

    callbacks (Required)
        Type: List[Literal["wandb", "optuna_prune"]]
        Default: ["wandb"]
        Description: A list of callbacks to use during the training process.
        Example: ["wandb"]

    workers (Required)
        Type: Literal[2, 4, 8, 10, 12, 14, 16]
        Description: The number of workers (CPU cores) to use for running the simulations. It must be one of the allowed values.
        Example: 2

    n_parallel (Required)
        Type: int
        Description: The number of parallel simulations to run with the same parameters. This allows for more efficient experimentation by leveraging parallelism.
        Example: 2

    config_overwrite (Required)
        Type: dict
        Description: This section defines a dictionary that contains parameters and their corresponding values for each experiment to run. Each sub-key represents a different experimental variable, and the values are indexed to align with specific experiment setups. Here are the sub-keys:
        Sub-keys:
            names (dict of str): The name assigned to each experiment setup.
            time_of_day (dict of bool): Include the time of day in the state of the agent.

    hyperparameters (Required)
        Type: dict
        Description: A dictionary of hyperparameters used for the reinforcement learning model. These parameters are essential for configuring the learning algorithm.
        Fields:
            batch_size (int): Batch size for training.
            ent_coef (float): Entropy coefficient for the loss calculation.
            gae_lambda (float): Generalized Advantage Estimation lambda.
            gamma (float): Discount factor.
            learning_rate (float): Learning rate for the optimizer.
            policy (str): Policy architecture to use.
            stats_window_size (int): Window size for logging statistics.
            verbose (int): Verbosity level.
            vf_coef (float): Value function coefficient for the loss calculation.

    sim_config (Required)
        Type: dict
        Description: Contains configuration parameters specific to the simulation environment.
        Fields:
            delta_time (int): Time delta between simulation steps.
            deterministic (bool): Whether the simulation runs deterministically.
            gradient_save_freq (int): Frequency of saving model gradients.
            num_seconds (int): Total time duration for each simulation.
            random_seed (Optional[int]): Seed for random start time; null if not used.
            reward_fn (str): Reward function to use.
            run_up_time (int): Warm-up time for the simulation.
            time_to_teleport (int): Time to teleport vehicles in the simulation.
            total_timesteps (int): Total timesteps for the training.
            use_config_lanes (bool): Whether to use configured lanes.
            wandb (bool): Whether to use Weights and Biases for experiment tracking.

"""
