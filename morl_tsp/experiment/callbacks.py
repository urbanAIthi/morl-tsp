# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

import os
import pickle
import warnings
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import optuna
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from wandb.integration.sb3 import WandbCallback

import wandb
from morl_tsp.environment.typing import trajectory_type


def push_episode_history(
    episode_push_history: dict[str, list[float]],
    episode_info: dict[str, str | float | list[float]],
    prefix: str = "",
    drop_str: None | str | list[str] = None,
) -> None:
    """
    push the min,max and mean of the episode history to wandb
    """
    stats = {}
    for k, values in episode_push_history.items():
        if any(
            [True if s in k.split("/")[1] else False for s in ["episode", "phase"]]
        ):  # skip these
            continue
        values = [v for v in values if v is not None]
        if len(values) == 0:
            continue
        stats[f"{prefix}{k}_mean"] = sum(values) / len(values)
        if any([True if s in k else False for s in ["episode_lane"]]):  # skip these
            continue
        stats[f"{prefix}{k}_max"] = max(values)
        stats[f"{prefix}{k}_min"] = min(values)
    log_to_wandb(stats, drop_str=drop_str)
    log_to_wandb(
        {f"{prefix}episode_info/{k}": v for k, v in episode_info.items()}, drop_str=drop_str
    )


def save_trajectory(
    trajectories: trajectory_type,
    ep: int | float,
    prefix: str = "",
) -> None:
    run = wandb.run
    if run is None or getattr(run, "dir", None) is None:
        return
    run_dir = run.dir
    file_name = f"{prefix}trajectories_{ep - 1}"
    trajectory_path = os.path.join(run_dir, f"{file_name}.pkl")
    with open(trajectory_path, "wb") as f:
        pickle.dump(trajectories, f)


def log_to_wandb(d: dict[str, Any], drop_str: None | str | list[str] = None) -> None:
    """
    log a dictionary to wandb
    """
    if wandb.run is None:
        return
    if isinstance(drop_str, str):
        drop_str = [drop_str]
    try:
        wandb.log(
            {
                k: v
                for k, v in d.items()
                if drop_str is None or all(ds.lower() not in k.lower() for ds in drop_str)
            }
        )
    except Exception as e:
        print("Error logging to wandb:", e)
        print(
            {
                k: (v, type(v))
                for k, v in d.items()
                if drop_str is None or all(ds.lower() not in k.lower() for ds in drop_str)
            }
        )
        raise e


class SumoRLTrainingCallback(BaseCallback):
    def __init__(
        self,
        train_prefix: str = "",
        drop_str: None | str | list[str] = None,
        log_every_n_steps: int = 1,
        save_trajectories: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.train_prefix = train_prefix
        self.drop_str = drop_str
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.save_trajectories = bool(save_trajectories)
        self._log_step_counter = 0
        self.train_episode_push_history: dict[str, list[float]] = defaultdict(list)
        self.last_push: dict = {}

    def _log_train_ts(self) -> None:
        # this warning is thrown by wandb, because of stable baselines but it is not relevant for this use case
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            current_push = {
                f"{self.train_prefix}ts/{k}": v
                for k, v in self.model.env.env_method("get_numerical_info_dict")[0].items() # type: ignore
            }
            current_push.update(
                {
                    f"{self.train_prefix}ts_bus/{k}": v
                    for k, v in self.model.env.env_method("get_numerical_info_dict_bus")[0].items() # type: ignore
                }
            )
            current_push.update(
                {
                    f"{self.train_prefix}rewards/{k}": v
                    for k, v in self.model.env.env_method("get_all_reward_metrics")[0].items() # type: ignore
                }
            )

        push_on_change = {k: v for k, v in current_push.items() if v != self.last_push.get(k, None)}
        if len(push_on_change) > 0:
            log_to_wandb(push_on_change, drop_str=self.drop_str)

        for k, v in current_push.items():
            self.train_episode_push_history[f"episode/{k.split('/')[1]}"].append(v)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            {
                self.train_episode_push_history[f"episode_lane/{k}"].append(v)
                for k, v in self.model.env.env_method("get_mean_accumulated_waiting_time_per_lane")[ # type: ignore
                    0
                ].items()
            }

        # check if the episode is done
        if self.last_push and self.last_push["ts/episode"] != current_push["ts/episode"]:
            push_episode_history(
                episode_push_history=self.train_episode_push_history,
                episode_info=self.train_episode_info,
                prefix=self.train_prefix,
            )
            if self.save_trajectories:
                save_trajectory(
                    trajectories=self.trajectories,
                    ep=current_push["ts/episode"],
                    prefix=self.train_prefix,
                )
            self.train_episode_push_history = defaultdict(list)

        self.last_push = current_push
        self.trajectories = self.model.env.get_attr("trajectory")[0] # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.train_episode_info = self.model.env.env_method("get_episode_info")[0] # type: ignore

        return None

    def _on_step(self) -> bool:
        self._log_step_counter += 1
        if self._log_step_counter % self.log_every_n_steps == 0:
            self._log_train_ts()
        return True


class OptunaPruningCallback(BaseCallback):
    def __init__(
        self, trial: optuna.Trial, check_freq: int = 1000, warmup_steps: int = 0, verbose: int = 0
    ):

        super().__init__(verbose)
        self.trial = trial
        self.check_freq = check_freq
        self.warmup_steps = warmup_steps

        self._sum = 0.0
        self._count = 0

    def _on_step(self) -> bool:
        # SB3 provides step rewards here; for VecEnv it's an array
        rewards = self.locals["rewards"]

        r = float(np.mean(rewards))
        # Optionally ignore warmup in the aggregation
        if self.num_timesteps > self.warmup_steps:
            self._sum += r
            self._count += 1

        if self.num_timesteps % self.check_freq != 0:
            return True

        if self._count == 0:
            return True

        intermediate_value = self._sum / self._count

        # Use SB3 timesteps as the Optuna "step" axis.
        self.trial.report(intermediate_value, step=self.num_timesteps)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at timestep={self.num_timesteps}, intermediate_value={intermediate_value:.6f}"
            )

        return True


def get_callbacks(
    callbacks: Sequence[Literal["wandb", "optuna_prune"]],
    config: dict[str, str | int | float | bool] | None,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,  # type: ignore
    optuna_trial: optuna.Trial | None = None,
    train_prefix: str = "",
) -> CallbackList:

    train_log_every_n_steps = int(config.get("rl_wandb_log_every_n_steps", 1)) if config else 1
    if train_log_every_n_steps <= 0:
        raise ValueError("rl_wandb_log_every_n_steps must be >= 1.")
    save_train_trajectories = bool(config.get("rl_save_trajectories", True)) if config else True
    training_callback = SumoRLTrainingCallback(
        train_prefix=train_prefix,
        drop_str=None,  # ["bus","car"] if config is not None and "scenarios" in config else None, # dont push bus data if intersectionZoo is used
        log_every_n_steps=train_log_every_n_steps,
        save_trajectories=save_train_trajectories,
        verbose=0,
    )

    callback_list: list[BaseCallback] = []

    n_vec_envs = int(config.get("n_vec_envs", 1)) if config else 1
    checkpoint_freq_steps = int(config.get("rl_checkpoint_freq_steps", 0)) if config else 0
    checkpoint_freq_callbacks = (
        max(checkpoint_freq_steps // max(n_vec_envs, 1), 1) if checkpoint_freq_steps > 0 else 0
    )
    upload_checkpoints_to_wandb = (
        bool(config.get("rl_checkpoint_upload_to_wandb", False)) if config else False
    )

    if "wandb" in callbacks:
        assert wandb_run is not None, "wandb_run must be provided if wandb is in the callbacks"
        callback_list.append(
            WandbCallback(
                gradient_save_freq=0,
                model_save_freq=checkpoint_freq_callbacks if upload_checkpoints_to_wandb else 0,
                model_save_path=f"{config['log_dir']}/models/{wandb_run.id}",  # type: ignore
                verbose=2,
            )
        )

    if checkpoint_freq_callbacks > 0:
        callback_list.append(
            CheckpointCallback(
                save_freq=checkpoint_freq_callbacks,
                save_path=f"{config['log_dir']}/models/local_checkpoints",  # type: ignore
                name_prefix=f"{config['experiment_name']}_step",  # type: ignore
                save_replay_buffer=False,
                save_vecnormalize=False,
                verbose=1,
            )
        )

    callback_list.append(training_callback)

    if "optuna_prune" in callbacks:
        assert optuna_trial is not None
        callback_list.append(
            OptunaPruningCallback(
                trial=optuna_trial,
                check_freq=1000,
                warmup_steps=config.get("optuna_warmup_steps", 0) if config else 0,  # type: ignore
                verbose=1,
            )
        )

    return CallbackList(callback_list)
