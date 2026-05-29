# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import multiprocessing
import warnings

import yaml

from morl_tsp import config
from morl_tsp.experiment.hp_experiment import ParallelHPExperiments
from morl_tsp.experiment.utils import resolve_hpt_base_config

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*env\\.trajectory.*")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"gymnasium\.core",
    message=r".*env\.trajectory to get variables from other wrappers is deprecated.*",
)

parser = argparse.ArgumentParser(description="Run an experiment given a yaml file and the index of the experiment to run")
parser.add_argument('--yaml_path', required=False, type=str, default=f"{config.ROOT_PATH}/morl_tsp/experiment/yaml/test_hpt.yaml", help='Path to the yaml file that contains the HP experiment configuration')

args = parser.parse_args()

if __name__ == "__main__":
    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    multiprocessing.set_start_method(start_method, force=True)

    with open(args.yaml_path) as file:
        data = yaml.safe_load(file)

    assert isinstance(data.get("database"), dict), "database should be a dictionary."
    assert isinstance(data["database"].get("study_name"), str), "database.study_name should be a string."
    assert isinstance(data.get("hyperparameter_object"), dict), "hyperparameter_object should be a dictionary."
    assert isinstance(data["hyperparameter_object"]["hyperparameter_ranges"], dict), "hyperparameter_ranges should be a dictionary."

    callbacks = data["hyperparameter_object"].get("callbacks", data.get("callbacks", ["wandb"]))
    assert isinstance(callbacks, list), "callbacks must be a list."
    base_config = resolve_hpt_base_config(data)

    parallel_hp_experiments = ParallelHPExperiments(
        hp_tuning_config    = data["hyperparameter_object"]["hyperparameter_ranges"],
        agent_tuning_config = data["hyperparameter_object"].get("agent_parameter_ranges", {}),
        db_study_name       = data["database"]["study_name"],
        db_load_if_exists   = bool(data["database"].get("load_if_exists", True)),
        db_connection_string=data["database"].get("connection_string"),
        base_config         = base_config,
        env_seeds           = data["hyperparameter_object"]["env_seeds"],
        experiment_type     = data.get("experiment_type", None),
        callbacks           = callbacks,
        stable_baselines_model = data["hyperparameter_object"].get(
            "stable_baselines_model",
            data.get("stable_baselines_model", "PPO"),
        ),
        experiment_runner   = data["hyperparameter_object"].get("experiment_runner", "auto"),
        objective_metric    = data["hyperparameter_object"].get("objective_metric", "reward"),
        group               = data["hyperparameter_object"].get("group", ""),
        n_trials            = data["hyperparameter_object"]["n_trials"],
        n_workers           = data["workers"],
        n_startup_trials    = data["hyperparameter_object"]["startup_trials"],
        n_warmup_steps      = data["hyperparameter_object"]["warmup_steps"],
        agent_seed          = data["hyperparameter_object"]["agent_seed"],
        )

    hp_results = parallel_hp_experiments.run()
