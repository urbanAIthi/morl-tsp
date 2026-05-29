# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""
This file is a helper script that runs a single experiment from a set of experiments that are defined by a yaml file.

This script should be started from the script: scripts/run_experiments.py
"""

import argparse
import inspect
import logging
import os
import sys
import traceback
import warnings

import yaml

from morl_tsp.experiment.experiment import eval_experiemnt, train_experiment
from morl_tsp.experiment.morl_experiment import train_morl_experiment

logging.getLogger().setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*env\\.trajectory.*")

parser = argparse.ArgumentParser(
    description="Run an experiment given a yaml file and the index of the experiment to run"
)
parser.add_argument(
    "--yaml_path", type=str, help="Path to the yaml file that contains the experiment configuration"
)
parser.add_argument("--experiment_idx", type=int, help="The index of the experiment to run")
parser.add_argument("--nth_run", type=int, help="The nth run with the same config", default=0)

args = parser.parse_args()

if __name__ == "__main__":
    try:
        with open(args.yaml_path) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        data["config_overwrite"] = data["config_overwrite"][args.experiment_idx]

        data["name"] = data["config_overwrite"]["name"]

        data["model_hyperparameters"] = data["hyperparameters"]

        data["wandb_tags"] = ["experiment"]

        print("\nThe data is :\n", data)

        if "morl_algorithm" in data["config_overwrite"] or "morl_algorithm" in data["base_config"]:
            experiment_class = train_morl_experiment
        elif "model_path" in data["config_overwrite"] or "model_path" in data["base_config"]:
            experiment_class = eval_experiemnt
        else:
            experiment_class = train_experiment

        # filter the data dict to only include the parameters that are required by the experiment
        init_params = [
            param_name
            for param_name in inspect.signature(experiment_class.__init__).parameters
            if param_name != "self"
        ]  # Exclude 'self' parameter
        data = {k: v for k, v in data.items() if k in init_params}
        print(f"Running {experiment_class.__name__}: {data}")

        ex = experiment_class(**data)
        ex.run()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
