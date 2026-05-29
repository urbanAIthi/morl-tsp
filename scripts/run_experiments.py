# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
import sys
import traceback

from morl_tsp.experiment.experiment import config, parallel_experiments_yaml

parser = argparse.ArgumentParser(description="Run a set of experiments given a yaml file")

parser.add_argument(
    "--yaml_path",
    type=str,
    help="Path to the yaml file that contains the experiment configuration",
    default=f"{config.ROOT_PATH}/morl_tsp/experiment/yaml/test1.yaml",
)
parser.add_argument(
    "--logging_path",
    type=str,
    help="Path to the logging directory",
    default=f"{config.ROOT_PATH}/logs",
)
parser.add_argument(
    "--index_list",
    type=int,
    nargs="+",
    help="List of the experiments that should be run defined by there index",
    default=None,
)
parser.add_argument(
    "--runner",
    default=f"{config.ROOT_PATH}/morl_tsp/experiment/run_experiment.py",
    help="Python experiment-builder file used for each YAML entry.",
)

args = parser.parse_args()

if __name__ == "__main__":
    try:
        print(f"Running experiments from yaml file: {args.yaml_path}")
        # Keep YAML orchestration generic so paper eval YAMLs can use the existing MORL eval builder.
        ex = parallel_experiments_yaml(
            yaml_path=args.yaml_path, path_to_py_file=args.runner, index_list=args.index_list
        )
        ex.run_experiments()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
