# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_ROOT_PATH = Path(__file__).resolve().parents[1]
ROOT_PATH = os.getenv("MORL_TSP_ROOT_PATH", str(_DEFAULT_ROOT_PATH))

MIN_GAP = 2.5
DEFAULT_VEHICLE_LENGTH = 5.0

MAX_NUM_LANES = 12
MAX_NUM_PHASES = 4
MAX_NUM_BUS_PER_LANE_IN_OBS = 2

STATE_RANGE = 250
DEF_OCCUPANCY = 1.2

def _default_intersection_zoo_path() -> str:
    spec = importlib.util.find_spec("intersection_zoo")
    search_locations = list(spec.submodule_search_locations or []) if spec else []
    if search_locations:
        package_path = Path(search_locations[0]).resolve()
        return f"{package_path.parent}/"
    return f"{Path(ROOT_PATH).resolve().parent / 'intersection_zoo'}/"


INTERSECTION_ZOO_PATH = os.getenv(
    "INTERSECTION_ZOO_PATH",
    os.getenv("INTERSECTION_ZOO_ROOT_PATH", _default_intersection_zoo_path()),
)
INTERSECTION_ZOO_DATA_PATH = os.getenv(
    "INTERSECTION_ZOO_DATA_PATH",
    f"{Path(INTERSECTION_ZOO_PATH) / 'dataset'}/",
)
INTERSECTION_ZOO_SEED = 42

EXP_STEP_LENGTH = 1
EXP_NUM_SECONDS = 10800
EXP_RUN_UP_TIME = 1200
EXP_TRAIN_TEST_SPLIT = 0.8
EXP_MORL_DEFAULTS: dict[str, object] = {
    "morl_enable_algorithm_logging": True,
    "morl_save_model": True,
    "morl_checkpoint_freq_steps": 0,
    "morl_checkpoint_upload_to_wandb": False,
    "morl_checkpoint_select_best_by_hv": True,
    "morl_checkpoint_hv_metric": "pareto_hv_nondom",
    "morl_checkpoint_hv_episode_window": 32,
    "morl_checkpoint_hv_nondominated_bonus": 0.01,
    "morl_periodic_eval": False,
    "morl_periodic_eval_every_steps": 0,
    "morl_periodic_eval_episodes": 1,
    "morl_periodic_eval_use_best_hv_checkpoint": True,
    "morl_periodic_eval_eval_log_on_step": False,
    "morl_periodic_eval_on_final": True,
    "morl_periodic_eval_reuse_training_scenarios": False,
    "morl_wandb_log_every_n_steps": 1,
    "morl_save_trajectories": True,
    "morl_auto_build_eval_env": True,
}

REWARD_MIN_CLIP = -5
REWARD_MAX_CLIP = 5
SAFETY_BREAK_THRESHOLD = -2.5

WANDB_PROJECT_NAME = "itsc_2026_morl"
