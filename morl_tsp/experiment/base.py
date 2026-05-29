# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Base experiment class with common functionality for all experiment types.

This module provides BaseExperiment which handles:
- Configuration parsing and merging
- WandB initialization
- IntersectionZoo scenario resolution
- Random seed management
- Log directory creation
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import wandb.sdk

# Lazy imports for optional dependencies
from intersection_zoo.env.config import IntersectionZooEnvConfig
from intersection_zoo.env.task_context import PathTaskContext
from intersection_zoo.sumo_adapter import generate_files

import wandb
from morl_tsp import config
from morl_tsp.experiment.reproducibility import (
    build_run_provenance,
    provenance_summary,
    sha256_file,
)
from morl_tsp.experiment.scenario_generation import (
    BusTimetableConfig,
    ScenarioGenerationWrapper,
    ScenarioGenerationWrapperWithBusTimetable,
    VehicleAttributeConfig,
)
from morl_tsp.experiment.utils import deep_merge_dicts, to_plain_data


class BaseExperiment(ABC):
    """Abstract base class for all experiment types.

    Provides shared functionality for configuration handling, WandB integration,
    IntersectionZoo scenario management, and random seed management.

    Subclasses must implement:
        - run(): Execute the experiment
    """

    def __init__(
        self,
        name: str,
        group: str,
        base_config: dict | None = None,
        config_overwrite: dict | None = None,
        experiment_type: Literal["intersection_zoo", None] = None,
        callbacks: Sequence[Literal["wandb", "optuna_prune"]] | None = None,
        wandb_tags: list[str] | None = None,
    ) -> None:
        """Initialize the base experiment.

        Args:
            name: Name of the experiment run.
            group: Group name for organizing runs.
            base_config: Base configuration dictionary.
            config_overwrite: Overrides to merge on top of base_config.
            experiment_type: Type of experiment ("intersection_zoo" or None).
            callbacks: List of callbacks to enable.
            wandb_tags: Tags for WandB logging.
        """
        base_config = {} if base_config is None else dict(base_config)
        config_overwrite = {} if config_overwrite is None else dict(config_overwrite)
        callbacks = ["wandb"] if callbacks is None else list(callbacks)
        wandb_tags = [] if wandb_tags is None else list(wandb_tags)

        self.name = name
        self.group = group
        self.sim_config = deep_merge_dicts(base_config, config_overwrite)
        self.callbacks = callbacks
        self.experiment_type = experiment_type
        self.wandb_tags = wandb_tags
        self.base_config = dict(base_config)
        self.config_overwrite = dict(config_overwrite)

        self.logging_path = f"{config.ROOT_PATH}/wandb/{self.group}"
        self.wandb_run: wandb.sdk.wandb_run.Run | None = None

    @abstractmethod
    def run(self) -> Any:
        """Run the experiment. Must be implemented by subclasses."""
        pass

    # -------------------------------------------------------------------------
    # Random seed management
    # -------------------------------------------------------------------------
    def _set_global_random_seed(self) -> None:
        """Set global random seeds for reproducibility.

        Sets seeds for:
        - Python random module
        - NumPy random
        - PyTorch (if available), including CUDA and deterministic algorithms
        """
        seed = self.sim_config.get("random_seed")
        if not isinstance(seed, int):
            return

        random.seed(seed)
        np.random.seed(seed)

        # Required for deterministic CUDA kernels in some torch ops.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        try:
            import torch
        except Exception:
            return

        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if "OMP_NUM_THREADS" in os.environ:
                try:
                    torch.set_num_threads(max(1, int(os.environ["OMP_NUM_THREADS"])))
                except Exception:
                    pass
        except Exception as exc:
            warnings.warn(f"Failed to fully configure torch determinism: {exc}", stacklevel=2)

    # -------------------------------------------------------------------------
    # WandB integration
    # -------------------------------------------------------------------------
    def _init_wandb_run(
        self,
        wandb_config: dict,
        experiment_name: str,
        sync_tensorboard: bool = True,
        reinit: bool = False,
        tags: list[str] | None = None,
    ) -> wandb.sdk.wandb_run.Run:
        """Initialize a WandB run with the given configuration.

        Args:
            wandb_config: Configuration dictionary to log.
            experiment_name: Name for the run.
            sync_tensorboard: Whether to sync TensorBoard metrics.
            reinit: Whether to reinitialize if already running.
            tags: Additional tags (merged with self.wandb_tags).

        Returns:
            The initialized WandB run object.
        """
        all_tags = list(self.wandb_tags)
        if tags:
            all_tags.extend(tags)

        wandb_dir_raw = self.sim_config.get("log_dir", self.logging_path)
        wandb_dir = str(wandb_dir_raw) if wandb_dir_raw is not None else str(self.logging_path)
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)

        wandb_run = wandb.init(
            project=config.WANDB_PROJECT_NAME,
            dir=wandb_dir,
            config=wandb_config,
            sync_tensorboard=sync_tensorboard,
            monitor_gym=False,
            save_code=True,
            name=experiment_name,
            group=self.group,
            tags=all_tags if all_tags else None,
            reinit=reinit,
        )
        return wandb_run

    # -------------------------------------------------------------------------
    # Log directory management
    # -------------------------------------------------------------------------
    def _create_log_dir_and_add_to_sim_config(self) -> None:
        """Create a unique log directory and add it to sim_config."""
        if self.logging_path is None:
            warnings.warn(
                "_create_log_dir_and_add_to_sim_config was called but no logging path given.",
                stacklevel=2,
            )
            return

        os.makedirs(self.logging_path, exist_ok=True)

        experiment_name = self.sim_config.get("experiment_name", self.name)
        run_idx = 0
        while True:
            log_dir = f"{self.logging_path}/{experiment_name}_{run_idx}"
            try:
                os.makedirs(log_dir)
                break
            except FileExistsError:
                run_idx += 1

        if "log_dir" not in self.sim_config:
            self.sim_config["log_dir"] = log_dir
        else:
            warnings.warn("log_dir already in sim_config; using existing value", stacklevel=2)

    def _add_defaults_to_sim_config(self) -> None:
        """Add default values to sim_config if not present."""
        if "step_length" not in self.sim_config:
            self.sim_config["step_length"] = config.EXP_STEP_LENGTH
        if "run_up_time" not in self.sim_config:
            self.sim_config["run_up_time"] = config.EXP_RUN_UP_TIME
        if "num_seconds" not in self.sim_config:
            self.sim_config["num_seconds"] = config.EXP_NUM_SECONDS
        if "dataset_split" not in self.sim_config:
            self.sim_config["dataset_split"] = config.EXP_TRAIN_TEST_SPLIT

    # -------------------------------------------------------------------------
    # IntersectionZoo scenario management
    # -------------------------------------------------------------------------
    def _init_intersection_zoo_scenarios(self) -> list[tuple[str, str]]:
        """Initialize IntersectionZoo scenarios for the experiment.

        Returns:
            List of (net_file, route_file) tuples for each scenario.
        """
        assert self.experiment_type == "intersection_zoo", (
            "Should only call _init_intersection_zoo_scenarios for intersection_zoo experiments"
        )
        assert "simulation" not in self.sim_config or self.sim_config["simulation"] is None, (
            "simulation should not be set for intersection_zoo experiments"
        )
        assert self.sim_config.get("log_dir") is not None, (
            "log_dir must be set before calling _init_intersection_zoo_scenarios"
        )

        self.sim_config["tls_id"] = "TL"

        replay_scenarios = self._iz_get_replay_scenarios()
        if replay_scenarios is not None:
            return replay_scenarios

        # Set random seed for scenario sampling
        random.seed(self.sim_config.get("iz_seed", config.INTERSECTION_ZOO_SEED))

        # Calculate scenario count and duration
        scenario_duration_override = self.sim_config.get("iz_scenario_duration_seconds")
        if scenario_duration_override is not None:
            n_scenarios = int(self.sim_config.get("iz_n_scenarios", 1))
            scenario_duration = int(scenario_duration_override)
        else:
            n_scenarios = (
                self.sim_config["iz_n_scenarios"]
                if "iz_n_scenarios" in self.sim_config
                else math.ceil(
                    self.sim_config["total_timesteps"]
                    / (self.sim_config["run_up_time"] + self.sim_config["num_seconds"])
                )
            )
            scenario_duration = math.ceil(self.sim_config["total_timesteps"] / n_scenarios)

        # Handle random start time
        if self.sim_config.get("iz_random_start_time"):
            episode_horizon = int(self.sim_config.get("run_up_time", 0)) + int(
                self.sim_config.get("num_seconds", 0)
            )
            if episode_horizon > 0 and scenario_duration <= episode_horizon:
                warnings.warn(
                    "iz_random_start_time=true with route horizon shorter than one episode.",
                    stacklevel=2,
                )
            self.sim_config["random_start_time"] = (0, scenario_duration)

        # Create output folders
        path_to_route_files = os.path.join(self.sim_config["log_dir"], "sumo")
        os.makedirs(path_to_route_files, exist_ok=True)

        # Get network candidates
        configured_net_rel_paths = self.sim_config.get("iz_net_rel_paths")
        if configured_net_rel_paths:
            if not isinstance(configured_net_rel_paths, list):
                raise ValueError("'iz_net_rel_paths' must be a list[str] when set.")
            net_candidates = [
                Path(config.INTERSECTION_ZOO_DATA_PATH) / path for path in configured_net_rel_paths
            ]
            missing_paths = [str(path) for path in net_candidates if not path.exists()]
            if missing_paths:
                raise FileNotFoundError(f"Configured IZ net path(s) do not exist: {missing_paths}")
        else:
            net_candidates = sorted(Path(config.INTERSECTION_ZOO_DATA_PATH).rglob("net.net.xml"))

        # Select scenarios
        if len(net_candidates) < n_scenarios:
            if configured_net_rel_paths:
                selected_net_paths = [
                    net_candidates[i % len(net_candidates)] for i in range(n_scenarios)
                ]
            else:
                raise ValueError(
                    f"Requested {n_scenarios} IZ scenarios, but only found {len(net_candidates)}."
                )
        else:
            selected_net_paths = random.sample(net_candidates, k=n_scenarios)

        # Setup caching
        use_cache = bool(self.sim_config.get("iz_cache_scenarios", True))
        scenario_cache_root = self._iz_get_scenario_cache_root() if use_cache else None
        if scenario_cache_root is not None:
            os.makedirs(scenario_cache_root, exist_ok=True)

        scenarios: list[tuple[str, str]] = []
        for scenario_idx, iz_net_file_path in enumerate(selected_net_paths):
            city_name = str(iz_net_file_path).split("/")[-3]
            net_file_idx = str(iz_net_file_path).split("/")[-2]
            scenario_suffix = f"_{scenario_idx}" if n_scenarios > 1 else ""
            scenario_path = f"{path_to_route_files}/{city_name}_{net_file_idx}{scenario_suffix}/"
            os.makedirs(scenario_path, exist_ok=True)

            scenario_bus_timetable_overrides = self._iz_get_scenario_bus_timetable_overrides(
                scenario_idx=scenario_idx
            )
            cache_key = self._iz_build_scenario_cache_key(
                net_file_path=iz_net_file_path,
                scenario_duration=scenario_duration,
                scenario_bus_timetable_overrides=scenario_bus_timetable_overrides,
            )

            # Try cache first
            if use_cache and scenario_cache_root is not None:
                cached_scenario = self._iz_try_load_cached_scenario(
                    cache_root=scenario_cache_root,
                    cache_key=cache_key,
                )
                if cached_scenario is not None:
                    cached_net, cached_route = cached_scenario
                    local_sumo_dir = Path(scenario_path) / "sumo"
                    self._iz_materialize_cached_scenario_for_run(
                        local_sumo_dir=local_sumo_dir,
                        cached_net_path=Path(cached_net),
                        cached_route_path=Path(cached_route),
                    )
                    scenarios.append(
                        (
                            str(local_sumo_dir / "net.net.xml"),
                            str(local_sumo_dir / "routes.rou.xml"),
                        )
                    )
                    continue

            # Generate new scenario
            intzoo_task = PathTaskContext(
                dir=iz_net_file_path.parent,
                single_approach=False,
                penetration_rate=self.sim_config.get("electric_penetration_rate", 0),
                temperature_humidity="25_50", # type: ignore
                electric_or_regular="REGULAR", # type: ignore
            ) # type: ignore

            intzoo_config = IntersectionZooEnvConfig(
                working_dir=Path(scenario_path),
                task_context=intzoo_task,
                visualize_sumo=False,
                simulation_duration=scenario_duration + self.sim_config["run_up_time"],
                sim_step_duration=self.sim_config["step_length"],
            )

            self._iz_generate_scenarios(
                intzoo_config=intzoo_config,
                intzoo_task=intzoo_task,
                scenario_bus_timetable_overrides=scenario_bus_timetable_overrides,
            )

            scenario_tuple = (
                f"{scenario_path}/sumo/net.net.xml",
                f"{scenario_path}/sumo/routes.rou.xml",
            )
            scenarios.append(scenario_tuple)

            # Store in cache
            if use_cache and scenario_cache_root is not None:
                self._iz_store_cached_scenario(
                    cache_root=scenario_cache_root,
                    cache_key=cache_key,
                    source_sumo_dir=Path(scenario_path) / "sumo",
                )

        return scenarios

    def _iz_get_scenario_bus_timetable_overrides(
        self, scenario_idx: int | None
    ) -> dict[str, object]:
        """Get per-scenario bus timetable overrides."""
        scenario_bus_timetable_overrides: dict[str, object] = {}
        per_scenario_overrides = self.sim_config.get("iz_bus_timetable_config_per_scenario")
        if (
            scenario_idx is not None
            and isinstance(per_scenario_overrides, list)
            and scenario_idx < len(per_scenario_overrides)
        ):
            override = per_scenario_overrides[scenario_idx]
            if isinstance(override, dict):
                scenario_bus_timetable_overrides = dict(override)
        return scenario_bus_timetable_overrides

    def _iz_get_replay_scenarios(self) -> list[tuple[str, str]] | None:
        """Return explicitly configured SUMO files for YAML route replay.

        ITSC paper evals need to replay exact route/net files for auditability.
        Keeping this in the generic IZ setup lets any YAML experiment bypass
        generation without adding a paper-specific runner.
        """
        raw_scenarios = self.sim_config.get("iz_replay_scenarios")
        if raw_scenarios is None:
            net_file = self.sim_config.get("iz_replay_net_file")
            route_file = self.sim_config.get("iz_replay_route_file")
            if net_file is not None and route_file is not None:
                raw_scenarios = [{"net_file": net_file, "route_file": route_file}]

        if raw_scenarios is None:
            return None
        if not isinstance(raw_scenarios, list) or len(raw_scenarios) == 0:
            raise ValueError("'iz_replay_scenarios' must be a non-empty list when set.")

        scenarios: list[tuple[str, str]] = []
        for idx, entry in enumerate(raw_scenarios):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                net_raw, route_raw = entry
            elif isinstance(entry, dict):
                net_raw = entry.get("net_file", entry.get("net"))
                route_raw = entry.get("route_file", entry.get("route"))
            else:
                raise ValueError(
                    f"Invalid iz_replay_scenarios[{idx}]. Expected dict or [net, route]."
                )
            if net_raw is None or route_raw is None:
                raise ValueError(
                    f"iz_replay_scenarios[{idx}] must include net_file and route_file."
                )
            net_path = self._resolve_replay_path(str(net_raw))
            route_path = self._resolve_replay_path(str(route_raw))
            if not net_path.exists():
                raise FileNotFoundError(f"Replay net_file does not exist: {net_path}")
            if not route_path.exists():
                raise FileNotFoundError(f"Replay route_file does not exist: {route_path}")
            scenarios.append((str(net_path), str(route_path)))
        return scenarios

    @staticmethod
    def _resolve_replay_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        root_path = Path(config.ROOT_PATH) / path
        if root_path.exists():
            return root_path
        iz_path = Path(config.INTERSECTION_ZOO_DATA_PATH) / path
        if iz_path.exists():
            return iz_path
        return root_path

    def _iz_get_scenario_cache_root(self) -> Path:
        """Get the root directory for scenario caching."""
        configured_root = self.sim_config.get(
            "iz_scenario_cache_root",
            f"{config.ROOT_PATH}/wandb/_iz_scenario_cache",
        )
        return Path(str(configured_root)).expanduser().resolve()

    def _iz_build_scenario_cache_key(
        self,
        *,
        net_file_path: Path,
        scenario_duration: int,
        scenario_bus_timetable_overrides: dict[str, object],
    ) -> str:
        """Build a unique cache key for a scenario configuration."""
        generation_seed = int(
            self.sim_config.get("iz_bus_gen_seed", self.sim_config.get("iz_seed", 420))
        )
        cache_payload = {
            "version": 1,
            "net_file_path": str(net_file_path.resolve()),
            "scenario_duration": int(scenario_duration),
            "run_up_time": int(self.sim_config.get("run_up_time", 0)),
            "step_length": float(self.sim_config.get("step_length", 1.0)),
            "generation_seed": generation_seed,
            "iz_bus_multiplier": self.sim_config.get("iz_bus_multiplier"),
            "iz_bus_attribute_config": self.sim_config.get("iz_bus_attribute_config", {}),
            "iz_bus_timetable_config": self.sim_config.get("iz_bus_timetable_config", {}),
            "iz_bus_timetable_config_per_scenario": scenario_bus_timetable_overrides,
        }
        digest_input = json.dumps(
            to_plain_data(cache_payload), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha1(digest_input).hexdigest()
        return digest[:24]

    @staticmethod
    def _iz_try_load_cached_scenario(cache_root: Path, cache_key: str) -> tuple[str, str] | None:
        """Try to load a cached scenario."""
        sumo_dir = cache_root / cache_key / "sumo"
        net_path = sumo_dir / "net.net.xml"
        route_path = sumo_dir / "routes.rou.xml"
        if net_path.exists() and route_path.exists():
            return str(net_path), str(route_path)
        return None

    @staticmethod
    def _iz_materialize_cached_scenario_for_run(
        *,
        local_sumo_dir: Path,
        cached_net_path: Path,
        cached_route_path: Path,
    ) -> None:
        """Materialize a cached scenario for a run (symlink or copy)."""
        del cached_route_path
        local_sumo_dir.mkdir(parents=True, exist_ok=True)
        source_sumo_dir = cached_net_path.parent
        for src in source_sumo_dir.iterdir():
            dst = local_sumo_dir / src.name
            if dst.exists():
                continue
            try:
                os.symlink(src, dst, target_is_directory=src.is_dir())
            except Exception:
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

    @staticmethod
    def _iz_store_cached_scenario(
        *,
        cache_root: Path,
        cache_key: str,
        source_sumo_dir: Path,
    ) -> None:
        """Store a generated scenario in the cache."""
        if not source_sumo_dir.is_dir():
            return

        target_dir = cache_root / cache_key
        target_sumo_dir = target_dir / "sumo"
        if (target_sumo_dir / "net.net.xml").exists() and (
            target_sumo_dir / "routes.rou.xml"
        ).exists():
            return

        tmp_dir = cache_root / f".tmp_{cache_key}_{os.getpid()}"
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_sumo_dir, tmp_dir / "sumo", dirs_exist_ok=True)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(tmp_dir, target_dir)
            except Exception:
                if not target_sumo_dir.exists():
                    shutil.copytree(source_sumo_dir, target_sumo_dir, dirs_exist_ok=True)
        finally:
            if tmp_dir.exists() and tmp_dir != target_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _iz_generate_scenarios(
        self,
        intzoo_config: IntersectionZooEnvConfig,
        intzoo_task: PathTaskContext,
        scenario_bus_timetable_overrides: dict[str, object] | None = None,
    ) -> None:
        """Generate IntersectionZoo scenarios."""
        bus_attribute_config_dict = {
            "occupancy_zero_prob": 0.3,
            "occupancy_lognormal_mean": 2.0,
            "occupancy_lognormal_sigma": 0.5,
            "occupancy_max": 50,
            "schedule_deviation_mean": 0.0,
            "schedule_deviation_std": 5.0,
        } | self.sim_config.get("iz_bus_attribute_config", {})

        bus_attribute_config = VehicleAttributeConfig(**bus_attribute_config_dict)
        route_seed = self.sim_config.get("iz_bus_route_seed")
        timing_seed = self.sim_config.get("iz_bus_timing_seed")
        generation_seed = self.sim_config.get(
            "iz_bus_gen_seed",
            route_seed if route_seed is not None else self.sim_config.get("iz_seed", 420),
        )

        if "iz_bus_multiplier" in self.sim_config:
            scenario_generation_wrapper = ScenarioGenerationWrapper(generate_files)
            scenario_generation_wrapper.generate(
                config=intzoo_config,
                attribute_config=bus_attribute_config,
                prefix="",
                seed=generation_seed,
                task_context=intzoo_task,
                bus_multiplier=self.sim_config.get("iz_bus_multiplier", 1.0),
            )
        else:
            scenario_generation_wrapper = ScenarioGenerationWrapperWithBusTimetable(generate_files)

            scenario_bus_timetable_overrides = (
                {}
                if scenario_bus_timetable_overrides is None
                else dict(scenario_bus_timetable_overrides)
            )

            bus_timetable_config_dict = (
                {
                    "line_prefix": "tt",
                    "maintain_total_vehicles": True,
                    "suppress_original_buses": True,
                    "allow_route_reuse": True,
                    "route_count": 8,
                    "headway_min_seconds": 300.0,
                    "headway_max_seconds": 900.0,
                    "deviation_min_seconds": -120.0,
                    "deviation_max_seconds": 600.0,
                    "route_duration_min_seconds": 3_600.0,
                    "route_duration_max_seconds": 10_800.0,
                    "depart_lane": "0",
                    "depart_speed": "5",
                    "log_path": None,
                    "bus_seed": generation_seed,
                    "bus_route_seed": route_seed,
                    "bus_timing_seed": timing_seed,
                }
                | self.sim_config.get("iz_bus_timetable_config", {})
                | scenario_bus_timetable_overrides
            )
            if bus_timetable_config_dict.get("bus_route_seed") is None:
                bus_timetable_config_dict["bus_route_seed"] = generation_seed
            if bus_timetable_config_dict.get("bus_timing_seed") is None:
                bus_timetable_config_dict["bus_timing_seed"] = generation_seed

            bus_config = BusTimetableConfig(**bus_timetable_config_dict)

            scenario_generation_wrapper.generate(
                config=intzoo_config,
                attribute_config=bus_attribute_config,
                prefix="",
                seed=generation_seed,
                task_context=intzoo_task,
                bus_config=bus_config,
            )

    # -------------------------------------------------------------------------
    # Static helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        """Backwards-compatible wrapper for shared config merge utility."""
        return deep_merge_dicts(base, overrides)

    @classmethod
    def _to_plain_data(cls, value: Any) -> Any:
        """Backwards-compatible wrapper for shared plain-data conversion utility."""
        del cls
        return to_plain_data(value)

    def _build_generic_provenance(self) -> dict[str, Any]:
        model_paths: list[dict[str, str]] = []
        for key in ("model_path", "checkpoint_path"):
            raw_path = self.sim_config.get(key)
            if not isinstance(raw_path, str) or not raw_path:
                continue
            path = Path(raw_path)
            model_paths.append(
                {
                    "key": key,
                    "path": raw_path,
                    "exists": str(path.exists()),
                    "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                }
            )

        data_paths: list[dict[str, str]] = []
        for net_file, route_file in self.sim_config.get("scenarios", []):
            for kind, raw_path in (("net_file", net_file), ("route_file", route_file)):
                path = Path(str(raw_path))
                data_paths.append(
                    {
                        "kind": kind,
                        "path": str(raw_path),
                        "exists": str(path.exists()),
                        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
                    }
                )

        return build_run_provenance(
            command=sys.argv,
            output_root=str(self.sim_config.get("log_dir", "")),
            extra={
                "experiment": {
                    "name": self.name,
                    "group": self.group,
                    "type": self.experiment_type or "",
                },
                "models": model_paths,
                "data_files": data_paths,
                "resolved_sim_config": to_plain_data(self.sim_config),
            },
        )

    def _persist_generic_provenance(self) -> None:
        """Persist provenance locally and to W&B when available.

        YAML-run experiments need machine/code/data/model provenance in the
        durable W&B run folder. This method is intentionally generic so train
        and eval builders share the same audit trail.
        """
        try:
            provenance = to_plain_data(self._build_generic_provenance())
        except Exception as exc:
            warnings.warn(f"Failed to build experiment provenance: {exc}", stacklevel=2)
            return

        output_dirs: list[Path] = []
        log_dir = self.sim_config.get("log_dir")
        if isinstance(log_dir, str) and log_dir:
            output_dirs.append(Path(log_dir))
        run_dir = getattr(getattr(self, "wandb_run", None), "dir", None)
        if isinstance(run_dir, str) and run_dir:
            output_dirs.append(Path(run_dir))

        written_paths: list[Path] = []
        for out_dir in output_dirs:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / "experiment_provenance.json"
                path.write_text(
                    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                written_paths.append(path)
            except Exception as exc:
                warnings.warn(
                    f"Failed to persist experiment provenance in {out_dir}: {exc}", stacklevel=2
                )

        if self.wandb_run is None:
            return
        if hasattr(self.wandb_run, "config"):
            try:
                self.wandb_run.config.update(
                    {
                        "experiment_provenance_summary": provenance_summary(provenance),
                        "experiment_provenance_artifact": "experiment_provenance.json",
                    },
                    allow_val_change=True,
                )
            except Exception as exc:
                warnings.warn(f"Failed to update W&B config with provenance: {exc}", stacklevel=2)
        for path in written_paths[:1]:
            if not hasattr(self.wandb_run, "log_artifact"):
                continue
            try:
                artifact = wandb.Artifact(
                    name=f"{self.name}-experiment-provenance",
                    type="experiment_provenance",
                )
                artifact.add_file(str(path))
                self.wandb_run.log_artifact(artifact)
            except Exception as exc:
                warnings.warn(f"Failed to log W&B provenance artifact: {exc}", stacklevel=2)
