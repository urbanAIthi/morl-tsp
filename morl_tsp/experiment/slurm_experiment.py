# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH: Path = Path("morl_tsp/experiment/slurm_profiles.yaml")
type Backend = Literal["local", "slurm"]
type YamlDict = dict[str, Any]
type EnvOverrides = dict[str, str]


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    backend: Backend
    cwd: Path
    python: str
    env: dict[str, str]
    sync_before_submit: bool
    sync_paths: list[Path]
    sync_excludes: list[str]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ResolvedExperiment:
    yaml_path: Path
    runner: str
    job_name: str
    runner_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlurmResources:
    cpus: int | None
    gpus: int | None
    mem: str | None


def _load_yaml_mapping(path: Path) -> YamlDict:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cast(YamlDict, data)


def _deep_merge(base: Mapping[str, Any], overwrite: Mapping[str, Any]) -> YamlDict:
    merged: YamlDict = dict(base)
    for key, value in overwrite.items():
        if key == "base":
            continue
        old_value: Any = merged.get(key)
        if isinstance(old_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(cast(Mapping[str, Any], old_value), value)
        else:
            merged[key] = value
    return merged


def _resolve_path(path: str | Path, cwd: Path = ROOT) -> Path:
    raw_path: Path = Path(path).expanduser()
    return raw_path if raw_path.is_absolute() else cwd / raw_path


def _profile_data(profile_path: Path, profile_name: str) -> YamlDict:
    config: YamlDict = _load_yaml_mapping(profile_path)
    bases: Any = config.get("bases", {})
    profiles: Any = config.get("profiles", {})
    if not isinstance(bases, dict) or not isinstance(profiles, dict):
        raise ValueError(f"{profile_path} must contain bases: and profiles: mappings")
    if profile_name not in profiles:
        available: str = ", ".join(sorted(str(key) for key in profiles))
        raise ValueError(f"Unknown profile {profile_name!r}. Available profiles: {available}")

    profile: Any = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"Profile {profile_name!r} must be a mapping")
    base_name: str = str(profile.get("base", "local"))
    base: Any = bases.get(base_name)
    if not isinstance(base, dict):
        raise ValueError(f"Profile {profile_name!r} references unknown base {base_name!r}")
    return _deep_merge(cast(Mapping[str, Any], base), cast(Mapping[str, Any], profile))


def load_profile(
    profile_path: Path, profile_name: str, env_overrides: EnvOverrides
) -> ExperimentProfile:
    data: YamlDict = _profile_data(profile_path, profile_name)
    backend: str = str(data.get("backend", "local"))
    if backend not in {"local", "slurm"}:
        raise ValueError(f"Unsupported backend {backend!r} in profile {profile_name!r}")

    cwd: Path = _resolve_path(str(data.get("cwd", ".")))
    env_data: Any = data.get("env", {})
    if not isinstance(env_data, dict):
        raise ValueError(f"Profile {profile_name!r} env must be a mapping")
    env: dict[str, str] = {str(key): str(value) for key, value in env_data.items()}
    env.update(env_overrides)
    sync_paths_data: Any = data.get("sync_paths", [])
    if not isinstance(sync_paths_data, list):
        raise ValueError(f"Profile {profile_name!r} sync_paths must be a list")
    sync_excludes_data: Any = data.get("sync_excludes", [])
    if not isinstance(sync_excludes_data, list):
        raise ValueError(f"Profile {profile_name!r} sync_excludes must be a list")

    return ExperimentProfile(
        name=profile_name,
        backend=cast(Backend, backend),
        cwd=cwd,
        python=str(data.get("python", "uv run python")),
        env=env,
        sync_before_submit=bool(data.get("sync_before_submit", False)),
        sync_paths=[_resolve_path(str(path), cwd) for path in sync_paths_data],
        sync_excludes=[str(pattern) for pattern in sync_excludes_data],
        raw=data,
    )


def _is_hpt_yaml(data: Mapping[str, Any]) -> bool:
    return isinstance(data.get("hyperparameter_object"), dict)


def _is_eval_yaml(data: Mapping[str, Any]) -> bool:
    return isinstance(data.get("eval"), dict)


def _safe_name(name: str) -> str:
    safe: str = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "experiment"


def _study_name(data: Mapping[str, Any], yaml_path: Path) -> str:
    database: Any = data.get("database")
    if not isinstance(database, dict):
        raise ValueError(f"{yaml_path} is an HPT YAML and must define database.study_name")
    study_name: Any = database.get("study_name")
    if not isinstance(study_name, str) or not study_name.strip():
        raise ValueError(f"{yaml_path} is an HPT YAML and must define database.study_name")
    return study_name.strip()


def resolve_experiment(yaml_path: Path) -> ResolvedExperiment:
    data: YamlDict = _load_yaml_mapping(yaml_path)
    if _is_hpt_yaml(data):
        study_name: str = _study_name(data, yaml_path)
        return ResolvedExperiment(
            yaml_path=yaml_path,
            runner="scripts/run_hp_experiment.py",
            job_name=_safe_name(study_name),
        )
    if _is_eval_yaml(data):
        return ResolvedExperiment(
            yaml_path=yaml_path,
            runner="scripts/run_experiments.py",
            job_name=_safe_name(yaml_path.stem),
            runner_args=("--runner", "morl_tsp/experiment/run_morl_eval_experiment.py"),
        )

    return ResolvedExperiment(
        yaml_path=yaml_path,
        runner="scripts/run_experiments.py",
        job_name=_safe_name(yaml_path.stem),
    )


def _int_or_none(value: Any, *, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"slurm.{key} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError(f"slurm.{key} must be an integer")


def _slurm_resources(data: Mapping[str, Any]) -> SlurmResources:
    raw_slurm: Any = data.get("slurm", {})
    if raw_slurm is None:
        raw_slurm = {}
    if not isinstance(raw_slurm, dict):
        raise ValueError("slurm must be a mapping when provided")

    cpus: int | None = _int_or_none(raw_slurm.get("cpus"), key="cpus")
    gpus: int | None = _int_or_none(raw_slurm.get("gpus"), key="gpus")
    mem_raw: Any = raw_slurm.get("mem", raw_slurm.get("ram"))
    mem: str | None = str(mem_raw).strip() if mem_raw is not None else None

    if cpus is not None and cpus <= 0:
        raise ValueError("slurm.cpus must be greater than 0")
    if gpus is not None and gpus < 0:
        raise ValueError("slurm.gpus must be greater than or equal to 0")
    if mem is not None and len(mem) == 0:
        raise ValueError("slurm.mem must be non-empty when provided")
    return SlurmResources(cpus=cpus, gpus=gpus, mem=mem)


def _python_command(profile: ExperimentProfile) -> list[str]:
    return shlex.split(profile.python)


def _local_command(profile: ExperimentProfile, experiment: ResolvedExperiment) -> list[str]:
    return [
        *_python_command(profile),
        experiment.runner,
        "--yaml_path",
        str(experiment.yaml_path),
        *experiment.runner_args,
    ]


def _remote_wrapper_command(
    *,
    args: argparse.Namespace,
    config_arg: Path,
    profile: ExperimentProfile,
    env_overrides: EnvOverrides,
) -> list[str]:
    remote_env: EnvOverrides = {**profile.env, **env_overrides}
    command: list[str] = [
        *_python_command(profile),
        "scripts/slurm_experiment.py",
        "--yaml_path",
        str(args.yaml_path),
        "--profile",
        "local",
        "--config",
        str(config_arg),
    ]
    for key, value in sorted(remote_env.items()):
        command.extend([f"--{key}", value])
    return command


def _format_slurm_command(
    profile: ExperimentProfile,
    *,
    job_name: str,
    command: Sequence[str],
    resources: SlurmResources,
) -> list[str]:
    ssh_target = str(profile.raw.get("ssh_target", "")).strip()
    remote_repo = str(profile.raw.get("remote_repo", "")).strip()
    out_dir = str(profile.raw.get("out_dir", "")).strip()
    if not ssh_target or not remote_repo or not out_dir:
        raise ValueError(f"Profile {profile.name!r} must define ssh_target, remote_repo, and out_dir")

    env_exports: str = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(profile.env.items())
    )
    wrapped_command: str = f"{env_exports} {shlex.join(command)}".strip()
    cpus = resources.cpus if resources.cpus is not None else int(profile.raw.get("cpus_per_task", 1))
    mem = resources.mem if resources.mem is not None else str(profile.raw.get("mem", "16G"))

    partition = str(profile.raw.get("partition", "cpu"))
    qos: str | None = str(profile.raw["qos"]) if "qos" in profile.raw else None
    gres: str | None = str(profile.raw["gres"]) if "gres" in profile.raw else None
    if resources.gpus is not None:
        if resources.gpus == 0:
            partition = "cpu" if partition.startswith("gpu-") else partition
            qos = None
            gres = None
        else:
            partition = f"gpu-{resources.gpus}"
            qos = f"qos-gpu-{resources.gpus}"
            gres = f"gpu:{resources.gpus}"

    sbatch_parts = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={profile.raw.get('time', '04:00:00')}",
        f"--output={out_dir}/%x_%j.out",
        f"--error={out_dir}/%x_%j.err",
    ]
    if qos is not None:
        sbatch_parts.append(f"--qos={qos}")
    if gres is not None:
        sbatch_parts.append(f"--gres={gres}")
    sbatch_parts.append(f"--wrap={shlex.quote(wrapped_command)}")

    remote_command = f"cd {shlex.quote(remote_repo)} && mkdir -p {shlex.quote(out_dir)} && {' '.join(sbatch_parts)}"
    return ["ssh", ssh_target, remote_command]


def _path_for_rsync(path: Path, cwd: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(cwd.resolve()))
    except ValueError as exc:
        raise ValueError(f"Can only sync paths under {cwd}: {path}") from exc


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        deduped.append(path)
    return deduped


def _sync_remote_checkout(
    profile: ExperimentProfile,
    *,
    yaml_path: Path,
    config_path: Path,
    submit: bool,
    dry_run: bool,
) -> None:
    if not profile.sync_before_submit:
        return
    ssh_target = str(profile.raw.get("ssh_target", "")).strip()
    remote_repo = str(profile.raw.get("remote_repo", "")).strip()
    if not ssh_target or not remote_repo:
        raise ValueError(f"Profile {profile.name!r} must define ssh_target and remote_repo to sync")

    sync_paths = _dedupe_paths([*profile.sync_paths, yaml_path, config_path])
    relative_paths = [_path_for_rsync(path, profile.cwd) for path in sync_paths if path.exists()]
    if len(relative_paths) == 0:
        return

    command = ["rsync", "-azR"]
    for pattern in profile.sync_excludes:
        command.extend(["--exclude", pattern])
    command.extend(relative_paths)
    command.append(f"{ssh_target}:{remote_repo.rstrip('/')}/")
    print(shlex.join(command), flush=True)
    if submit and not dry_run:
        subprocess.run(command, cwd=profile.cwd, check=True)


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], dry_run: bool) -> None:
    print(shlex.join(list(command)), flush=True)
    if dry_run:
        return
    full_env: dict[str, str] = os.environ.copy()
    full_env.update(env)
    subprocess.run(list(command), cwd=cwd, env=full_env, check=True)


def _parse_env_overrides(unknown_args: Sequence[str]) -> EnvOverrides:
    overrides: EnvOverrides = {}
    index: int = 0
    while index < len(unknown_args):
        token: str = unknown_args[index]
        if not token.startswith("--"):
            raise ValueError(
                f"Unexpected argument {token!r}; env overrides must look like --WANDB_MODE offline"
            )
        raw_key: str = token[2:]
        if "=" in raw_key:
            key, value = raw_key.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(unknown_args):
                raise ValueError(f"Missing value for environment override {token!r}")
            key = raw_key
            value = unknown_args[index + 1]
            index += 2
        if not key:
            raise ValueError(f"Invalid environment override {token!r}")
        overrides[key] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run a MORL-TSP experiment YAML locally or submit it through a Slurm profile."
    )
    parser.add_argument("--yaml_path", required=True, type=Path)
    parser.add_argument("--profile", default="slurm1_gpu1")
    parser.add_argument("--config", default=DEFAULT_PROFILE_PATH, type=Path)
    parser.add_argument("--submit", action="store_true", default=True)
    parser.add_argument("--no-submit", action="store_false", dest="submit")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser: argparse.ArgumentParser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    config_arg: Path = args.config
    env_overrides: EnvOverrides = _parse_env_overrides(unknown_args)
    args.config = _resolve_path(args.config)
    profile: ExperimentProfile = load_profile(args.config, str(args.profile), env_overrides)
    yaml_path: Path = _resolve_path(args.yaml_path, profile.cwd)

    if profile.backend == "local":
        experiment: ResolvedExperiment = resolve_experiment(yaml_path)
        _run(
            _local_command(profile, experiment),
            cwd=profile.cwd,
            env=profile.env,
            dry_run=args.dry_run,
        )
        return

    yaml_data: YamlDict = _load_yaml_mapping(yaml_path)
    job_name: str = (
        _safe_name(_study_name(yaml_data, yaml_path))
        if _is_hpt_yaml(yaml_data)
        else _safe_name(yaml_path.stem)
    )
    remote_command: list[str] = _remote_wrapper_command(
        args=args,
        config_arg=config_arg,
        profile=profile,
        env_overrides=env_overrides,
    )
    _sync_remote_checkout(
        profile,
        yaml_path=yaml_path,
        config_path=args.config,
        submit=args.submit,
        dry_run=args.dry_run,
    )
    slurm_command: list[str] = _format_slurm_command(
        profile,
        job_name=job_name,
        command=remote_command,
        resources=_slurm_resources(yaml_data),
    )
    _run(slurm_command, cwd=profile.cwd, env=profile.env, dry_run=args.dry_run or not args.submit)


if __name__ == "__main__":
    main()
