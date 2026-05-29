# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from morl_tsp import config

TRACKED_ENV_VARS = (
    "SUMO_HOME",
    "SUMO_BINARY",
    "MORL_TSP_TRACI_BACKEND",
    "MORL_TSP_USE_LIBSUMO",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "TF_DETERMINISTIC_OPS",
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_output(repo_root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def resolve_python_bin() -> str:
    if sys.executable:
        return sys.executable
    for candidate in ("python3", "python"):
        found = shutil.which(candidate)
        if found:
            return found
    return "python3"


def _resolve_packaged_sumo_home() -> str | None:
    for package_name in ("sumo", "libsumo", "sumolib", "traci"):
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            continue
        package_dir: Path | None = None
        if spec.submodule_search_locations:
            package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
        elif spec.origin:
            package_dir = Path(spec.origin).resolve().parent
        if package_dir is None:
            continue
        if package_name == "sumo":
            return str(package_dir)
        return str(package_dir.parent)
    return None


def resolve_sumo_home() -> str | None:
    packaged_sumo = _resolve_packaged_sumo_home()
    if packaged_sumo:
        return packaged_sumo
    env_sumo = os.environ.get("SUMO_HOME")
    if env_sumo:
        return env_sumo
    for candidate in (
        Path("/usr/share/sumo"),
        Path("/usr/local/share/sumo"),
        Path("/opt/homebrew/share/sumo"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        pass
    try:
        module = __import__(package_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version.strip()
    return None


def _sumo_binary_path() -> str | None:
    env_binary = os.environ.get("SUMO_BINARY", "").strip()
    if env_binary:
        return env_binary
    sumo_home = resolve_sumo_home()
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / "sumo"
        if candidate.exists():
            return str(candidate)
    python_bin = Path(resolve_python_bin())
    sibling = python_bin.with_name("sumo")
    if sibling.exists():
        return str(sibling)
    return shutil.which("sumo")


def _sumo_binary_version() -> str | None:
    binary_path = _sumo_binary_path()
    if binary_path is None:
        return None
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    text = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"Version\s+(\d+\.\d+\.\d+)", text)
    if match is not None:
        return match.group(1)
    match = re.search(r"sumo\s+(\d+\.\d+\.\d+)", text)
    if match is not None:
        return match.group(1)
    return None


def _sumo_binary_details() -> dict[str, Any]:
    binary_path = _sumo_binary_path()
    path = Path(binary_path).resolve() if binary_path is not None else None
    return {
        "path": str(path) if path is not None else "",
        "exists": bool(path is not None and path.exists()),
        "sha256": sha256_file(path)
        if path is not None and path.exists() and path.is_file()
        else "",
        "version": _sumo_binary_version(),
    }


def sumo_runtime_versions() -> dict[str, str | None]:
    return {
        "eclipse-sumo": _package_version("eclipse-sumo"),
        "libsumo": _package_version("libsumo"),
        "sumo_binary": _sumo_binary_version(),
        "traci": _package_version("traci"),
        "sumolib": _package_version("sumolib"),
    }


def _module_provenance(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    path = ""
    if spec is not None:
        if spec.origin:
            path = str(Path(spec.origin).resolve())
        elif spec.submodule_search_locations:
            path = str(Path(next(iter(spec.submodule_search_locations))).resolve())
    file_path = Path(path) if path else None
    return {
        "module": module_name,
        "path": path,
        "version": _package_version(module_name),
        "exists": bool(file_path is not None and file_path.exists()),
        "sha256": (
            sha256_file(file_path)
            if file_path is not None and file_path.exists() and file_path.is_file()
            else ""
        ),
    }


def _tracked_environment_variables() -> dict[str, str]:
    return {key: str(os.environ.get(key, "")) for key in TRACKED_ENV_VARS if key in os.environ}


def _dependency_snapshot() -> dict[str, Any]:
    packages: list[dict[str, str]] = []
    for distribution in importlib_metadata.distributions():
        name = str(
            distribution.metadata.get("Name", distribution.metadata.get("Summary", ""))
        ).strip()
        if not name:
            name = str(distribution.metadata.get("name", "")).strip()
        if not name:
            continue
        version = str(distribution.version).strip()
        packages.append({"name": name, "version": version})
    packages.sort(key=lambda row: (row["name"].lower(), row["version"]))
    payload = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "package_count": len(packages),
        "packages_sha256": hashlib.sha256(payload).hexdigest(),
        "packages": packages,
    }


def _repo_lockfile_provenance(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = [
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
        repo_root / "docs" / "requirements.txt",
    ]
    for path in candidates:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    return rows


def backend_provenance() -> dict[str, Any]:
    try:
        from morl_tsp.util import backend as backend_module

        module_name = getattr(getattr(backend_module, "traci", None), "__name__", "")
        if backend_module.LIBSUMO:
            actual_backend = "libsumo"
        elif module_name == "libtraci":
            actual_backend = "libtraci"
        elif module_name == "traci":
            actual_backend = "traci"
        else:
            actual_backend = module_name
        return {
            "requested_backend": str(os.environ.get("MORL_TSP_TRACI_BACKEND", "")).strip(),
            "prefer_libsumo_env": str(os.environ.get("MORL_TSP_USE_LIBSUMO", "")).strip(),
            "actual_backend": actual_backend,
            "module_name": module_name,
            "libsumo_active": bool(getattr(backend_module, "LIBSUMO", False)),
            "supports_connection_labels": bool(
                getattr(backend_module, "TRACI_SUPPORTS_CONNECTION_LABELS", False)
            ),
        }
    except Exception as exc:
        return {
            "requested_backend": str(os.environ.get("MORL_TSP_TRACI_BACKEND", "")).strip(),
            "prefer_libsumo_env": str(os.environ.get("MORL_TSP_USE_LIBSUMO", "")).strip(),
            "actual_backend": "",
            "module_name": "",
            "libsumo_active": False,
            "supports_connection_labels": False,
            "error": str(exc),
        }


def resolve_intersection_zoo_root() -> Path | None:
    for env_var in ("INTERSECTION_ZOO_PATH", "INTERSECTION_ZOO_ROOT_PATH"):
        raw_path = os.environ.get(env_var, "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve()
        if (candidate / "dataset").exists():
            return candidate
    spec = importlib.util.find_spec("intersection_zoo")
    if spec is None:
        return None
    candidates: list[Path] = []
    if spec.submodule_search_locations:
        candidates.extend(Path(path).resolve() for path in spec.submodule_search_locations)
    elif spec.origin:
        candidates.append(Path(spec.origin).resolve().parent)
    for candidate in candidates:
        for root in (candidate, candidate.parent):
            if (root / "dataset").exists():
                return root
    return None


def _git_tracked(repo_root: Path, rel_path: Path) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", str(rel_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return True


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root or config.ROOT_PATH).resolve()
    status = _git_output(root, ["status", "--porcelain"])
    return {
        "root": str(root),
        "commit": _git_output(root, ["rev-parse", "HEAD"]),
        "branch": _git_output(root, ["branch", "--show-current"]),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def intersection_zoo_provenance() -> dict[str, Any]:
    root = resolve_intersection_zoo_root()
    if root is None:
        return {"root": "", "commit": "", "dirty": None}
    return {
        "root": str(root),
        "commit": _git_output(root, ["rev-parse", "HEAD"]),
        "branch": _git_output(root, ["branch", "--show-current"]),
        "dirty": bool(_git_output(root, ["status", "--porcelain"])),
    }


def machine_provenance() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "container_id": Path("/etc/hostname").read_text(encoding="utf-8").strip()
        if Path("/etc/hostname").exists()
        else "",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "python_executable": resolve_python_bin(),
        "python_version": sys.version.replace("\n", " "),
        "sumo_home": resolve_sumo_home() or "",
        "sumo_versions": sumo_runtime_versions(),
        "sumo_binary": _sumo_binary_details(),
    }


def dataset_file_provenance(networks: list[str]) -> list[dict[str, Any]]:
    intersection_zoo_root = resolve_intersection_zoo_root()
    if intersection_zoo_root is None:
        return []
    rows: list[dict[str, Any]] = []
    for network in sorted(set(networks)):
        network_rel = Path(network)
        dataset_dir = intersection_zoo_root / "dataset" / network_rel.parent
        for rel_name in ("net.net.xml", "inflows.txt"):
            path = dataset_dir / rel_name
            rel_path = (
                path.relative_to(intersection_zoo_root)
                if path.exists()
                else Path("dataset") / network_rel.parent / rel_name
            )
            rows.append(
                {
                    "network": network,
                    "path": str(path),
                    "exists": path.exists(),
                    "tracked": _git_tracked(intersection_zoo_root, rel_path)
                    if path.exists()
                    else False,
                    "sha256": sha256_file(path) if path.exists() else "",
                }
            )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_run_provenance(
    *,
    command: list[str] | None = None,
    manifest_path: str | None = None,
    output_root: str | None = None,
    extra: dict[str, Any] | None = None,
    dataset_networks: list[str] | None = None,
    config_path: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(config.ROOT_PATH)
    dependency_snapshot = _dependency_snapshot()
    out: dict[str, Any] = {
        "machine": machine_provenance(),
        "repo": git_provenance(repo_root),
        "intersection_zoo": intersection_zoo_provenance(),
        "command": command or sys.argv,
        "cwd": os.getcwd(),
        "manifest_path": manifest_path or "",
        "config_path": config_path or "",
        "model_path": model_path or "",
        "output_root": output_root or "",
        "runtime": {
            "backend": backend_provenance(),
            "environment_variables": _tracked_environment_variables(),
            "lockfiles": _repo_lockfile_provenance(repo_root),
            "dependency_snapshot": {
                "package_count": dependency_snapshot["package_count"],
                "packages_sha256": dependency_snapshot["packages_sha256"],
            },
            "python_modules": [
                _module_provenance("sumo"),
                _module_provenance("libsumo"),
                _module_provenance("sumolib"),
                _module_provenance("traci"),
                _module_provenance("libtraci"),
            ],
        },
        "dependencies": dependency_snapshot,
    }
    if dataset_networks is not None:
        out["dataset_files"] = dataset_file_provenance(dataset_networks)
    if extra:
        out.update(extra)
    return out


def provenance_summary(provenance: dict[str, Any]) -> dict[str, Any]:
    runtime = provenance.get("runtime", {})
    dependency_snapshot = (
        runtime.get("dependency_snapshot", {}) if isinstance(runtime, dict) else {}
    )
    machine = provenance.get("machine", {})
    repo = provenance.get("repo", {})
    return {
        "hostname": machine.get("hostname", "") if isinstance(machine, dict) else "",
        "python_version": machine.get("python_version", "") if isinstance(machine, dict) else "",
        "sumo_binary_version": (
            machine.get("sumo_binary", {}).get("version", "")
            if isinstance(machine, dict) and isinstance(machine.get("sumo_binary"), dict)
            else ""
        ),
        "sumo_binary_sha256": (
            machine.get("sumo_binary", {}).get("sha256", "")
            if isinstance(machine, dict) and isinstance(machine.get("sumo_binary"), dict)
            else ""
        ),
        "backend": runtime.get("backend", {}) if isinstance(runtime, dict) else {},
        "repo_commit": repo.get("commit", "") if isinstance(repo, dict) else "",
        "repo_dirty": repo.get("dirty", False) if isinstance(repo, dict) else False,
        "dependency_package_count": (
            dependency_snapshot.get("package_count", "")
            if isinstance(dependency_snapshot, dict)
            else ""
        ),
        "dependency_packages_sha256": (
            dependency_snapshot.get("packages_sha256", "")
            if isinstance(dependency_snapshot, dict)
            else ""
        ),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha1_payload(payload: dict[str, Any]) -> str:
    digest_input = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha1(digest_input).hexdigest()


def route_config_with_seeds(
    route_config: dict[str, Any], route_seed: int, timing_seed: int
) -> dict[str, Any]:
    cfg = dict(route_config)
    cfg["bus_route_seed"] = int(route_seed)
    cfg["bus_timing_seed"] = int(timing_seed)
    return cfg


def build_route_generation_identity(
    *,
    name: str,
    suite: str,
    city_tag: str,
    network: str,
    route_config: dict[str, Any],
    route_seed: int,
    timing_seed: int,
    scenario_duration: int = 10_800,
    run_up_time: int = 3_600,
    step_length: float = 1.0,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    intersection_zoo_root = resolve_intersection_zoo_root()
    net_file = (
        intersection_zoo_root / "dataset" / network
        if intersection_zoo_root is not None
        else Path("dataset") / network
    )
    inflows_file = net_file.parent / "inflows.txt"
    seeded_route_config = route_config_with_seeds(route_config, route_seed, timing_seed)
    cache_payload = {
        "version": 1,
        "net_file_path": str(net_file.resolve()) if net_file.exists() else str(net_file),
        "scenario_duration": int(scenario_duration),
        "run_up_time": int(run_up_time),
        "step_length": float(step_length),
        "generation_seed": int(route_seed),
        "iz_bus_multiplier": None,
        "iz_bus_attribute_config": {},
        "iz_bus_timetable_config": seeded_route_config,
        "iz_bus_timetable_config_per_scenario": {},
    }
    cache_key = _sha1_payload(cache_payload)[:24]
    settings_sha1 = _sha1_payload(cache_payload)
    if cache_root is None:
        cache_root = Path(config.ROOT_PATH) / "wandb" / "_iz_scenario_cache"
    cache_dir = cache_root / cache_key / "sumo"
    cached_route_file = cache_dir / "routes.rou.xml"
    cached_net_file = cache_dir / "net.net.xml"
    return {
        "name": name,
        "suite": suite,
        "city": city_tag,
        "network": network,
        "route_seed": int(route_seed),
        "timing_seed": int(timing_seed),
        "route_generation_hash": cache_key,
        "route_generation_settings_sha1": settings_sha1,
        "route_generation_settings": cache_payload,
        "source_net_file": str(net_file),
        "source_net_sha256": sha256_file(net_file) if net_file.exists() else "",
        "source_inflows_file": str(inflows_file),
        "source_inflows_sha256": sha256_file(inflows_file) if inflows_file.exists() else "",
        "cached_route_file": str(cached_route_file),
        "cached_route_exists": cached_route_file.exists(),
        "cached_route_sha256": sha256_file(cached_route_file) if cached_route_file.exists() else "",
        "cached_net_file": str(cached_net_file),
        "cached_net_exists": cached_net_file.exists(),
        "cached_net_sha256": sha256_file(cached_net_file) if cached_net_file.exists() else "",
    }


def route_generation_records(manifest: dict[str, Any], suites: list[str]) -> list[dict[str, Any]]:
    seeds = manifest.get("seeds", {})
    route_seeds = [int(seed) for seed in seeds.get("eval_route_seeds", [])]
    timing_seeds = [int(seed) for seed in seeds.get("eval_timing_seeds", route_seeds)]
    if len(timing_seeds) != len(route_seeds):
        raise ValueError("eval_timing_seeds length must match eval_route_seeds length")

    route_configs = manifest.get("route_configs", {})
    suite_cfgs = manifest.get("suites", {})
    city_groups = manifest.get("cities", {})
    cache_root = Path(config.ROOT_PATH) / "wandb" / "_iz_scenario_cache"
    records: list[dict[str, Any]] = []
    for suite in suites:
        suite_cfg = suite_cfgs.get(suite, {}) if isinstance(suite_cfgs, dict) else {}
        if not isinstance(suite_cfg, dict):
            continue
        route_config_key = str(suite_cfg.get("route_config", "")).strip()
        route_config = (
            route_configs.get(route_config_key, {}) if isinstance(route_configs, dict) else {}
        )
        if not isinstance(route_config, dict):
            route_config = {}
        city_key = str(suite_cfg.get("cities", "")).strip()
        cities = city_groups.get(city_key, []) if isinstance(city_groups, dict) else []
        if not isinstance(cities, list):
            continue
        for city in cities:
            if not isinstance(city, dict):
                continue
            tag = str(city.get("tag", "")).strip()
            network = str(city.get("network", "")).strip()
            if not tag or not network:
                continue
            for route_seed, timing_seed in zip(route_seeds, timing_seeds, strict=True):
                records.append(
                    build_route_generation_identity(
                        name=f"{suite}:{tag}:rs{route_seed}:ts{timing_seed}",
                        suite=suite,
                        city_tag=tag,
                        network=network,
                        route_config=dict(route_config),
                        route_seed=route_seed,
                        timing_seed=timing_seed,
                        cache_root=cache_root,
                    )
                )
    return records
