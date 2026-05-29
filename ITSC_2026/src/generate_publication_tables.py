#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

MethodName = Literal["MORL", "PPO-S", "FixedTime", "RuleTSP"]

ITSC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ITSC_ROOT.parent
DEFAULT_RESULTS_ROOT = ITSC_ROOT / "results" / "evaluations"
DEFAULT_OUTPUT_DIR = ITSC_ROOT / "results" / "tables"

MORL_RC4_DIRNAME = "morl_rc4_dataset"
MORL_IZ_DIRNAME = "morl_iz_default_dataset"
PPO_RC4_DIRNAME = "ppo_rc4_dataset"

CASE_TAGS: list[str] = ["chi2412", "la2114", "nyc10802"]
CASE_WEIGHTS: list[float] = [0.3, 0.5, 0.7]
EXPECTED_SETPOINTS: list[float] = [round(i / 10.0, 1) for i in range(11)]

HV_REF_BUS = 500.0
HV_REF_NONBUS = 500.0

STATIC_INTERSECTION_FEATURES: dict[str, dict[str, float]] = {
    "dal1082": {
        "total_flow": 3888.072,
        "traffic_imbalance": 0.07409122053295314,
        "num_approaches": 3.0,
        "num_green_phases": 2.0,
        "num_lanes": 8.0,
        "cycle_s": 84.0,
        "green_imbalance": 0.07409122053295314,
        "max_phase_share": 0.5370456102664766,
    },
    "chi1027": {
        "total_flow": 3154.2912,
        "traffic_imbalance": 0.4293488185237939,
        "num_approaches": 3.0,
        "num_green_phases": 2.0,
        "num_lanes": 7.0,
        "cycle_s": 51.0,
        "green_imbalance": 0.429348818523794,
        "max_phase_share": 0.714674409261897,
    },
    "sea543": {
        "total_flow": 3142.906200000001,
        "traffic_imbalance": 0.47543989063370695,
        "num_approaches": 4.0,
        "num_green_phases": 4.0,
        "num_lanes": 12.0,
        "cycle_s": 102.0,
        "green_imbalance": 0.47543989063370706,
        "max_phase_share": 0.5178959015703363,
    },
    "la477": {
        "total_flow": 3111.494400000001,
        "traffic_imbalance": 0.3999999999999999,
        "num_approaches": 4.0,
        "num_green_phases": 4.0,
        "num_lanes": 12.0,
        "cycle_s": 92.0,
        "green_imbalance": 0.4,
        "max_phase_share": 0.4499999999999999,
    },
    "chi2412": {
        "total_flow": 3077.6256,
        "traffic_imbalance": 0.4000000000000001,
        "num_approaches": 4.0,
        "num_green_phases": 4.0,
        "num_lanes": 12.0,
        "cycle_s": 107.0,
        "green_imbalance": 0.4,
        "max_phase_share": 0.45000000000000007,
    },
    "chi758": {
        "total_flow": 2836.3104000000008,
        "traffic_imbalance": 0.44545454545454544,
        "num_approaches": 4.0,
        "num_green_phases": 4.0,
        "num_lanes": 11.0,
        "cycle_s": 107.0,
        "green_imbalance": 0.4454545454545455,
        "max_phase_share": 0.4909090909090909,
    },
    "sf838": {
        "total_flow": 2810.094,
        "traffic_imbalance": 0.25918065374325555,
        "num_approaches": 4.0,
        "num_green_phases": 2.0,
        "num_lanes": 11.0,
        "cycle_s": 53.0,
        "green_imbalance": 0.25918065374325555,
        "max_phase_share": 0.6295903268716279,
    },
    "slc47": {
        "total_flow": 2558.0160000000005,
        "traffic_imbalance": 0.6470811754109435,
        "num_approaches": 4.0,
        "num_green_phases": 3.0,
        "num_lanes": 7.0,
        "cycle_s": 79.0,
        "green_imbalance": 0.6470811754109435,
        "max_phase_share": 0.6471796892591758,
    },
    "nyc10802": {
        "total_flow": 2533.674,
        "traffic_imbalance": 0.28956921845509725,
        "num_approaches": 3.0,
        "num_green_phases": 2.0,
        "num_lanes": 7.0,
        "cycle_s": 56.0,
        "green_imbalance": 0.28956921845509725,
        "max_phase_share": 0.6447846092275487,
    },
    "bos666": {
        "total_flow": 1875.8880000000001,
        "traffic_imbalance": 0.8,
        "num_approaches": 3.0,
        "num_green_phases": 3.0,
        "num_lanes": 7.0,
        "cycle_s": 79.0,
        "green_imbalance": 0.8000000000000002,
        "max_phase_share": 0.8166666666666668,
    },
    "atl66": {
        "total_flow": 1651.4400000000005,
        "traffic_imbalance": 0.2232960325534079,
        "num_approaches": 3.0,
        "num_green_phases": 2.0,
        "num_lanes": 8.0,
        "cycle_s": 40.0,
        "green_imbalance": 0.22329603255340796,
        "max_phase_share": 0.6116480162767038,
    },
    "la2114": {
        "total_flow": 1550.7492000000002,
        "traffic_imbalance": 0.5832211166060896,
        "num_approaches": 4.0,
        "num_green_phases": 3.0,
        "num_lanes": 8.0,
        "cycle_s": 77.0,
        "green_imbalance": 0.5832211166060896,
        "max_phase_share": 0.621110106005536,
    },
    "nyc1078": {
        "total_flow": 1110.5220000000004,
        "traffic_imbalance": 0.5023032411784728,
        "num_approaches": 3.0,
        "num_green_phases": 3.0,
        "num_lanes": 7.0,
        "cycle_s": 74.0,
        "green_imbalance": 0.5023032411784728,
        "max_phase_share": 0.6676903294126545,
    },
}

HEATMAP_FEATURES: tuple[tuple[str, str], ...] = (
    ("Total Flow", "total_flow"),
    ("Traffic Imb.", "traffic_imbalance"),
    (r"\#App.", "num_approaches"),
    (r"\#Phases", "num_green_phases"),
    (r"\#Lanes", "num_lanes"),
    ("Cycle (s)", "cycle_s"),
    ("Green Imb.", "green_imbalance"),
    ("Max Phase Share", "max_phase_share"),
)

MODEL_MAP: dict[str, MethodName] = {
    "morl": "MORL",
    "ppo": "PPO-S",
    "fixed_ts": "FixedTime",
    "tsp": "RuleTSP",
}

CITY_PREFIX_MAP: dict[str, str] = {
    "atlanta": "Atlanta",
    "boston": "Boston",
    "chicago": "Chicago",
    "dallas": "Dallas",
    "los-angeles": "Los Angeles",
    "new-york-city": "New York",
    "salt-lake-city": "Salt Lake City",
    "san-francisco": "San Francisco",
    "seattle": "Seattle",
    "atl": "Atlanta",
    "bos": "Boston",
    "chi": "Chicago",
    "dal": "Dallas",
    "la": "Los Angeles",
    "nyc": "New York",
    "sea": "Seattle",
    "sf": "San Francisco",
    "slc": "Salt Lake City",
}


@dataclass(frozen=True)
class TableInputs:
    morl_rc4: pd.DataFrame
    morl_iz: pd.DataFrame
    ppo_rc4: pd.DataFrame


def _std_or_zero(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def _round_weight(value: object) -> float:
    return round(float(value) + 1e-12, 1)


def _format_mean_std(value: object, std: object, digits: int = 1) -> str:
    if pd.isna(value):
        return "--"
    if pd.isna(std):
        return f"{float(value):.{digits}f}"
    return f"{float(value):.{digits}f}$\\pm${float(std):.{digits}f}"


def _format_range(min_value: object, max_value: object, digits: int = 1) -> str:
    if pd.isna(min_value) or pd.isna(max_value):
        return "--"
    return f"[{float(min_value):.{digits}f}--{float(max_value):.{digits}f}]"


def _numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")


def _city_from_network_or_tag(network: object, tag: str) -> str:
    if isinstance(network, str) and "/" in network:
        prefix = network.split("/", maxsplit=1)[0]
    else:
        match = re.match(r"([a-z\-]+)", tag)
        prefix = match.group(1) if match else tag
    return CITY_PREFIX_MAP.get(prefix, prefix.replace("-", " ").title())


def _canonical_rows_files(folder: Path) -> dict[str, Path]:
    rows_by_tag: dict[str, list[tuple[str, Path, int]]] = defaultdict(list)
    for subdir in sorted(folder.iterdir()):
        if not subdir.is_dir():
            continue
        rows_csv = subdir / "rows.csv"
        if not rows_csv.is_file():
            continue
        with rows_csv.open(encoding="utf-8", errors="replace") as file:
            row_count = max(sum(1 for _ in file) - 1, 0)

        tag = subdir.name
        for suffix in ("_extra", "_iz_default"):
            if tag.endswith(suffix):
                tag = tag[: -len(suffix)]
        rows_by_tag[tag].append((subdir.name, rows_csv, row_count))

    selected: dict[str, Path] = {}
    for tag, candidates in rows_by_tag.items():
        max_rows = max(candidate[2] for candidate in candidates)
        largest = [candidate for candidate in candidates if candidate[2] == max_rows]
        preferred = next((candidate for candidate in largest if candidate[0] == tag), largest[0])
        selected[tag] = preferred[1]
    return dict(sorted(selected.items()))


def _load_dataset(folder: Path, regime: str, source_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tag, csv_path in _canonical_rows_files(folder).items():
        rows = pd.read_csv(csv_path)
        if rows.empty:
            continue
        rows["tag"] = tag
        rows["regime"] = regime
        rows["source_file"] = str(csv_path)
        frames.append(rows)

    if not frames:
        raise RuntimeError(f"No rows.csv files found under {folder}")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data = data[data["model_type"].isin(MODEL_MAP)].copy()
    data["method"] = data["model_type"].map(MODEL_MAP)
    data["source_name"] = source_name
    data["seed"] = _numeric_column(data, "seed")
    data["w_bus"] = _numeric_column(data, "weight_bus")
    data.loc[~data["method"].isin(["MORL", "PPO-S"]), "w_bus"] = np.nan
    bus_crossing_time = _numeric_column(data, "j_bus")
    nonbus_crossing_time = _numeric_column(data, "j_all")
    bus_delay = _numeric_column(data, "Bus_delay")
    nonbus_delay = _numeric_column(data, "Car_delay")
    data["j_bus"] = bus_delay.where(bus_delay.notna(), bus_crossing_time)
    data["j_nonbus"] = nonbus_delay.where(nonbus_delay.notna(), nonbus_crossing_time)
    data["tail_nonbus"] = _numeric_column(data, "tail_cvar_10")
    data["phase_change_rate_per_min"] = _numeric_column(data, "phase_change_rate_per_min")
    data["mean_green_duration_seconds"] = _numeric_column(data, "mean_green_duration_seconds")

    networks = data.get("network", pd.Series([""] * len(data)))
    data["city"] = [
        _city_from_network_or_tag(network, tag)
        for network, tag in zip(networks, data["tag"], strict=False)
    ]
    return data[
        [
            "tag",
            "city",
            "method",
            "regime",
            "seed",
            "w_bus",
            "j_bus",
            "j_nonbus",
            "tail_nonbus",
            "phase_change_rate_per_min",
            "mean_green_duration_seconds",
            "source_name",
            "source_file",
            "model_type",
        ]
    ].copy()


def _load_inputs(
    results_root: Path,
    *,
    morl_rc4_dataset: str,
    morl_iz_dataset: str,
    ppo_rc4_dataset: str,
) -> TableInputs:
    return TableInputs(
        morl_rc4=_load_dataset(
            results_root / morl_rc4_dataset,
            regime="augmented_eval",
            source_name="morl_rc4",
        ),
        morl_iz=_load_dataset(
            results_root / morl_iz_dataset,
            regime="iz_default_shift",
            source_name="morl_iz_default",
        ),
        ppo_rc4=_load_dataset(
            results_root / ppo_rc4_dataset,
            regime="augmented_eval",
            source_name="ppo_rc4",
        ),
    )


def _nondominated_mask(points: Sequence[tuple[float, float]]) -> list[bool]:
    mask: list[bool] = []
    for idx, (bus_i, nonbus_i) in enumerate(points):
        dominated = False
        for other_idx, (bus_j, nonbus_j) in enumerate(points):
            if idx == other_idx:
                continue
            if (
                bus_j <= bus_i
                and nonbus_j <= nonbus_i
                and (bus_j < bus_i or nonbus_j < nonbus_i)
            ):
                dominated = True
                break
        mask.append(not dominated)
    return mask


def _hypervolume_2d(points: Sequence[tuple[float, float]], ref_bus: float, ref_nonbus: float) -> float:
    finite_points = [
        (float(bus), float(nonbus))
        for bus, nonbus in points
        if math.isfinite(bus)
        and math.isfinite(nonbus)
        and bus < ref_bus
        and nonbus < ref_nonbus
    ]
    if not finite_points:
        return 0.0

    nondominated = [
        point for point, keep in zip(finite_points, _nondominated_mask(finite_points), strict=True) if keep
    ]
    nondominated.sort(key=lambda point: point[0])

    hv = 0.0
    previous_nonbus = ref_nonbus
    for bus, nonbus in nondominated:
        if nonbus < previous_nonbus:
            hv += max(ref_bus - bus, 0.0) * max(previous_nonbus - nonbus, 0.0)
            previous_nonbus = nonbus
    return hv


def _dominates(candidate: tuple[float, float], baseline: tuple[float, float]) -> bool:
    return (
        candidate[0] <= baseline[0]
        and candidate[1] <= baseline[1]
        and (candidate[0] < baseline[0] or candidate[1] < baseline[1])
    )


def _aggregate_setpoints(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["tag", "city", "method", "regime", "w_bus"], dropna=False)
    out = grouped.agg(
        j_bus_mean=("j_bus", "mean"),
        j_bus_std=("j_bus", _std_or_zero),
        j_nonbus_mean=("j_nonbus", "mean"),
        j_nonbus_std=("j_nonbus", _std_or_zero),
        tail_nonbus_mean=("tail_nonbus", "mean"),
        tail_nonbus_std=("tail_nonbus", _std_or_zero),
        phase_change_rate_per_min_mean=("phase_change_rate_per_min", "mean"),
        mean_green_duration_seconds_mean=("mean_green_duration_seconds", "mean"),
        n_seeds=("seed", lambda values: int(pd.to_numeric(values, errors="coerce").dropna().nunique())),
    )
    return out.reset_index().sort_values(["tag", "regime", "method", "w_bus"], na_position="last")


def _frontier_by_seed(rows: pd.DataFrame) -> pd.DataFrame:
    result_rows: list[dict[str, object]] = []
    grouped = rows.groupby(["tag", "city", "method", "regime", "seed"], dropna=False)
    for (tag, city, method, regime, seed), group in grouped:
        points = [
            (float(bus), float(nonbus))
            for bus, nonbus in zip(group["j_bus"], group["j_nonbus"], strict=False)
            if math.isfinite(float(bus)) and math.isfinite(float(nonbus))
        ]
        if not points:
            continue
        hv = _hypervolume_2d(points, HV_REF_BUS, HV_REF_NONBUS)
        result_rows.append(
            {
                "tag": tag,
                "city": city,
                "method": method,
                "regime": regime,
                "seed": int(seed),
                "hv_pct": 100.0 * hv / (HV_REF_BUS * HV_REF_NONBUS),
                "nd_count": int(sum(_nondominated_mask(points))),
            }
        )
    return pd.DataFrame(result_rows)


def _frontier_agg(frontier: pd.DataFrame) -> pd.DataFrame:
    grouped = frontier.groupby(["tag", "city", "method", "regime"], dropna=False)
    return grouped.agg(
        hv_pct_mean=("hv_pct", "mean"),
        hv_pct_std=("hv_pct", _std_or_zero),
        nd_mean=("nd_count", "mean"),
        nd_std=("nd_count", _std_or_zero),
        n_seeds=("seed", lambda values: int(pd.to_numeric(values, errors="coerce").dropna().nunique())),
    ).reset_index()


def _dominance_by_seed(rows: pd.DataFrame) -> pd.DataFrame:
    policies = rows[rows["method"].isin(["MORL", "PPO-S"])].copy()
    baselines = rows[rows["method"].isin(["FixedTime", "RuleTSP"])].copy()

    baseline_lookup: dict[tuple[str, str, int, MethodName], tuple[float, float]] = {}
    for (tag, regime, seed, method), group in baselines.groupby(
        ["tag", "regime", "seed", "method"], dropna=False
    ):
        row = group.iloc[0]
        baseline_lookup[(str(tag), str(regime), int(seed), method)] = (
            float(row["j_bus"]),
            float(row["j_nonbus"]),
        )

    result_rows: list[dict[str, object]] = []
    for (tag, city, method, regime, seed), group in policies.groupby(
        ["tag", "city", "method", "regime", "seed"]
    ):
        seed_int = int(seed)
        points = [
            (float(bus), float(nonbus))
            for bus, nonbus in zip(group["j_bus"], group["j_nonbus"], strict=False)
            if math.isfinite(float(bus)) and math.isfinite(float(nonbus))
        ]
        ft = baseline_lookup.get((str(tag), str(regime), seed_int, "FixedTime"))
        tsp = baseline_lookup.get((str(tag), str(regime), seed_int, "RuleTSP"))
        result_rows.append(
            {
                "tag": tag,
                "city": city,
                "method": method,
                "regime": regime,
                "seed": seed_int,
                "ft_dom_count": (
                    np.nan if ft is None else int(sum(_dominates(point, ft) for point in points))
                ),
                "tsp_dom_count": (
                    np.nan if tsp is None else int(sum(_dominates(point, tsp) for point in points))
                ),
            }
        )
    return pd.DataFrame(result_rows)


def _dominance_agg(dominance: pd.DataFrame) -> pd.DataFrame:
    grouped = dominance.groupby(["tag", "city", "method", "regime"], dropna=False)
    return grouped.agg(
        ft_dom_mean=("ft_dom_count", "mean"),
        ft_dom_std=("ft_dom_count", _std_or_zero),
        tsp_dom_mean=("tsp_dom_count", "mean"),
        tsp_dom_std=("tsp_dom_count", _std_or_zero),
    ).reset_index()


def _metric_row(table: pd.DataFrame, tag: str, method: MethodName, weight: float | None) -> pd.Series:
    sub = table[table["tag"].eq(tag) & table["method"].eq(method)].copy()
    if weight is not None:
        sub = sub[sub["w_bus"].round(1).eq(weight)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.iloc[0]


def _case_table_row(
    tag: str,
    method: MethodName,
    case_agg: pd.DataFrame,
    frontier_agg: pd.DataFrame,
    dominance_agg: pd.DataFrame,
) -> str:
    if method in {"MORL", "PPO-S"}:
        frontier = frontier_agg[
            frontier_agg["tag"].eq(tag)
            & frontier_agg["regime"].eq("augmented_eval")
            & frontier_agg["method"].eq(method)
        ]
        frontier_row = frontier.iloc[0] if not frontier.empty else pd.Series(dtype=float)
        frontier_cells = [
            _format_mean_std(frontier_row.get("hv_pct_mean"), frontier_row.get("hv_pct_std")),
            _format_mean_std(frontier_row.get("nd_mean"), frontier_row.get("nd_std")),
        ]
        dom = dominance_agg[
            dominance_agg["tag"].eq(tag)
            & dominance_agg["regime"].eq("augmented_eval")
            & dominance_agg["method"].eq(method)
        ]
        dom_row = dom.iloc[0] if not dom.empty else pd.Series(dtype=float)
        frontier_cells.extend(
            [
                _format_mean_std(dom_row.get("ft_dom_mean"), dom_row.get("ft_dom_std")),
                _format_mean_std(dom_row.get("tsp_dom_mean"), dom_row.get("tsp_dom_std")),
            ]
        )
    else:
        frontier_cells = ["--", "--", "--", "--"]

    cells = [tag if method == "MORL" else "", method, *frontier_cells]
    for weight in CASE_WEIGHTS:
        row = _metric_row(case_agg, tag, method, weight if method in {"MORL", "PPO-S"} else None)
        cells.extend(
            [
                _format_mean_std(row.get("j_bus_mean"), row.get("j_bus_std")),
                _format_mean_std(row.get("j_nonbus_mean"), row.get("j_nonbus_std")),
                _format_mean_std(row.get("tail_nonbus_mean"), row.get("tail_nonbus_std")),
            ]
        )
    return " & ".join(cells) + r" \\"


def _write_case_study_table(
    output_path: Path,
    case_agg: pd.DataFrame,
    frontier_agg: pd.DataFrame,
    dominance_agg: pd.DataFrame,
    morl_rc4: pd.DataFrame,
    ppo_rc4: pd.DataFrame,
) -> None:
    del morl_rc4, ppo_rc4
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{Case-study intersections (evaluation protocol in Section~\ref{sec:exp}). Frontier quality is summarized by hypervolume (HV\%) and the number of non-dominated points (\#ND) over the 11-point preference sweep. FT-dom and TSP-dom report the mean$\pm$std number of preferences (out of 11) where the policy dominates the FixedTime or RuleTSP baseline. We also report operating-point mean delays (s) at $w_{\mathrm{bus}}\in\{0.3,0.5,0.7\}$. $J^{\mathrm{tail}}_{\mathrm{nb}}$ denotes the non-bus episode tail-delay metric (defined in Section~\ref{sec:exp})}",
        r"\label{tab:case_studies_wide}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrr|rrr|rrr|rrr}",
        r"\toprule",
        r"& & \multicolumn{4}{c|}{Frontier quality} & \multicolumn{3}{c|}{$w_{\mathrm{bus}}=0.3$} & \multicolumn{3}{c|}{$w_{\mathrm{bus}}=0.5$} & \multicolumn{3}{c}{$w_{\mathrm{bus}}=0.7$} \\",
        r"Tag & Method & HV\% & \#ND & FT-dom & TSP-dom & $J_b$ & $J_{\mathrm{nb}}$ & $J^{\mathrm{tail}}_{\mathrm{nb}}$ & $J_b$ & $J_{\mathrm{nb}}$ & $J^{\mathrm{tail}}_{\mathrm{nb}}$ & $J_b$ & $J_{\mathrm{nb}}$ & $J^{\mathrm{tail}}_{\mathrm{nb}}$ \\",
        r"\midrule",
    ]
    for idx, tag in enumerate(CASE_TAGS):
        if idx > 0:
            lines.append(r"\addlinespace")
        for method in ["MORL", "PPO-S", "FixedTime", "RuleTSP"]:
            lines.append(
                _case_table_row(
                    tag,
                    method,
                    case_agg=case_agg,
                    frontier_agg=frontier_agg,
                    dominance_agg=dominance_agg,
                )
            )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_additional_city_table(
    output_path: Path,
    additional_tags: list[str],
    morl_aug_agg: pd.DataFrame,
    frontier_agg: pd.DataFrame,
    dominance_agg: pd.DataFrame,
) -> None:
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{Additional intersections (evaluation protocol in Section~\ref{sec:exp}). MORL Pareto-set quality is summarized by HV\% and \#ND over the 11-point preference sweep ($w_{\mathrm{bus}}\in\{0.0,0.1,\dots,1.0\}$). FT-dom and TSP-dom report the mean$\pm$std number of preferences (out of 11) where MORL dominates FixedTime or RuleTSP. Baseline delays are reported as $(J_b, J_{\mathrm{nb}})$ (mean$\pm$std). MORL ranges report the min/max of the mean delay across preferences.}",
        r"\label{tab:additional_10cities}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Tag & \#ND & HV\% & FT-dom & TSP-dom & FixedTime $(J_b,J_{\mathrm{nb}})$ & RuleTSP $(J_b,J_{\mathrm{nb}})$ & MORL $J_b$ range & MORL $J_{\mathrm{nb}}$ range \\",
        r"\midrule",
    ]

    for tag in additional_tags:
        frontier = frontier_agg[
            frontier_agg["tag"].eq(tag)
            & frontier_agg["regime"].eq("augmented_eval")
            & frontier_agg["method"].eq("MORL")
        ]
        dom = dominance_agg[
            dominance_agg["tag"].eq(tag)
            & dominance_agg["regime"].eq("augmented_eval")
            & dominance_agg["method"].eq("MORL")
        ]
        fixed = _metric_row(morl_aug_agg, tag, "FixedTime", None)
        tsp = _metric_row(morl_aug_agg, tag, "RuleTSP", None)
        morl = morl_aug_agg[
            morl_aug_agg["tag"].eq(tag)
            & morl_aug_agg["regime"].eq("augmented_eval")
            & morl_aug_agg["method"].eq("MORL")
        ]
        frontier_row = frontier.iloc[0] if not frontier.empty else pd.Series(dtype=float)
        dom_row = dom.iloc[0] if not dom.empty else pd.Series(dtype=float)
        cells = [
            tag,
            _format_mean_std(frontier_row.get("nd_mean"), frontier_row.get("nd_std")),
            _format_mean_std(frontier_row.get("hv_pct_mean"), frontier_row.get("hv_pct_std")),
            _format_mean_std(dom_row.get("ft_dom_mean"), dom_row.get("ft_dom_std")),
            _format_mean_std(dom_row.get("tsp_dom_mean"), dom_row.get("tsp_dom_std")),
            (
                f"({_format_mean_std(fixed.get('j_bus_mean'), fixed.get('j_bus_std'))}, "
                f"{_format_mean_std(fixed.get('j_nonbus_mean'), fixed.get('j_nonbus_std'))})"
            ),
            (
                f"({_format_mean_std(tsp.get('j_bus_mean'), tsp.get('j_bus_std'))}, "
                f"{_format_mean_std(tsp.get('j_nonbus_mean'), tsp.get('j_nonbus_std'))})"
            ),
            _format_range(morl["j_bus_mean"].min(), morl["j_bus_mean"].max()),
            _format_range(morl["j_nonbus_mean"].min(), morl["j_nonbus_mean"].max()),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_heatmap_correlations(
    frontier_agg: pd.DataFrame,
    dominance_agg: pd.DataFrame,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    morl_frontier = frontier_agg[
        frontier_agg["method"].eq("MORL") & frontier_agg["regime"].eq("augmented_eval")
    ][["tag", "hv_pct_mean", "nd_mean"]].copy()
    morl_dominance = dominance_agg[
        dominance_agg["regime"].eq("augmented_eval") & dominance_agg["method"].eq("MORL")
    ][["tag", "ft_dom_mean", "tsp_dom_mean"]].copy()
    metrics = morl_frontier.merge(morl_dominance, on="tag", how="inner")

    feature_rows = [
        {"tag": tag, **features}
        for tag, features in STATIC_INTERSECTION_FEATURES.items()
    ]
    features = pd.DataFrame(feature_rows)
    data = metrics.merge(features, on="tag", how="inner")
    expected_tags = set(STATIC_INTERSECTION_FEATURES)
    found_tags = set(data["tag"])
    if found_tags != expected_tags:
        missing = sorted(expected_tags - found_tags)
        extra = sorted(found_tags - expected_tags)
        raise RuntimeError(f"Heatmap correlation tag mismatch. Missing={missing}, extra={extra}")

    metric_columns = ["hv_pct_mean", "nd_mean", "ft_dom_mean", "tsp_dom_mean"]
    correlations: list[tuple[str, tuple[float, float, float, float]]] = []
    for label, feature_column in HEATMAP_FEATURES:
        values = tuple(
            float(data[metric_column].corr(data[feature_column], method="spearman"))
            for metric_column in metric_columns
        )
        correlations.append((label, values))
    return correlations


def _write_heatmap_table(
    output_path: Path,
    correlations: list[tuple[str, tuple[float, float, float, float]]],
) -> None:
    lines = [
        r"% heatmap cell: soft red/blue gradient scaled by |rho|",
        "",
        r"\sisetup{round-mode=places,round-precision=2}",
        r"\newcommand{\heat}[1]{%",
        r"  \pgfmathsetmacro{\absrho}{abs(#1)}%",
        r"\pgfmathsetmacro{\pct}{ifthenelse(\absrho<0.45,0,min(100,30*\absrho))}% white below 0.45",
        r"  \ifdim #1 pt > 0pt",
        r"    \edef\heatcolor{blue!\pct!white}%",
        r"  \else",
        r"    \edef\heatcolor{red!\pct!white}%",
        r"  \fi",
        r"  \expandafter\cellcolor\expandafter{\heatcolor}\num{#1}%",
        r"}",
        "",
        r"\begin{table}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{Spearman rank correlations ($\rho$) between MORL performance summaries and static intersection features across 13 intersections. Cell color encodes sign and magnitude (blue=positive, red=negative). Significance testing is omitted/handled separately due to multiple comparisons and small $n$.}",
        r"\label{tab:spearman}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Feature & HV\% & \# ND & FT Dom & TSP Dom \\",
        r"\midrule",
    ]

    for feature_name, values in correlations:
        cells = [feature_name, *[r"\heat{" + f"{value:.2f}" + "}" for value in values]]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ITSC 2026 paper tables from eval rows.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing the MORL and PPO evaluation datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where LaTeX tables are written.",
    )
    parser.add_argument("--morl_rc4_dataset", default=MORL_RC4_DIRNAME)
    parser.add_argument("--morl_iz_dataset", default=MORL_IZ_DIRNAME)
    parser.add_argument("--ppo_rc4_dataset", default=PPO_RC4_DIRNAME)
    return parser


def _validate_setpoints(rows: pd.DataFrame, method: MethodName, label: str) -> None:
    found = sorted(
        _round_weight(weight)
        for weight in rows.loc[rows["method"].eq(method), "w_bus"].dropna().unique()
    )
    if found != EXPECTED_SETPOINTS:
        raise RuntimeError(f"{label} {method} setpoints are {found}, expected {EXPECTED_SETPOINTS}")


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _load_inputs(
        args.results_root,
        morl_rc4_dataset=args.morl_rc4_dataset,
        morl_iz_dataset=args.morl_iz_dataset,
        ppo_rc4_dataset=args.ppo_rc4_dataset,
    )
    _validate_setpoints(inputs.morl_rc4, "MORL", "morl_rc4")
    _validate_setpoints(inputs.morl_iz, "MORL", "morl_iz_default")
    _validate_setpoints(inputs.ppo_rc4, "PPO-S", "ppo_rc4")

    case_seed_rows = pd.concat(
        [
            inputs.morl_rc4[
                inputs.morl_rc4["tag"].isin(CASE_TAGS)
                & inputs.morl_rc4["method"].isin(["MORL", "FixedTime", "RuleTSP"])
            ],
            inputs.ppo_rc4[
                inputs.ppo_rc4["tag"].isin(CASE_TAGS) & inputs.ppo_rc4["method"].eq("PPO-S")
            ],
        ],
        ignore_index=True,
    )
    case_agg = _aggregate_setpoints(case_seed_rows)

    sweep_rows = pd.concat(
        [
            inputs.morl_rc4[inputs.morl_rc4["method"].eq("MORL")],
            inputs.morl_iz[inputs.morl_iz["method"].eq("MORL")],
            inputs.ppo_rc4[
                inputs.ppo_rc4["tag"].isin(CASE_TAGS) & inputs.ppo_rc4["method"].eq("PPO-S")
            ],
        ],
        ignore_index=True,
    )
    frontier_agg = _frontier_agg(_frontier_by_seed(sweep_rows))
    dominance_agg = _dominance_agg(
        _dominance_by_seed(
            pd.concat([inputs.morl_rc4, inputs.morl_iz, inputs.ppo_rc4], ignore_index=True)
        )
    )

    morl_aug_agg = _aggregate_setpoints(inputs.morl_rc4)
    additional_tags = sorted(tag for tag in inputs.morl_rc4["tag"].unique() if tag not in CASE_TAGS)

    output_paths = [
        output_dir / "tab_case_studies_wide.tex",
        output_dir / "tab_10_cities.tex",
        output_dir / "heat_map.tex",
    ]
    _write_case_study_table(
        output_paths[0],
        case_agg=case_agg,
        frontier_agg=frontier_agg,
        dominance_agg=dominance_agg,
        morl_rc4=inputs.morl_rc4,
        ppo_rc4=inputs.ppo_rc4,
    )
    _write_additional_city_table(
        output_paths[1],
        additional_tags=additional_tags,
        morl_aug_agg=morl_aug_agg,
        frontier_agg=frontier_agg,
        dominance_agg=dominance_agg,
    )
    _write_heatmap_table(
        output_paths[2],
        correlations=_build_heatmap_correlations(
            frontier_agg=frontier_agg,
            dominance_agg=dominance_agg,
        ),
    )

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
