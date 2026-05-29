#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

DelayMode = Literal["fixed", "laneff"]
MethodName = Literal["morl", "ppo", "fixed_ts", "tsp"]

ITSC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ITSC_ROOT.parent
RESULTS_ROOT = ITSC_ROOT / "results" / "evaluations"
FIGURE_ROOT = ITSC_ROOT / "results" / "figures"
IEEE_FONT_PATH = Path("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf")

MORL_COLOR = "#2563eb"
PPO_COLOR = "#64748b"
SHIFT_COLOR = "#e11d48"
FT_COLOR = "#c0392b"
TSP_COLOR = "#0f766e"
BUS_COLOR = "#b91c1c"
CAR_COLOR = "#1d4ed8"
PPO_BUS_COLOR = "#7f1d1d"
PPO_CAR_COLOR = "#1e3a8a"


@dataclass(frozen=True)
class CityPlotConfig:
    tag: str
    label: str
    ff_bus: float
    ff_car: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    axis_ticks: list[float]


CITIES: list[CityPlotConfig] = [
    CityPlotConfig(
        tag="nyc10802",
        label="(a) New York / 10802",
        ff_bus=19.565446,
        ff_car=17.418672,
        x_min=5,
        y_min=5,
        x_max=30,
        y_max=30,
        axis_ticks=[5, 10, 15, 20, 25, 30],
    ),
    CityPlotConfig(
        tag="la2114",
        label="(b) Los Angeles / 2114",
        ff_bus=13.982103,
        ff_car=13.947834,
        x_min=15,
        y_min=15,
        x_max=60,
        y_max=60,
        axis_ticks=[15, 30, 45, 60],
    ),
    CityPlotConfig(
        tag="chi2412",
        label="(c) Chicago / 2412",
        ff_bus=22.327909,
        ff_car=22.191380,
        x_min=10,
        y_min=10,
        x_max=150,
        y_max=150,
        axis_ticks=[10, 50, 100, 150],
    ),
]


def _configure_matplotlib() -> None:
    if IEEE_FONT_PATH.exists():
        matplotlib.font_manager.fontManager.addfont(str(IEEE_FONT_PATH))
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "TeX Gyre Termes", "Times"],
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
        }
    )


def _rows_path(dataset: str, city: str) -> Path:
    return RESULTS_ROOT / dataset / city / "rows.csv"


def _with_delay_columns(
    data: pd.DataFrame,
    *,
    ff_bus: float,
    ff_car: float,
    delay_mode: DelayMode,
) -> pd.DataFrame:
    out = data.copy()
    out["bus_delay"] = out["j_bus"] - ff_bus
    out["car_delay"] = out["j_all"] - ff_car
    if delay_mode == "laneff":
        if "d_bus" in out.columns:
            bus_mask = out["d_bus"].notna()
            out.loc[bus_mask, "bus_delay"] = out.loc[bus_mask, "d_bus"]
        if "d_all" in out.columns:
            car_mask = out["d_all"].notna()
            out.loc[car_mask, "car_delay"] = out.loc[car_mask, "d_all"]
    return out


def _load_rows(dataset: str, city: CityPlotConfig, delay_mode: DelayMode) -> pd.DataFrame:
    path = _rows_path(dataset, city.tag)
    if not path.exists():
        raise FileNotFoundError(path)
    return _with_delay_columns(
        pd.read_csv(path),
        ff_bus=city.ff_bus,
        ff_car=city.ff_car,
        delay_mode=delay_mode,
    )


def _aggregate_curve(data: pd.DataFrame, model_type: MethodName) -> pd.DataFrame:
    sub = data[data["model_type"] == model_type].copy()
    if "weight_bus" in sub.columns:
        sub = sub[sub["weight_bus"].notna()]
    sub = sub.dropna(subset=["bus_delay", "car_delay", "weight_bus"])
    grouped = (
        sub.groupby("weight_bus")
        .agg(
            bus_mean=("bus_delay", "mean"),
            bus_std=("bus_delay", "std"),
            car_mean=("car_delay", "mean"),
            car_std=("car_delay", "std"),
            n=("bus_delay", "count"),
        )
        .reset_index()
        .sort_values("weight_bus")
    )
    grouped["bus_std"] = grouped["bus_std"].fillna(0.0)
    grouped["car_std"] = grouped["car_std"].fillna(0.0)
    return grouped


def _load_baseline(data: pd.DataFrame, model_type: MethodName) -> tuple[tuple[float, float], tuple[float, float]]:
    sub = data[data["model_type"] == model_type].dropna(subset=["bus_delay", "car_delay"])
    if len(sub) == 0:
        return (np.nan, np.nan), (0.0, 0.0)
    return (
        (float(sub["car_delay"].mean()), float(sub["bus_delay"].mean())),
        (
            float(sub["car_delay"].std(ddof=1)) if len(sub) > 1 else 0.0,
            float(sub["bus_delay"].std(ddof=1)) if len(sub) > 1 else 0.0,
        ),
    )


def _non_dominated(car_delay: np.ndarray, bus_delay: np.ndarray) -> np.ndarray:
    mask = np.ones(len(car_delay), dtype=bool)
    for idx in range(len(car_delay)):
        for other_idx in range(len(car_delay)):
            if idx == other_idx:
                continue
            dominates = (
                car_delay[other_idx] <= car_delay[idx]
                and bus_delay[other_idx] <= bus_delay[idx]
                and (car_delay[other_idx] < car_delay[idx] or bus_delay[other_idx] < bus_delay[idx])
            )
            if dominates:
                mask[idx] = False
                break
    return mask


def _plot_frontier(
    ax: Axes,
    curve: pd.DataFrame,
    *,
    color: str,
    marker: str,
    marker_size: float,
    filled: bool,
    line_style: str = "-",
    alpha: float = 1.0,
) -> int:
    car = curve["car_mean"].to_numpy(dtype=float)
    bus = curve["bus_mean"].to_numpy(dtype=float)
    nd = _non_dominated(car, bus)

    nd_car = car[nd]
    nd_bus = bus[nd]
    order = np.argsort(nd_car)
    if len(order) >= 2:
        step_x: list[float] = [float(nd_car[order[0]])]
        step_y: list[float] = [float(nd_bus[order[0]])]
        for idx in order[1:]:
            step_x.extend([float(nd_car[idx]), float(nd_car[idx])])
            step_y.extend([step_y[-1], float(nd_bus[idx])])
        ax.plot(step_x, step_y, color=color, lw=1.8, ls=line_style, zorder=3, alpha=0.85 * alpha)

    fill = color if filled else "white"
    if (~nd).any():
        ax.scatter(
            car[~nd],
            bus[~nd],
            c=fill,
            marker=marker,
            s=40,
            edgecolors=color,
            linewidths=0.5,
            alpha=0.28 * alpha,
            zorder=5,
        )
    if nd.any():
        ax.errorbar(
            car[nd],
            bus[nd],
            xerr=curve["car_std"].to_numpy(dtype=float)[nd],
            yerr=curve["bus_std"].to_numpy(dtype=float)[nd],
            fmt="none",
            ecolor=color,
            elinewidth=0.6,
            alpha=0.45 * alpha,
            capsize=1.5,
            capthick=0.35,
            zorder=4,
        )
        ax.scatter(
            car[nd],
            bus[nd],
            c=fill,
            marker=marker,
            s=marker_size,
            edgecolors=color,
            linewidths=1.6,
            alpha=alpha,
            zorder=6,
        )
    return int(nd.sum())


def _plot_baselines(ax: Axes, rows: pd.DataFrame) -> None:
    for model_type, marker, color, label, size in [
        ("fixed_ts", "X", FT_COLOR, "FT", 85),
        ("tsp", "^", TSP_COLOR, "TSP", 70),
    ]:
        mean, std = _load_baseline(rows, model_type)  # type: ignore[arg-type]
        if np.isnan(mean[0]):
            continue
        ax.errorbar(
            mean[0],
            mean[1],
            xerr=std[0],
            yerr=std[1],
            fmt=marker,
            color=color,
            markersize=size / 10,
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=0.6,
            alpha=0.85,
            capsize=1.8,
            capthick=0.4,
            zorder=7,
        )
        ax.annotate(
            label,
            mean,
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=7.2,
            color=color,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        )


def _format_tradeoff_axes(ax: Axes, city: CityPlotConfig) -> None:
    ax.set_xlim(city.x_min, city.x_max)
    ax.set_ylim(city.y_min, city.y_max)
    ax.set_xticks(city.axis_ticks)
    ax.set_yticks(city.axis_ticks)
    ax.set_xlabel("Mean non-bus delay (s)", fontsize=10.6)
    ax.set_ylabel("Mean bus delay (s)", fontsize=10.6)
    ax.set_title(city.label, fontsize=11.6, fontweight="bold", pad=6)
    ax.grid(True, alpha=0.10, lw=0.3)
    ax.tick_params(labelsize=8.8)


def _save_figure(fig: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    delay_mode: DelayMode,
    output_dir: Path,
    *,
    morl_rc4_dataset: str = "morl_rc4_dataset",
    ppo_rc4_dataset: str = "ppo_rc4_dataset",
) -> Path:
    fig = plt.figure(figsize=(7.6, 3.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.20)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, 1)

    for idx, city in enumerate(CITIES):
        ax = axes[idx]
        morl_rows = _load_rows(morl_rc4_dataset, city, delay_mode)
        ppo_rows = _load_rows(ppo_rc4_dataset, city, delay_mode)
        morl_curve = _aggregate_curve(morl_rows, "morl")
        ppo_curve = _aggregate_curve(ppo_rows, "ppo")

        morl_car = morl_curve["car_mean"].to_numpy(dtype=float)
        morl_bus = morl_curve["bus_mean"].to_numpy(dtype=float)
        morl_weights = morl_curve["weight_bus"].to_numpy(dtype=float)
        morl_nd = _non_dominated(morl_car, morl_bus)

        nd_car = morl_car[morl_nd]
        nd_bus = morl_bus[morl_nd]
        nd_order = np.argsort(nd_car)
        if len(nd_order) >= 2:
            step_x = [float(nd_car[nd_order[0]])]
            step_y = [float(nd_bus[nd_order[0]])]
            for point_idx in nd_order[1:]:
                step_x.extend([float(nd_car[point_idx]), float(nd_car[point_idx])])
                step_y.extend([step_y[-1], float(nd_bus[point_idx])])
            ax.plot(
                step_x,
                step_y,
                color="#154360",
                lw=2.0,
                solid_capstyle="round",
                zorder=3,
                alpha=0.85,
            )

        if morl_nd.any():
            ax.errorbar(
                morl_car[morl_nd],
                morl_bus[morl_nd],
                xerr=morl_curve["car_std"].to_numpy(dtype=float)[morl_nd],
                yerr=morl_curve["bus_std"].to_numpy(dtype=float)[morl_nd],
                fmt="none",
                ecolor="#555",
                elinewidth=0.6,
                alpha=0.45,
                zorder=4,
                capsize=1.5,
                capthick=0.35,
            )
            ax.scatter(
                morl_car[morl_nd],
                morl_bus[morl_nd],
                c=morl_weights[morl_nd],
                cmap=cmap,
                norm=norm,
                s=95,
                edgecolors="#154360",
                linewidths=1.8,
                zorder=6,
            )
        if (~morl_nd).any():
            ax.scatter(
                morl_car[~morl_nd],
                morl_bus[~morl_nd],
                c=morl_weights[~morl_nd],
                cmap=cmap,
                norm=norm,
                s=40,
                edgecolors="white",
                linewidths=0.4,
                zorder=5,
                alpha=0.30,
            )

        ppo_car = ppo_curve["car_mean"].to_numpy(dtype=float)
        ppo_bus = ppo_curve["bus_mean"].to_numpy(dtype=float)
        ppo_nd = _non_dominated(ppo_car, ppo_bus)
        ppo_nd_car = ppo_car[ppo_nd]
        ppo_nd_bus = ppo_bus[ppo_nd]
        ppo_order = np.argsort(ppo_nd_car)
        if len(ppo_order) >= 2:
            step_x = [float(ppo_nd_car[ppo_order[0]])]
            step_y = [float(ppo_nd_bus[ppo_order[0]])]
            for point_idx in ppo_order[1:]:
                step_x.extend([float(ppo_nd_car[point_idx]), float(ppo_nd_car[point_idx])])
                step_y.extend([step_y[-1], float(ppo_nd_bus[point_idx])])
            ax.plot(
                step_x,
                step_y,
                color=PPO_COLOR,
                lw=1.5,
                ls="--",
                solid_capstyle="round",
                zorder=3,
                alpha=0.7,
            )

        if (~ppo_nd).any():
            ax.scatter(
                ppo_car[~ppo_nd],
                ppo_bus[~ppo_nd],
                color=PPO_COLOR,
                s=40,
                marker="s",
                alpha=0.30,
                edgecolors="white",
                linewidths=0.4,
                zorder=5,
            )
        if ppo_nd.any():
            ax.errorbar(
                ppo_car[ppo_nd],
                ppo_bus[ppo_nd],
                xerr=ppo_curve["car_std"].to_numpy(dtype=float)[ppo_nd],
                yerr=ppo_curve["bus_std"].to_numpy(dtype=float)[ppo_nd],
                fmt="s",
                color=PPO_COLOR,
                ms=6.8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                ecolor=PPO_COLOR,
                elinewidth=0.7,
                alpha=0.80,
                zorder=7,
                capsize=1.5,
                capthick=0.35,
            )

        _plot_baselines(ax, morl_rows)
        _format_tradeoff_axes(ax, city)
        if idx > 0:
            ax.set_ylabel("")
        ax.text(
            0.03,
            0.97,
            f"MORL: {int(morl_nd.sum())} ND | PPO-S: {int(ppo_nd.sum())} ND",
            transform=ax.transAxes,
            fontsize=9.2,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbb", alpha=0.90, lw=0.5),
            zorder=10,
        )
        ax.annotate(
            "",
            xy=(0.04, 0.06),
            xytext=(0.15, 0.17),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=0.9, mutation_scale=8),
        )
        ax.text(
            0.10,
            0.18,
            "better",
            transform=ax.transAxes,
            fontsize=6.2,
            color="#aaa",
            ha="center",
            fontstyle="italic",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(0.3),
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.3,
            alpha=0.35,
            label="MORL dominated",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(0.5),
            markersize=6,
            markeredgecolor="#154360",
            markeredgewidth=1.3,
            label="MORL non-dominated",
        ),
        Line2D([0], [0], color="#154360", lw=1.8, label="MORL Pareto front"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=PPO_COLOR,
            markersize=5,
            markeredgecolor="white",
            label="PPO-S",
        ),
        Line2D([0], [0], color=PPO_COLOR, lw=1.5, ls="--", label="PPO-S Pareto front"),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor=FT_COLOR,
            markersize=7,
            markeredgecolor="white",
            label="Fixed-time",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=TSP_COLOR,
            markersize=6,
            markeredgecolor="white",
            label="TSP",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9.4,
        frameon=True,
        framealpha=0.92,
        edgecolor="#ccc",
        bbox_to_anchor=(0.48, -0.04),
        columnspacing=0.72,
        handletextpad=0.3,
    )
    fig.subplots_adjust(bottom=0.27, top=0.89, left=0.07, right=0.968)

    panel_pos = axes[-1].get_position()
    colorbar_ax = fig.add_axes([panel_pos.x1 + 0.008, panel_pos.y0, 0.012, panel_pos.height])
    scalar_mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax)
    colorbar.set_label("$w_{\\mathrm{bus}}$", fontsize=12.0, labelpad=1)
    colorbar.ax.tick_params(labelsize=10.4)

    output_path = output_dir / "pareto_tradeoff.svg"
    _save_figure(fig, output_path)
    return output_path


def plot_shift(
    delay_mode: DelayMode,
    output_dir: Path,
    *,
    morl_rc4_dataset: str = "morl_rc4_dataset",
    morl_iz_dataset: str = "morl_iz_default_dataset",
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2))
    for idx, city in enumerate(CITIES):
        ax = axes[idx]
        rc4_rows = _load_rows(morl_rc4_dataset, city, delay_mode)
        iz_rows = _load_rows(morl_iz_dataset, city, delay_mode)
        rc4_curve = _aggregate_curve(rc4_rows, "morl")
        iz_curve = _aggregate_curve(iz_rows, "morl")
        rc4_nd = _plot_frontier(
            ax, rc4_curve, color=MORL_COLOR, marker="o", marker_size=82, filled=True
        )
        iz_nd = _plot_frontier(
            ax,
            iz_curve,
            color=SHIFT_COLOR,
            marker="D",
            marker_size=62,
            filled=False,
            line_style="--",
            alpha=0.95,
        )
        _plot_baselines(ax, rc4_rows)
        _format_tradeoff_axes(ax, city)
        if idx > 0:
            ax.set_ylabel("")
        ax.text(
            0.03,
            0.97,
            f"RC4: {rc4_nd} ND | IZ: {iz_nd} ND",
            transform=ax.transAxes,
            fontsize=8.6,
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", alpha=0.90, lw=0.5),
        )

    handles = [
        Line2D([0], [0], marker="o", color=MORL_COLOR, markerfacecolor=MORL_COLOR, markersize=6, label="RC4"),
        Line2D([0], [0], marker="D", color=SHIFT_COLOR, markerfacecolor="white", markersize=6, label="IZ default"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=FT_COLOR, markeredgecolor=FT_COLOR, markersize=7, label="Fixed-time"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=TSP_COLOR, markeredgecolor=TSP_COLOR, markersize=7, label="TSP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.8, framealpha=0.95)
    fig.subplots_adjust(bottom=0.22, wspace=0.24)
    output_path = output_dir / "scenario_shift.svg"
    _save_figure(fig, output_path)
    return output_path


def plot_tunability(
    delay_mode: DelayMode,
    output_dir: Path,
    *,
    morl_rc4_dataset: str = "morl_rc4_dataset",
    ppo_rc4_dataset: str = "ppo_rc4_dataset",
) -> Path:
    city = CITIES[0]
    morl_rows = _load_rows(morl_rc4_dataset, city, delay_mode)
    ppo_rows = _load_rows(ppo_rc4_dataset, city, delay_mode)
    morl_curve = _aggregate_curve(morl_rows, "morl")
    ppo_curve = _aggregate_curve(ppo_rows, "ppo")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    weights = morl_curve["weight_bus"].to_numpy(dtype=float)
    ax.plot(weights, morl_curve["bus_mean"], "o-", color=BUS_COLOR, ms=4, lw=1.7, label="MORL bus")
    ax.fill_between(weights, morl_curve["bus_mean"] - morl_curve["bus_std"], morl_curve["bus_mean"] + morl_curve["bus_std"], color=BUS_COLOR, alpha=0.14)
    ax.plot(weights, morl_curve["car_mean"], "o-", color=CAR_COLOR, ms=4, lw=1.7, label="MORL non-bus")
    ax.fill_between(weights, morl_curve["car_mean"] - morl_curve["car_std"], morl_curve["car_mean"] + morl_curve["car_std"], color=CAR_COLOR, alpha=0.14)

    ppo_weights = ppo_curve["weight_bus"].to_numpy(dtype=float)
    ax.plot(ppo_weights, ppo_curve["bus_mean"], "s--", color=PPO_BUS_COLOR, ms=3.8, lw=1.2, label="PPO bus")
    ax.fill_between(ppo_weights, ppo_curve["bus_mean"] - ppo_curve["bus_std"], ppo_curve["bus_mean"] + ppo_curve["bus_std"], color=PPO_BUS_COLOR, alpha=0.10)
    ax.plot(ppo_weights, ppo_curve["car_mean"], "s--", color=PPO_CAR_COLOR, ms=3.8, lw=1.2, label="PPO non-bus")
    ax.fill_between(ppo_weights, ppo_curve["car_mean"] - ppo_curve["car_std"], ppo_curve["car_mean"] + ppo_curve["car_std"], color=PPO_CAR_COLOR, alpha=0.10)

    fixed_mean, _fixed_std = _load_baseline(morl_rows, "fixed_ts")
    tsp_mean, _tsp_std = _load_baseline(morl_rows, "tsp")
    if not np.isnan(fixed_mean[1]):
        ax.axhline(fixed_mean[1], color=FT_COLOR, ls="--", lw=0.9, alpha=0.7, label="FT bus")
    if not np.isnan(tsp_mean[1]):
        ax.axhline(tsp_mean[1], color=TSP_COLOR, ls=":", lw=1.1, alpha=0.8, label="TSP bus")

    ax.set_xlabel("$w_{\\mathrm{bus}}$", fontsize=9)
    ax.set_ylabel("Mean delay (s)", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.12, lw=0.3)
    ax.tick_params(labelsize=8)
    ax.legend(
        fontsize=7.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=3,
        framealpha=0.92,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(bottom=0.30)
    output_path = output_dir / "weight_tunability.svg"
    _save_figure(fig, output_path)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the three ITSC 2026 publication figures from new eval rows.")
    parser.add_argument("--delay-mode", choices=["fixed", "laneff"], default="fixed")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--morl-rc4-dataset", default="morl_rc4_dataset")
    parser.add_argument("--morl-iz-dataset", default="morl_iz_default_dataset")
    parser.add_argument("--ppo-rc4-dataset", default="ppo_rc4_dataset")
    args = parser.parse_args(argv)

    _configure_matplotlib()
    paths = [
        plot_pareto(
            args.delay_mode,
            args.output_dir,
            morl_rc4_dataset=args.morl_rc4_dataset,
            ppo_rc4_dataset=args.ppo_rc4_dataset,
        ),
        plot_shift(
            args.delay_mode,
            args.output_dir,
            morl_rc4_dataset=args.morl_rc4_dataset,
            morl_iz_dataset=args.morl_iz_dataset,
        ),
        plot_tunability(
            args.delay_mode,
            args.output_dir,
            morl_rc4_dataset=args.morl_rc4_dataset,
            ppo_rc4_dataset=args.ppo_rc4_dataset,
        ),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
