# SPDX-FileCopyrightText: Copyright (c) 2024 Vindula Jayawardana, Baptiste Freydt, Ao Qu, Cameron Hickert, Zhongxia Yan, Cathy Wu
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Scenario generation wrapper for timetable-driven buses."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree

import numpy as np

from .utils import (
    VehicleAttributeConfig,
    iter_vehicle_elements,
    reduce_route_traffic,
    sample_vehicle_attributes,
    set_vehicle_param_elements,
    validate_traffic_multiplier,
    write_pretty_xml,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BusTimetableConfig:
    """Configuration for injecting timetable-driven bus routes.

    All time-related fields are in seconds unless otherwise noted.
    """

    # Desired concurrent bus lines present at any time.
    route_count: int = 4
    # Headway between successive buses on the same route.
    headway_min_seconds: float = 300.0
    headway_max_seconds: float = 900.0
    # Log-normal deviation applied to each planned departure.
    deviation_min_seconds: float = -120.0
    deviation_max_seconds: float = 600.0
    # Planned duration range for each bus line.
    route_duration_min_seconds: float | None = 0.0
    route_duration_max_seconds: float | None = None
    depart_lane: str = "0"
    depart_speed: str = "5"
    # Prefix used to group buses on the same timetable route.
    line_prefix: str = "tt"
    maintain_total_vehicles: bool = True
    # When True, allow reuse of the same route even if enough unique routes exist.
    allow_route_reuse: bool = False
    # Optional file for logging timetable details; defaults to sumo/bus_timetable.log.
    log_path: Path | None = None
    suppress_original_buses: bool = False
    # Optional seed for bus generation; defaults to scenario seed when None.
    bus_seed: int | None = None
    # Optional seed for selecting route edge sequences.
    # Falls back to bus_seed (or scenario seed when bus_seed is None).
    bus_route_seed: int | None = None
    # Optional seed for timetable timing randomness (headways/deviations/duration).
    # Falls back to bus_seed (or scenario seed when bus_seed is None).
    bus_timing_seed: int | None = None


class ScenarioGenerationWrapperWithBusTimetable:
    """
    Wrapper that generates scenarios and then injects additional bus vehicles
    following random fixed routes and headways (timetables).
    """

    def __init__(self, generator_module) -> None:
        self._generator = generator_module

    def generate(
        self,
        config,
        prefix: str,
        seed: int,
        task_context,
        bus_config: BusTimetableConfig,
        traffic_multiplier: float = 1.0,
        attribute_config: VehicleAttributeConfig | None = None,
    ) -> None:
        """
        Generate scenario files, then add timetable-driven buses with fixed
        routes, and inject occupancy/schedule deviation parameters.
        """
        attr_config = attribute_config or VehicleAttributeConfig()
        validate_traffic_multiplier(traffic_multiplier)

        original_get_vehicle_mix = (
            self._generator.VehicleTypeParamsSampler.get_vehicle_mix
        )
        if bus_config.suppress_original_buses:

            def no_bus_vehicle_mix(instance) -> Dict[str, float]:
                vehicle_mix = original_get_vehicle_mix(instance)
                return self._zero_bus_mix(vehicle_mix)

            self._generator.VehicleTypeParamsSampler.get_vehicle_mix = (
                no_bus_vehicle_mix
            )
        try:
            self._generator.generate_temp_sumo_files(
                config=config,
                prefix=prefix,
                seed=seed,
                task_context=task_context,
            )
        finally:
            if bus_config.suppress_original_buses:
                self._generator.VehicleTypeParamsSampler.get_vehicle_mix = (
                    original_get_vehicle_mix
                )

        route_path = config.working_dir / f"sumo/routes{prefix}.rou.xml"
        reduce_route_traffic(
            route_path,
            keep_probability=traffic_multiplier,
            seed=seed,
        )
        self._add_bus_timetable_routes(
            route_path,
            seed=seed,
            config=config,
            bus_config=bus_config,
        )
        self._inject_vehicle_attributes_with_planned_depart(
            route_path,
            seed=seed,
            config=attr_config,
        )

    @staticmethod
    def _zero_bus_mix(vehicle_mix: Dict[str, float]) -> Dict[str, float]:
        scaled = {
            name: (0.0 if name.startswith("42") else prob)
            for name, prob in vehicle_mix.items()
        }
        total = sum(scaled.values())
        if total <= 0:
            raise ValueError("Zero-bus mix results in a zero-probability mix.")
        return {name: prob / total for name, prob in scaled.items()}

    @staticmethod
    def _inject_vehicle_attributes_with_planned_depart(
        route_path,
        seed: int,
        config: VehicleAttributeConfig,
    ) -> None:
        rng = np.random.default_rng(seed)
        tree = ElementTree.parse(route_path)
        root = tree.getroot()

        for vehicle in iter_vehicle_elements(root):
            attributes = sample_vehicle_attributes(
                rng=rng,
                config=config,
            )
            planned_depart = vehicle.attrib.get("plannedDepart") or vehicle.attrib.get(
                "depart"
            )
            if planned_depart is not None:
                attributes["plannedDepart"] = planned_depart
            schedule_deviation = vehicle.attrib.get("scheduleDeviation")
            if schedule_deviation is not None:
                attributes["scheduleDeviation"] = schedule_deviation
            set_vehicle_param_elements(vehicle, attributes)

        write_pretty_xml(tree, route_path) # type: ignore

    @staticmethod
    def _resolve_route_duration_seconds(
        simulation_duration: float,
        bus_config: BusTimetableConfig,
    ) -> Tuple[float, float]:
        if (
            bus_config.route_duration_min_seconds is None
            and bus_config.route_duration_max_seconds is None
        ):
            duration_min_seconds = simulation_duration
            duration_max_seconds = simulation_duration
        else:
            duration_min_seconds = (
                max(0.0, bus_config.route_duration_min_seconds)
                if bus_config.route_duration_min_seconds is not None
                else 0.0
            )
            duration_max_seconds = (
                bus_config.route_duration_max_seconds
                if bus_config.route_duration_max_seconds is not None
                else simulation_duration
            )
        duration_max_seconds = max(duration_min_seconds, duration_max_seconds)
        duration_max_seconds = min(simulation_duration, duration_max_seconds)
        if duration_max_seconds <= 0.0:
            raise ValueError("route_duration_max_seconds must be > 0.")
        if duration_min_seconds > simulation_duration:
            raise ValueError(
                "route_duration_min_seconds exceeds simulation duration."
            )
        return duration_min_seconds, duration_max_seconds

    @staticmethod
    def _sample_lognormal_deviation_seconds(
        rng: np.random.Generator,
        min_seconds: float,
        max_seconds: float,
    ) -> float:
        if max_seconds <= 0.0 and min_seconds >= 0.0:
            raise ValueError("deviation_max_seconds must be > 0 or min < 0.")

        positive_cap = max(0.0, max_seconds)
        negative_cap = max(0.0, -min_seconds)

        if positive_cap == 0.0 and negative_cap == 0.0:
            return 0.0

        if positive_cap > 0.0 and negative_cap > 0.0:
            negative_prob = negative_cap / (positive_cap + negative_cap)
        else:
            negative_prob = 1.0 if negative_cap > 0.0 else 0.0

        magnitude_cap = max(positive_cap, negative_cap)
        # Fit a log-normal so that ~1% is near 1% of cap and ~99% near cap.
        lower = max(magnitude_cap * 0.01, 1e-3)
        upper = magnitude_cap
        z_99 = 2.3263478740408408
        mu = (np.log(upper) + np.log(lower)) / 2.0
        sigma = (np.log(upper) - np.log(lower)) / (2.0 * z_99)

        magnitude = float(rng.lognormal(mean=mu, sigma=sigma))
        if magnitude > magnitude_cap:
            magnitude = magnitude_cap

        deviation = -magnitude if rng.random() < negative_prob else magnitude
        return float(min(max(deviation, min_seconds), max_seconds))

    @staticmethod
    def _choose_routes(
        available_routes: List[str],
        route_count: int,
        allow_route_reuse: bool,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, bool]:
        replace = allow_route_reuse or route_count > len(available_routes)
        chosen_routes = rng.choice(
            available_routes, size=route_count, replace=replace
        )
        return chosen_routes, replace

    @staticmethod
    def _resolve_rngs(
        bus_config: BusTimetableConfig,
        seed: int,
    ) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator, int, int]:
        """
        Resolve RNGs for route selection and timetable timing.

        Backwards compatibility:
        - If neither split seed is set, keep legacy single-RNG behavior so results
          remain unchanged for existing configs.
        """
        legacy_seed = bus_config.bus_seed if bus_config.bus_seed is not None else seed
        route_seed = (
            bus_config.bus_route_seed
            if bus_config.bus_route_seed is not None
            else legacy_seed
        )
        timing_seed = (
            bus_config.bus_timing_seed
            if bus_config.bus_timing_seed is not None
            else legacy_seed
        )

        if bus_config.bus_route_seed is None and bus_config.bus_timing_seed is None:
            legacy_rng = np.random.default_rng(legacy_seed)
            return legacy_rng, legacy_rng, legacy_rng, route_seed, timing_seed

        route_rng = np.random.default_rng(route_seed)
        timing_rng = np.random.default_rng(timing_seed)
        return route_rng, timing_rng, timing_rng, route_seed, timing_seed

    @staticmethod
    def _add_bus_timetable_routes(
        route_path,
        seed: int,
        config,
        bus_config: BusTimetableConfig,
    ) -> None:
        from intersection_zoo.sumo_adapter.vehicle_mix import VehicleTypeParamsSampler

        route_rng, timing_rng, removal_rng, route_seed, timing_seed = (
            ScenarioGenerationWrapperWithBusTimetable._resolve_rngs(
                bus_config=bus_config,
                seed=seed,
            )
        )
        tree = ElementTree.parse(route_path)
        root = tree.getroot()

        _ensure_bus_logger(bus_config, config.working_dir)
        min_duration_str = (
            f"{bus_config.route_duration_min_seconds:.2f}"
            if bus_config.route_duration_min_seconds is not None
            else "sim_end"
        )
        max_duration_str = (
            f"{bus_config.route_duration_max_seconds:.2f}"
            if bus_config.route_duration_max_seconds is not None
            else "sim_end"
        )
        logger.info(
            "Bus timetable config routes=%s headway=%.2f-%.2f s deviation=%.2f-%.2f s "
            "route_duration=%s-%s s lane=%s speed=%s line_prefix=%s "
            "maintain_total=%s allow_route_reuse=%s suppress_original=%s "
            "bus_seed=%s route_seed=%s timing_seed=%s",
            bus_config.route_count,
            bus_config.headway_min_seconds,
            bus_config.headway_max_seconds,
            bus_config.deviation_min_seconds,
            bus_config.deviation_max_seconds,
            min_duration_str,
            max_duration_str,
            bus_config.depart_lane,
            bus_config.depart_speed,
            bus_config.line_prefix,
            bus_config.maintain_total_vehicles,
            bus_config.allow_route_reuse,
            bus_config.suppress_original_buses,
            bus_config.bus_seed,
            route_seed,
            timing_seed,
        )

        available_routes = _collect_unique_routes_with_tl(root)
        if not available_routes:
            raise ValueError("No routes crossing TL were found in the route file.")

        concurrent_lines = max(0, bus_config.route_count)
        if concurrent_lines == 0:
            return

        duration_min_seconds, duration_max_seconds = (
            ScenarioGenerationWrapperWithBusTimetable._resolve_route_duration_seconds(
                config.simulation_duration,
                bus_config,
            )
        )

        avg_duration_seconds = (duration_min_seconds + duration_max_seconds) / 2.0
        if avg_duration_seconds <= 0.0:
            raise ValueError("Average route duration must be > 0.")
        if duration_min_seconds >= config.simulation_duration:
            cycles = 1
        else:
            cycles = int(np.ceil(config.simulation_duration / avg_duration_seconds))
        total_route_count = max(1, cycles * concurrent_lines)
        logger.info(
            "Selected %s concurrent lines, duration=%.1f-%.1f s, "
            "avg_duration=%.1f s, cycles=%s, total routes=%s.",
            concurrent_lines,
            duration_min_seconds,
            duration_max_seconds,
            avg_duration_seconds,
            cycles,
            total_route_count,
        )

        chosen_routes, replace = (
            ScenarioGenerationWrapperWithBusTimetable._choose_routes(
                available_routes,
                total_route_count,
                bus_config.allow_route_reuse,
                route_rng,
            )
        )
        logger.info(
            "Selected %s routes from %s available (replace=%s).",
            total_route_count,
            len(available_routes),
            replace,
        )

        params_sampler = VehicleTypeParamsSampler()
        vehicle_mix = params_sampler.get_vehicle_mix()
        bus_mix = {
            name: prob
            for name, prob in vehicle_mix.items()
            if name.startswith("42")
        }
        if not bus_mix:
            raise ValueError("No bus vehicle types found in vehicle mix.")
        bus_type_names = list(bus_mix.keys())
        bus_type_probs = np.array(list(bus_mix.values()), dtype=float)
        bus_type_probs = bus_type_probs / bus_type_probs.sum()

        bus_vehicles: List[ElementTree.Element] = []
        deviations_seconds: List[float] = []
        start_spacing_seconds = (
            config.simulation_duration / total_route_count
            if total_route_count > 0
            else config.simulation_duration
        )
        start_times = [
            i * start_spacing_seconds for i in range(total_route_count)
        ]
        for route_index, route_edges in enumerate(chosen_routes):
            start_time_seconds = start_times[route_index]
            if duration_max_seconds > duration_min_seconds:
                duration_seconds = timing_rng.uniform(
                    duration_min_seconds, duration_max_seconds
                )
            else:
                duration_seconds = duration_min_seconds
            end_time_seconds = min(
                config.simulation_duration, start_time_seconds + duration_seconds
            )

            headway_seconds = timing_rng.uniform(
                bus_config.headway_min_seconds, bus_config.headway_max_seconds
            )
            time_seconds = start_time_seconds
            bus_idx = 0
            schedule_times: List[float] = []
            actual_depart_times: List[float] = []
            while time_seconds <= end_time_seconds:
                deviation_seconds = (
                    ScenarioGenerationWrapperWithBusTimetable._sample_lognormal_deviation_seconds(
                        timing_rng,
                        bus_config.deviation_min_seconds,
                        bus_config.deviation_max_seconds,
                    )
                )
                depart_time = min(
                    config.simulation_duration,
                    max(0.0, time_seconds + deviation_seconds),
                )

                vehicle_type = timing_rng.choice(bus_type_names, p=bus_type_probs)
                vehicle_id = (
                    f"human_{vehicle_type}_TT{route_index}_{bus_config.depart_lane}_{bus_idx}"
                )
                v_type_id = f"vType_{vehicle_id}"
                root.insert(
                    _vtype_insert_index(root),
                    ElementTree.Element(
                        "vType",
                        attrib={
                            "id": v_type_id,
                            "color": "1,1,0",
                            **params_sampler.sample_idm_params(vehicle_type),
                        },
                    ),
                )
                route_elem = ElementTree.Element(
                    "route", attrib={"id": vehicle_id, "edges": route_edges}
                )
                vehicle = ElementTree.Element(
                    "vehicle",
                    attrib={
                        "id": vehicle_id,
                        "type": v_type_id,
                        "depart": str(depart_time),
                        "departLane": bus_config.depart_lane,
                        "departSpeed": bus_config.depart_speed,
                        "plannedDepart": str(time_seconds),
                        "scheduleDeviation": f"{deviation_seconds:.3f}",
                        "line": f"{bus_config.line_prefix}_{route_index}",
                    },
                )
                ElementTree.SubElement(
                    vehicle,
                    "param",
                    {"key": "timetableRoute", "value": str(route_index)},
                )
                vehicle.append(route_elem)
                bus_vehicles.append(vehicle)
                deviations_seconds.append(deviation_seconds)
                schedule_times.append(time_seconds)
                actual_depart_times.append(depart_time)

                time_seconds += headway_seconds
                bus_idx += 1

            logger.info(
                "Bus timetable route %s edges=%s headway=%.2f s window=%.1f-%.1f s "
                "duration=%.1f s planned=%s actual=%s",
                route_index,
                route_edges,
                headway_seconds,
                start_time_seconds,
                end_time_seconds,
                duration_seconds,
                ", ".join(f"{t:.1f}" for t in schedule_times),
                ", ".join(f"{t:.1f}" for t in actual_depart_times),
            )

        if not bus_vehicles:
            return

        delays_seconds = np.maximum(0.0, np.array(deviations_seconds, dtype=float))
        if delays_seconds.size > 0:
            max_delay = float(delays_seconds.max(initial=0.0))
            if max_delay <= 0.0:
                max_delay = 1.0
            bins = 10
            counts, edges = np.histogram(
                delays_seconds, bins=bins, range=(0.0, max_delay)
            )
            bucket_labels = [
                f"[{edges[i]:.1f}, {edges[i + 1]:.1f})"
                for i in range(len(edges) - 1)
            ]
            histogram_lines = [
                f"{label}: {int(count)}"
                for label, count in zip(bucket_labels, counts)
            ]
            logger.info(
                "Delay histogram (s) bins=%s max=%.1f avg=%.1f median=%.1f: %s",
                bins,
                max_delay,
                float(delays_seconds.mean()),
                float(np.median(delays_seconds)),
                "; ".join(histogram_lines),
            )

        non_vehicle_elements = [child for child in root if child.tag != "vehicle"]
        vehicle_elements = [child for child in root if child.tag == "vehicle"]

        if bus_config.maintain_total_vehicles:
            removal_pool = [
                v
                for v in vehicle_elements
                if not v.attrib.get("id", "").startswith("rl_")
            ]
            removal_count = min(len(bus_vehicles), len(removal_pool))
            if removal_count > 0:
                removed_indices = removal_rng.choice(
                    len(removal_pool), size=removal_count, replace=False
                )
                removed = [removal_pool[int(i)] for i in removed_indices]
                removed_ids = {v.attrib.get("id") for v in removed}
                removed_vtype_ids = {
                    v.attrib.get("type") for v in removed if v.attrib.get("type")
                }
                vehicle_elements = [
                    v
                    for v in vehicle_elements
                    if v.attrib.get("id") not in removed_ids
                ]
                non_vehicle_elements = [
                    element
                    for element in non_vehicle_elements
                    if not (
                        element.tag == "vType"
                        and element.attrib.get("id") in removed_vtype_ids
                    )
                ]
                logger.info(
                    "Removed %s existing vehicles to keep totals steady.",
                    removal_count,
                )

        vehicle_elements.extend(bus_vehicles)
        vehicle_elements.sort(key=lambda e: float(e.attrib["depart"]))

        root.clear()
        for element in non_vehicle_elements:
            root.append(element)
        for vehicle in vehicle_elements:
            root.append(vehicle)

        write_pretty_xml(tree, route_path) # type: ignore


def _collect_unique_routes_with_tl(
    root: ElementTree.Element,
) -> List[str]:
    routes: List[str] = []
    for vehicle in iter_vehicle_elements(root):
        for child in vehicle:
            if child.tag != "route":
                continue
            edges = child.attrib.get("edges")
            if edges and "TL" in edges and edges not in routes:
                routes.append(edges)
            break
    return routes


def _vtype_insert_index(root: ElementTree.Element) -> int:
    insert_index = 0
    for element in root:
        if element.tag != "vType":
            break
        insert_index += 1
    return insert_index


def _ensure_bus_logger(bus_config: BusTimetableConfig, working_dir: Path) -> None:
    log_path = bus_config.log_path
    if log_path is None:
        log_path = working_dir / "sumo" / "bus_timetable.log"
    if any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == Path(log_path)
        for handler in logger.handlers
    ):
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
