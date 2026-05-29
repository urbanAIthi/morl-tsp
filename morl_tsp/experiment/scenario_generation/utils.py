# SPDX-FileCopyrightText: Copyright (c) 2024 Vindula Jayawardana, Baptiste Freydt, Ao Qu, Cameron Hickert, Zhongxia Yan, Cathy Wu
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Shared helpers for scenario generation wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
from xml.etree import ElementTree

import numpy as np


@dataclass(frozen=True)
class VehicleAttributeConfig:
    occupancy_zero_prob: float = 0.3
    occupancy_lognormal_mean: float = 2.0
    occupancy_lognormal_sigma: float = 0.5
    occupancy_max: int = 50
    # Schedule deviation parameters are in seconds.
    schedule_deviation_mean: float = 0.0
    schedule_deviation_std: float = 5.0


def iter_vehicle_elements(root: ElementTree.Element) -> Iterable[ElementTree.Element]:
    for element in root:
        if element.tag == "vehicle":
            yield element


def sample_vehicle_attributes(
    rng: np.random.Generator,
    config: VehicleAttributeConfig,
) -> Dict[str, str]:
    if rng.random() < config.occupancy_zero_prob:
        occupancy = 0
    else:
        occupancy = int(
            round(
                rng.lognormal(
                    mean=config.occupancy_lognormal_mean,
                    sigma=config.occupancy_lognormal_sigma,
                )
            )
        )
        occupancy = min(occupancy, config.occupancy_max)

    schedule_deviation = rng.normal(
        loc=config.schedule_deviation_mean,
        scale=config.schedule_deviation_std,
    )

    return {
        "occupancy": str(occupancy),
        "scheduleDeviation": f"{schedule_deviation:.3f}",
    }


def set_vehicle_param_elements(
    vehicle: ElementTree.Element,
    attributes: Dict[str, str],
) -> None:
    if "occupancy" in attributes:
        value = attributes["occupancy"]
        if isinstance(value, bool):
            value = "1" if value else "0"
        elif isinstance(value, (int, float)):
            value = str(int(round(float(value))))
        elif isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "false"}:
                value = "1" if normalized == "true" else "0"
        attributes["occupancy"] = str(value)

    for key in attributes:
        vehicle.attrib.pop(key, None)
    for element in list(vehicle):
        if element.tag != "param":
            continue
        param_key = element.attrib.get("key")
        if param_key in attributes:
            vehicle.remove(element)
    for key, value in attributes.items():
        ElementTree.SubElement(vehicle, "param", {"key": key, "value": value})


def write_pretty_xml(tree: ElementTree.ElementTree, path) -> None:
    root = tree.getroot()
    try:
        ElementTree.indent(root, space="  ") # type: ignore
    except AttributeError:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)


def validate_traffic_multiplier(traffic_multiplier: float) -> None:
    if not 0.0 <= traffic_multiplier <= 1.0:
        raise ValueError("traffic_multiplier must be between 0 and 1.")


def reduce_route_traffic(
    route_path,
    keep_probability: float,
    seed: int,
) -> None:
    if keep_probability >= 1.0:
        return

    rng = np.random.default_rng(seed)
    tree = ElementTree.parse(route_path)
    root = tree.getroot()

    if keep_probability <= 0.0:
        kept_vehicles: List[ElementTree.Element] = []
    else:
        kept_vehicles = [
            vehicle
            for vehicle in iter_vehicle_elements(root)
            if rng.random() < keep_probability
        ]

    kept_vtype_ids = {
        vehicle.attrib.get("type")
        for vehicle in kept_vehicles
        if vehicle.attrib.get("type")
    }

    non_vehicle_elements = [
        element
        for element in root
        if element.tag != "vehicle"
    ]
    pruned_non_vehicle_elements = [
        element
        for element in non_vehicle_elements
        if not (
            element.tag == "vType"
            and element.attrib.get("id") not in kept_vtype_ids
        )
    ]

    kept_vehicles.sort(key=lambda e: float(e.attrib["depart"]))

    root.clear()
    for element in pruned_non_vehicle_elements:
        root.append(element)
    for vehicle in kept_vehicles:
        root.append(vehicle)

    write_pretty_xml(tree, route_path) # type: ignore
