# SPDX-FileCopyrightText: Copyright (c) 2024 Vindula Jayawardana, Baptiste Freydt, Ao Qu, Cameron Hickert, Zhongxia Yan, Cathy Wu
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Scenario generation wrapper for vehicle mix adjustments."""

from __future__ import annotations

from typing import Dict
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


class ScenarioGenerationWrapper:
    """
    Wrapper that patches scenario generation to adjust vehicle mix and add
    per-vehicle attributes in route files.

    vehicle mix:
    |    |   group | probability |
    |---:|--------:|------------:|
    |  0 |      21 |        42%  |
    |  1 |      31 |        47%  |
    |  2 |      32 |        10%  |
    |  3 |      42 |       0.3%  |
    bus_multiplier is used to scale the probability of group '42' (buses).
    For example, bus_multiplier=2.0 will double the probability of buses in the generated scenario to 0.6%.
    """

    def __init__(self, generator_module) -> None:
        self._generator = generator_module

    def generate(
        self,
        config,
        prefix: str,
        seed: int,
        task_context,
        bus_multiplier: float,
        traffic_multiplier: float = 1.0,
        attribute_config: VehicleAttributeConfig | None = None,
    ) -> None:
        """
        Generate scenario files while scaling bus vs non-bus vehicle mix and
        injecting occupancy/schedule deviation attributes into route vehicles.
        """
        attr_config = attribute_config or VehicleAttributeConfig()
        validate_traffic_multiplier(traffic_multiplier)
        original_get_vehicle_mix = (
            self._generator.VehicleTypeParamsSampler.get_vehicle_mix
        )

        def scaled_vehicle_mix(instance) -> Dict[str, float]:
            vehicle_mix = original_get_vehicle_mix(instance)
            return self._scale_vehicle_mix(vehicle_mix, bus_multiplier)

        self._generator.VehicleTypeParamsSampler.get_vehicle_mix = scaled_vehicle_mix
        try:
            self._generator.generate_temp_sumo_files(
                config=config,
                prefix=prefix,
                seed=seed,
                task_context=task_context,
            )
        finally:
            self._generator.VehicleTypeParamsSampler.get_vehicle_mix = (
                original_get_vehicle_mix
            )

        route_path = config.working_dir / f"sumo/routes{prefix}.rou.xml"
        reduce_route_traffic(
            route_path,
            keep_probability=traffic_multiplier,
            seed=seed,
        )
        self._inject_vehicle_attributes(
            route_path,
            seed=seed,
            config=attr_config,
        )

    @staticmethod
    def _scale_vehicle_mix(
        vehicle_mix: Dict[str, float], bus_multiplier: float
    ) -> Dict[str, float]:
        scaled = {
            name: prob * (bus_multiplier if name.startswith("42") else 1.0)
            for name, prob in vehicle_mix.items()
        }
        total = sum(scaled.values())
        if total <= 0:
            raise ValueError("Bus multiplier results in a zero-probability mix.")
        return {name: prob / total for name, prob in scaled.items()}

    @staticmethod
    def _inject_vehicle_attributes(
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
            planned_depart = vehicle.attrib.get("depart")
            if planned_depart is not None:
                attributes["plannedDepart"] = planned_depart
            set_vehicle_param_elements(vehicle, attributes)

        write_pretty_xml(tree, route_path) # type: ignore
