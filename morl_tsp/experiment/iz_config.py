# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from pathlib import Path

from intersection_zoo.config import ELECTRIC, REGULAR
from intersection_zoo.env.config import IntersectionZooEnvConfig

from morl_tsp import config


class IntersectionZooExperimentConfig(IntersectionZooEnvConfig):
    working_dir: Path = Path(".")
    """ where to retrieve and store artifacts """
    scenarios: list[tuple[str, str]] | None = (
        None  # If None, all scenarios in the zoo will be used by slecting random ones
    )
    seed: int = config.INTERSECTION_ZOO_SEED
    # PathTaskContext(NamedTuple) from intersection_zoo.env.task_contexts import PathTaskContext
    single_approach: str | bool = False
    """ Which approach to use:
    a str like "A", "B"...,
    True to use them separately, or
    False to use all of them at the same time """
    penetration_rate: float = 0.0
    """ The penetration rate, between 0 and 1 (both included) """
    temperature_humidity: str | list[int] = "68_46"
    """ temperature and humidty conditions, in the format temperature_humidity """
    electric_or_regular: str | list[int] = REGULAR
    """ what type of setup to use in term of having electric vehicles vs internal combustion engine vehicles.
        REGULAR for internal combustion engine vehicles, ELECTRIC for electric vehicles
    """
    electric = ELECTRIC
    regular = REGULAR
    # IntersectionZooEnvConfig from intersection_zoo.env.configs import IntersectionZooEnvConfig
    visualize_sumo: bool = False
    # FuelEmissionsModels from sumo_adapter.physical_models import FuelEmissionsModels
    moves_emissions_models: list[str] = ["68_46"]
    """ Which (if any) MOVES surrogate to use """
    moves_emissions_models_conditions: list[str] = [REGULAR]
    """ What is the condition for each MOVES surrogate """
    """ Can be either REGULAR or ELECTRIC. If ELECTRIC, the emission model is only used for non-electric 
    vehicles and electric vehicles have zero emissions. Note that only controlled vehicles can be electric. 
    The human driven vehicles are always internal combustion engine vehicles. """
    sim_step_duration: float = config.EXP_STEP_LENGTH
    """ Duration of SUMO steps """
