# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register

register(
    id="morl-tsp-v0",
    entry_point="morl_tsp.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
