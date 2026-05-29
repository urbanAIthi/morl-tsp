# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from typing import Any

import numpy as np
import numpy.typing as npt

type obs_type = npt.NDArray[np.float32]
type action_type = dict[Any, Any] | int | float | npt.NDArray[Any] | None
type reward_type = float | npt.NDArray[np.float64]

type trajectory_type = list[tuple[obs_type, reward_type, action_type, list[float]]]
