# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from typing import Any

import numpy as np
from numpy.typing import NDArray

from morl_tsp import config
from morl_tsp.util.utils import cvar_tail_mean


class Rewards:

    def __init__(self, 
                 ts,
                 reward_weights: dict[str, float],
                 reward_kwargs: dict[str, dict[str,Any]],
                 reward_clip:   tuple[float,float],
                 reward_norm_with_previous: bool,
                 reward_scales: dict[str, tuple[float,float]],
                 ) -> None:

        self.ts = ts
        self.reward_weights = reward_weights
        self.reward_kwargs  = reward_kwargs
        self.reward_clip   = reward_clip
        self.min_clip, self.max_clip = reward_clip
        self.reward_norm_with_previous = reward_norm_with_previous
        self.reward_scales = reward_scales
        self.reward_history:list[tuple[float,...]] = []

        self.previous_reward = 0.0

        self.reward_fns = {
            "phase_change_penalty": self._phase_change_penalty,

            "average_speed":        self._average_speed_reward,
            "queue":                self._queue_reward,
            
            "delay":                self._delay,
            "delay_car":            self._delay_car,
            "delay_bus":            self._delay_bus,
            "mean_delay":           self._mean_delay,
            "mean_delay_car":       self._mean_delay_car,
            "mean_delay_bus":       self._mean_delay_bus,

            "mean_ppd":             self._mean_ppd,
            "mean_ppd_car":         self._mean_ppd_car,
            "mean_ppd_bus":         self._mean_ppd_bus,
            "sum_ppd":              self._sum_ppd,
            "sum_ppd_car":          self._sum_ppd_car,
            "sum_ppd_bus":          self._sum_ppd_bus,

            "safety":               self._safety_reward,
            "total_co2":            self._total_co2,

            "cvar_delay":           self._cvar_delay,
            "cvar_delay_car":       self._cvar_delay_car,
            "cvar_delay_bus":       self._cvar_delay_bus,
            "cvar_delay_nonbus":    self._cvar_delay_nonbus,
            "mean_squared_delays":  self._mean_squared_delays,
        }

    def compute_reward(self) -> NDArray[np.float64]:
        """Computes the reward using the specified reward function.

        Args:
            reward_name (str): The name of the reward function to use.
            reward_kwargs (dict, optional): Additional keyword arguments for the reward function. Defaults to {}.

        Returns:
            float: The computed reward.
        """
        scaled_rewards:dict[str, float] = {
                name:(self.reward_fns[name](**self.reward_kwargs.get(name, {})) - self.reward_scales.get(name, (0.0, 1.0))[0])/(self.reward_scales.get(name, (0.0, 1.0))[1] - self.reward_scales.get(name, (0.0, 1.0))[0] + 1e-8)
                for (name,_) in self.reward_weights.items()
        }
        weighted_rewards:list[float] = [
                weight * scaled_rewards[name]
                for name, weight in self.reward_weights.items()
        ] 
        self.reward_history.append(tuple(weighted_rewards))
        rewards:NDArray[np.float64] = np.array(weighted_rewards, dtype=np.float64)

        np.clip(rewards, self.min_clip, self.max_clip, out=rewards)

        return rewards

    ''' REAWRDS '''
    def _phase_change_penalty(self,
                              penalty:float=1)->float:
        '''
        Penalizes phase changes to encourage stability in traffic signal operation.
        '''
        if len(self.ts.env.trajectory) == 0:
            return 0.0
        return - ((self.ts.env.trajectory[-1][2] != self.ts.current_phase) * penalty)

    def _average_speed_reward(self):
        return self.ts.get_average_speed()

    def _queue_reward(self):
        return -sum(self.ts.get_route_queue_lengths_list())

    '''
    Delay based rewards
    '''
    def _delay(
        self,
        nonbus_cvar_alpha: float | None = None,
        nonbus_cvar_threshold: float | None = None,
        nonbus_cvar_penalty: float = 0.0,
        infeasible_action_penalty: float = 0.0,
    )->float:
        base = -sum(self.ts.vehicle_delays, 0.0)
        if (
            nonbus_cvar_alpha is not None
            and nonbus_cvar_threshold is not None
            and nonbus_cvar_penalty > 0.0
        ):
            base -= self._nonbus_tail_guardrail_penalty(
                alpha=float(nonbus_cvar_alpha),
                threshold=float(nonbus_cvar_threshold),
                penalty=float(nonbus_cvar_penalty),
            )
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)

    def _delay_car(
        self,
        nonbus_cvar_alpha: float | None = None,
        nonbus_cvar_threshold: float | None = None,
        nonbus_cvar_penalty: float = 0.0,
        infeasible_action_penalty: float = 0.0,
    ) -> float:
        base = -sum(self.ts.car_delays, 0.0)
        if (
            nonbus_cvar_alpha is not None
            and nonbus_cvar_threshold is not None
            and nonbus_cvar_penalty > 0.0
        ):
            base -= self._nonbus_tail_guardrail_penalty(
                alpha=float(nonbus_cvar_alpha),
                threshold=float(nonbus_cvar_threshold),
                penalty=float(nonbus_cvar_penalty),
            )
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)

    def _delay_bus(
        self,
        infeasible_action_penalty: float = 0.0,
    ) -> float:
        base = -sum(self.ts.bus_delays, 0.0)
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)

    def _mean_delay(
        self,
        nonbus_cvar_alpha: float | None = None,
        nonbus_cvar_threshold: float | None = None,
        nonbus_cvar_penalty: float = 0.0,
        infeasible_action_penalty: float = 0.0,
    )->float:
        delays = self.ts.vehicle_delays
        if not delays:
            return 0.0
        base = -sum(delays) / len(delays)
        if (
            nonbus_cvar_alpha is not None
            and nonbus_cvar_threshold is not None
            and nonbus_cvar_penalty > 0.0
        ):
            base -= self._nonbus_tail_guardrail_penalty(
                alpha=float(nonbus_cvar_alpha),
                threshold=float(nonbus_cvar_threshold),
                penalty=float(nonbus_cvar_penalty),
            )
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)
    
    def _mean_delay_car(
        self,
        nonbus_cvar_alpha: float | None = None,
        nonbus_cvar_threshold: float | None = None,
        nonbus_cvar_penalty: float = 0.0,
        infeasible_action_penalty: float = 0.0,
    )->float:
        delays = self.ts.car_delays
        if not delays:
            return 0.0
        base = -sum(delays) / len(delays)
        if (
            nonbus_cvar_alpha is not None
            and nonbus_cvar_threshold is not None
            and nonbus_cvar_penalty > 0.0
        ):
            base -= self._nonbus_tail_guardrail_penalty(
                alpha=float(nonbus_cvar_alpha),
                threshold=float(nonbus_cvar_threshold),
                penalty=float(nonbus_cvar_penalty),
            )
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)
    
    def _mean_delay_bus(self)->float:
        delays = self.ts.bus_delays
        if not delays:
            return 0.0
        return -sum(delays) / len(delays)

    ''' Per-Person Delay Rewards '''
    def _mean_ppd(self)->float:
        ppd = self.ts.vehicle_ppd
        if not ppd:
            return 0.0
        return -sum(ppd) / len(ppd)
    
    def _mean_ppd_car(self)->float:
        ppd = self.ts.car_ppd
        if not ppd:
            return 0.0
        return -sum(ppd) / len(ppd)
    
    def _mean_ppd_bus(self)->float:
        ppd = self.ts.bus_ppd
        if not ppd:
            return 0.0
        return -sum(ppd) / len(ppd)
    
    def _sum_ppd(self, infeasible_action_penalty: float = 0.0)->float:
        ppd = self.ts.vehicle_ppd
        if not ppd:
            return 0.0
        base = -sum(ppd)
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)

    def _sum_ppd_car(self, infeasible_action_penalty: float = 0.0)->float:
        ppd = self.ts.car_ppd
        if not ppd:
            return 0.0
        base = -sum(ppd)
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)
    
    def _sum_ppd_bus(self, infeasible_action_penalty: float = 0.0)->float:
        ppd = self.ts.bus_ppd
        if not ppd:
            return 0.0
        base = -sum(ppd)
        return self._apply_infeasible_action_penalty(base, infeasible_action_penalty)

    '''SAFETY REWARDS'''
    def _safety_reward(self,
                       safety_break_threshold=config.SAFETY_BREAK_THRESHOLD)->float:
        assert safety_break_threshold < 0.0, f"safety_break_threshold must be negative you passed: {safety_break_threshold}"
        return sum([a/safety_break_threshold for a in self.ts.get_all_vehicle_accelerations() if a < safety_break_threshold])

    '''FAIRNESS REWARDS'''
    def _cvar_delay(self, alpha: float = 0.10) -> float:
        """
        CVaR / tail mean of the worst alpha fraction of delays.
        alpha=0.10 means average of worst 10% delays.
        """
        delays = self.ts.vehicle_delays
        return -cvar_tail_mean(delays, alpha)
    
    def _cvar_delay_car(self, alpha: float = 0.10) -> float:
        delays = self.ts.car_delays
        return -cvar_tail_mean(delays, alpha)
    
    def _cvar_delay_bus(self, alpha: float = 0.10) -> float:
        delays = self.ts.bus_delays
        return -cvar_tail_mean(delays, alpha)

    def _cvar_delay_nonbus(self, alpha: float = 0.10) -> float:
        delays = self.ts.car_delays
        return -cvar_tail_mean(delays, alpha)

    def _nonbus_tail_guardrail_penalty(
        self,
        alpha: float,
        threshold: float,
        penalty: float,
    ) -> float:
        nonbus_tail_delay = cvar_tail_mean(self.ts.car_delays, alpha=alpha)
        excess_delay = max(0.0, float(nonbus_tail_delay) - float(threshold))
        return float(penalty) * excess_delay

    def _consume_infeasible_action_penalty_flag(self) -> bool:
        controller = getattr(self.ts, "phase_controller", None)
        if controller is None:
            return False
        consume_fn = getattr(controller, "consume_pending_infeasible_penalty", None)
        if callable(consume_fn):
            try:
                return bool(consume_fn())
            except Exception:
                return False
        return bool(getattr(controller, "last_action_infeasible", False))

    def _apply_infeasible_action_penalty(self, base: float, penalty: float) -> float:
        if float(penalty) <= 0.0:
            return float(base)
        if self._consume_infeasible_action_penalty_flag():
            return float(base) - float(penalty)
        return float(base)
    
    def _mean_squared_delays(self,
                            scale = 10_000.0) -> float:
        """
        Second-moment waiting time (convex penalty):
        -E[w^2] (scaled) to punish long waits disproportionately.

        Use this as a continuous fairness pressure; combine with an efficiency term (queue/delay/speed).
        """
        delays = self.ts.vehicle_delays
        if not delays:
            return 0.0

        mean_sq = sum((w * w) for w in delays) / len(delays)
        return -(mean_sq / scale)

    '''
    Intersection Zoo Rewards
    '''
    def _total_co2(self)->float:
        return -sum(self.ts.co2_emission_history[-self.ts.env.delta_time:])
