# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from abc import ABC, abstractmethod
from numbers import Real
from typing import Any

from morl_tsp.environment.bus import BusStore
from morl_tsp.environment.typing import action_type, trajectory_type


class EnvMetrics(ABC):
    '''
    Protocol for SumoEnvironment class
    '''
    traffic_signals: dict[str, Any]
    buses: BusStore
    tls_ids: list[str]
    vehicles: dict[str, dict[str, float]]
    episode: int
    sumo: Any
    vehicleIDList: tuple[str, ...]
    _route: str
    begin_time: int
    add_system_info: bool
    add_per_agent_info: bool
    add_per_bus_info: bool
    reward_fn: str | None
    tls_id: str
    trajectory: trajectory_type

    @property
    @abstractmethod
    def sim_step(self) -> float:
        pass
    
    def _compute_info(self) -> dict[str, Any]:
        info = {"step": self.sim_step}
        if self.add_system_info:
            info = info | self._get_system_info()
        if self.add_per_agent_info:
            info = info | self._get_per_agent_info()
        if self.add_per_bus_info:
            info = info | self._get_per_bus_info()
        return info

    def _get_system_info(self) -> dict[str, float]:
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in self.vehicleIDList]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in self.vehicleIDList]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(self.vehicleIDList) == 0 else  (sum(waiting_times)/len(waiting_times)),
            "system_mean_speed": 0.0 if len(self.vehicleIDList) == 0 else (sum(speeds)/len(speeds)),
        }

    def _get_per_agent_info(self) -> dict[str, float]:
        stopped = [self.traffic_signals[ts].get_total_queued_vehicles() for ts in self.tls_ids]
        accumulated_waiting_time = [
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.tls_ids
        ]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.tls_ids]
        info:dict[str,Any] = {}
        for i, ts in enumerate(self.tls_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def _get_per_bus_info(self) -> dict[str, int | float | str]:
        per_bus_info_dict:dict[str, int | float | str] = {}
        for bus_id, bus in self.buses.items():
            for k, v in bus.info_dict.items():
                per_bus_info_dict[f"{bus_id}_{k}"] = v
        return per_bus_info_dict

    def get_all_reward_metrics(self) -> dict[str, float]:
        agent = self.traffic_signals[self.tls_ids[0]]
        return {k:v() for k,v in agent.rewards.reward_fns.items()}

    def get_numerical_info_dict(self) -> dict[str, float | action_type]:
        assert len(self.tls_ids) == 1, "Function only accommodates a single agent right now"

        numerical_info_dict: dict[str, float | action_type] = {}
        numerical_info_dict["episode"] = self.episode
        numerical_info_dict["current_ts"] = self.sumo.simulation.getTime()
        numerical_info_dict["num_vehicles"] = len(self.vehicleIDList)
        numerical_info_dict["vehicles_in_state"] = len(self.traffic_signals[self.tls_ids[0]].vids_in_state)
        numerical_info_dict["cars_in_state"] = len(self.traffic_signals[self.tls_ids[0]].cars_in_state)
        numerical_info_dict["buses_in_state"] = len(self.traffic_signals[self.tls_ids[0]].buses_in_state)
        reward = self.traffic_signals[self.tls_ids[0]].last_reward
        if len(reward) > 1:
            numerical_info_dict["reward"] = sum(reward)
            for reward_component, value in zip(self.traffic_signals[self.tls_ids[0]].reward_weights.keys(), reward, strict=True):
                numerical_info_dict[f"reward_{reward_component}"] = value
        else:
            numerical_info_dict["reward"] = reward[0]        
        numerical_info_dict["stopped"] = sum(
            self.traffic_signals[ts].get_total_queued_vehicles() for ts in self.tls_ids
        )
        numerical_info_dict["accumulated_waiting_time"] = sum(
            sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.tls_ids
        )
        numerical_info_dict["average_speed"] = sum(
            self.traffic_signals[ts].get_average_speed() for ts in self.tls_ids
        )
        numerical_info_dict["action"] = self.trajectory[-1][2] if len(self.trajectory) > 0 else 0
        current_state = self.sumo.trafficlight.getRedYellowGreenState(self.tls_ids[0])
        phase_idx = next(
            (
                i
                for i, p in enumerate(self.traffic_signals[self.tls_ids[0]].phase_controller.all_phases)
                if p.state == current_state
            ),
            -1,
        )
        numerical_info_dict["phase"] = phase_idx
        numerical_info_dict["route_density"] = sum(
            self.traffic_signals[self.tls_ids[0]].get_route_densities().values()
        )

        numerical_info_dict["mean_acc_waiting_time"] = sum(
            self.traffic_signals[self.tls_ids[0]].get_mean_accumulated_waiting_time_per_lane(by_vid=None).values()
        )
        numerical_info_dict["mean_acc_waiting_time_car"] = sum(
            self.traffic_signals[self.tls_ids[0]].get_mean_accumulated_waiting_time_per_lane(by_vid="car").values()
        )
        numerical_info_dict["mean_acc_waiting_time_bus"] = sum(
            self.traffic_signals[self.tls_ids[0]].get_mean_accumulated_waiting_time_per_lane(by_vid="bus").values()
        )

        numerical_info_dict["max_acc_delay"] = max(
            self.traffic_signals[self.tls_ids[0]].get_max_accumulated_waiting_time_per_lane(by_vid=None).values()
        )
        numerical_info_dict["max_acc_delay_car"] = max(
            self.traffic_signals[self.tls_ids[0]].get_max_accumulated_waiting_time_per_lane(by_vid="car").values()
        )
        numerical_info_dict["max_acc_delay_bus"] = max(
            self.traffic_signals[self.tls_ids[0]].get_max_accumulated_waiting_time_per_lane(by_vid="bus").values()
        )

        return numerical_info_dict

    def get_numerical_info_dict_bus(self) -> dict[str, float]:
        assert len(self.tls_ids) == 1, "Function only accommodates a single agent right now"

        numerical_info_dict: dict[str, float] = {}
        numerical_info_dict["num_buses"] = len(self.buses)
        numerical_info_dict["mean_bus_dwell_time"] = sum(
            self.traffic_signals[ts].mean_bus_dwell_time() for ts in self.tls_ids
        )
        numerical_info_dict["max_bus_dwell_time"] = max(
            self.traffic_signals[ts].mean_bus_dwell_time() for ts in self.tls_ids
        )
        # Only sum speeds where get_bus_speed() is not None
        numerical_info_dict["mean_bus_speed"] = sum(
            self.traffic_signals[ts].get_bus_speed() 
            for ts in self.tls_ids 
            if self.traffic_signals[ts].get_bus_speed() is not None
        )
        return numerical_info_dict

    def get_episode_info(self) -> dict[str, float | int | list[float]]:
        """
        Returns an info dict containing route metadata and aggregated crossing times.
        """
        episode_info: dict[str, float | int | list[float]] = {}
        if self._route is not None: # type: ignore
            route_bits = self._route.split("/")[-1].split(".")[0].split("_") # type: ignore
            if len(route_bits) > 1:
                episode_info["route_file"] = int(route_bits[1])
        episode_info["begin_time"] = self.begin_time

        episode_info["co2"] = self.traffic_signals[self.tls_ids[0]].co2_emission

        episode_info["cvar_crossing_times_10%"] = self.traffic_signals[self.tls_ids[0]].cvar_episode_crossing_times(alpha=0.1)
        episode_info["cvar_crossing_times_1%"] = self.traffic_signals[self.tls_ids[0]].cvar_episode_crossing_times(alpha=0.01)

        episode_info["cvar_CT_bus_10%"] = self.traffic_signals[self.tls_ids[0]].cvar_episode_crossing_times(alpha=0.1,by_vid="bus")

        # Aggregate crossing times by Car/Bus and always keep numeric outputs.
        # Some episodes can return None for empty sets; convert those to NaN so
        # eval summary extraction still carries the metric keys.
        for k, v in self.traffic_signals[self.tls_ids[0]].get_agg_crossing_time_info(by_vid="car").items():
            episode_info[f"Car_crossing_time_{k}"] = float(v) if isinstance(v, Real) else float("nan")

        for k, v in self.traffic_signals[self.tls_ids[0]].get_agg_crossing_time_info(by_vid="bus").items():
            episode_info[f"Bus_crossing_time_{k}"] = float(v) if isinstance(v, Real) else float("nan")

        # Per-vehicle delay = crossing_time - lane free_flow_time
        for k, v in self.traffic_signals[self.tls_ids[0]].get_agg_delay_info(by_vid="car").items():
            episode_info[f"Car_delay_{k}"] = float(v) if isinstance(v, Real) else float("nan")

        for k, v in self.traffic_signals[self.tls_ids[0]].get_agg_delay_info(by_vid="bus").items():
            episode_info[f"Bus_delay_{k}"] = float(v) if isinstance(v, Real) else float("nan")

        episode_info["Crossing_time_distrubution"] = list(self.traffic_signals[self.tls_id].crossing_times.values())
        episode_info["Crossing_time_distrubution_car"] = [v for k,v in self.traffic_signals[self.tls_id].crossing_times.items() if "bus" not in k.lower()]
        episode_info["Crossing_time_distrubution_bus"] = [v for k,v in self.traffic_signals[self.tls_id].crossing_times.items() if "bus" in k.lower()]

        # Controller-level stability metrics (true signal stats, not action proxies).
        try:
            stability_metrics = self.traffic_signals[self.tls_ids[0]].get_episode_stability_metrics()
            for key, value in stability_metrics.items():
                if isinstance(value, (int, float)):
                    episode_info[key] = float(value)
        except Exception:
            pass

        return episode_info
    
    def get_step_info(self) -> dict[str, float | int | action_type]:
        """
        Returns an info dict containing step-level aggregated metrics.
        """
        step_info: dict[str, float | int | action_type] = {}
        step_info["co2"] = self.traffic_signals[self.tls_ids[0]].get_total_vehicle_emissions()

        step_info = step_info | self.get_numerical_info_dict()

        step_info = step_info | self.get_mean_accumulated_waiting_time_per_lane()

        return step_info

    '''
    Forward methods from the traffic singla so that sb3 metrics can access them directly
    '''
    def get_mean_accumulated_waiting_time_per_lane(self):
        return self.traffic_signals[self.tls_ids[0]].get_mean_accumulated_waiting_time_per_lane()
