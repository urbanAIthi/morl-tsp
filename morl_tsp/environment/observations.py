# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""Observation functions for traffic signals."""
import heapq
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import spaces

from morl_tsp.config import MAX_NUM_BUS_PER_LANE_IN_OBS, MAX_NUM_LANES, MAX_NUM_PHASES
from morl_tsp.util.utils import clipped_min_max_normalization, cvar_tail_mean_per_group

from .bus import Bus, BusStore
from .traffic_signal import TrafficSignal
from .typing import obs_type


class ObservationFunction(ABC):
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts     = ts
        self.env    = ts.env
        self.sumo   = self.env.sumo
        '''Initialize additional state variables'''
        self._init_additional_obs()

    @abstractmethod
    def __call__(self) -> obs_type:
        """Subclasses must override this method."""
        pass

    def observation_space(self)->spaces.Box:
        """Subclasses must override this method."""
        obs_len = self._observation_len()
        return spaces.Box(
            low =np.zeros(obs_len,    dtype=np.float32),
            high=np.ones(obs_len,     dtype=np.float32),
        )

    @abstractmethod
    def _observation_len(self) -> int:
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def _get_dict_obs(self)->dict[str,Any]:
        pass

    @abstractmethod
    def _init_additional_obs(self)->None:
        pass
        
class EmptyObservationFunction(ObservationFunction):
    """Empty observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize empty observation function."""
        super().__init__(ts)

    def __call__(self) -> obs_type:
        """Return an empty observation."""
        return np.array([0], dtype=np.float32)

    def _observation_len(self) -> int:
        """Return the observation length."""
        return 1
class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> obs_type:
        """Return the default observation."""

        observation = np.array(self._get_obs, dtype=np.float32)
        if len(observation) == 0:
            observation = np.zeros(1, dtype=np.float32)
        return observation
    
    def _observation_len(self) -> int:
        """Return the observation length."""
        n_obs = 0
        if self.add_phase_id:
            n_obs += self.ts.num_phases
        if self.add_min_green:
            n_obs += 1
        if self.add_phase_last_seen:
            n_obs += MAX_NUM_PHASES
        if self.add_density:
            n_obs += MAX_NUM_LANES
        if self.add_queue:
            n_obs += MAX_NUM_LANES
        if self.add_co2:
            n_obs += MAX_NUM_LANES
        if self.add_mean_delay:
            n_obs += MAX_NUM_LANES
        if self.add_cvar_delay:
            n_obs += MAX_NUM_LANES    
        if self.add_reward_weights_to_obs:
            n_obs += len(self.ts.reward_weights)
        n_obs = n_obs + self._time_of_day_obs_len

        return n_obs

    def _get_dict_obs(self) -> dict[str, list[float] | tuple[float,...] | float | int]:
        return_dict:dict[str, list[float] | tuple[float,...] | float | int] = {}
        if self.add_phase_id:
            return_dict["phase_id"] = [1 if self.ts.current_phase == i else 0 for i in range(self.ts.num_phases)]
        if self.add_min_green:
            return_dict["min_green"] = self.ts.phase_controller.can_change_phase()
        if self.add_phase_last_seen:
            return_dict["phase_last_seen"] = self.phase_last_seen_feat
        if self.add_density:
            return_dict["density"] = (self.ts.get_route_densities_list())
        if self.add_queue:
            return_dict["queue"] = self.ts.get_route_queue_lengths_list()
        if self.add_co2:
            return_dict["co2"] = list(self.ts.get_idle_emissions_per_route().values())
        if self.add_mean_delay:
            return_dict["mean_delay"] = self.mean_delay_per_route
        if self.add_cvar_delay:
            return_dict["cvar_delay"] = self.cvar_per_route_delay
        if self.add_reward_weights_to_obs:
            return_dict["reward_weights"] = self.reward_weights_feat
        if self.time_of_day == "simple":
            return_dict["time_of_day"] = self._time_of_day_simple
        elif self.time_of_day == "circular":
            return_dict["time_of_day"] = self._time_of_day_circular
        return return_dict
    
    @property
    def _get_obs(self) -> list[float]:
        return [x for v in self._get_dict_obs().values() for x in ([v] if isinstance(v, float) or isinstance(v, int) else v) ]

    def _init_additional_obs(self)->None:
        self.add_phase_id = self.env.add_phase_id
        self.add_min_green = self.env.add_min_green
        self.add_density = self.env.add_density
        self.add_queue = self.env.add_queue
        self.add_co2 = self.env.add_co2
        self.add_phase_last_seen = self.env.add_phase_last_seen
        self.add_mean_delay = self.env.add_mean_delay
        self.add_cvar_delay = self.env.add_cvar_delay
        self.add_reward_weights_to_obs = self.env.add_reward_weights_to_obs
        self.time_of_day = self.env.time_of_day

    '''
    Additional state variables
    '''
    '''Time of day'''
    @property
    def _time_of_day_feat(self)->None|float|tuple[float,float]:
        if self.time_of_day is None:
            return None
        elif self.time_of_day == "simple":
            return self._time_of_day_simple
        elif self.time_of_day == "circular":
            return self._time_of_day_circular
    @property
    def _time_of_day_obs_len(self)->int: # type: ignore
        if self.time_of_day is None:
            return 0
        elif self.time_of_day == "simple":
            return 1
        elif self.time_of_day == "circular":
            return 2
    @property
    def _time_of_day_simple(self)->float:
        return self.sumo.simulation.getTime()
    @property
    def _time_of_day_circular(self)->tuple[float,float]:
        time_in_radians = (self.sumo.simulation.getTime() / 86400 ) * 2 * np.pi
        return ( np.sin(time_in_radians) + 1) / 2, ( np.cos(time_in_radians) + 1) / 2

    '''
    Phase last seen
    '''
    @property
    def phase_last_seen_feat(self)->list[float]:
        '''Returns the phase last seen feature as a list of floats'''
        return [self.ts.green_phase_last_seen.get(i, i)/300 for i in range(MAX_NUM_PHASES)]

    '''
    Equality obs
    '''
    @property
    def cvar_per_route_delay(self, alpha: float = 0.10, normalize: bool = True) -> list[float]:
        delays_per_route = self.ts.get_delay_per_route(normalize=normalize)
        return [delay/30 for delay in cvar_tail_mean_per_group(delays_per_route, alpha)]

    @property
    def mean_delay_per_route(self,normalize:bool=True) -> list[float]:
        '''
        Returns the mean delay per route
        '''
        delays:dict[str,list[float]] = self.ts.get_delay_per_route(normalize=normalize) #CHECK THE NORMALIZATION HERE
        return [(sum(lane_delays) / len(lane_delays))/30 if lane_delays else 0.0 for lane_delays in delays.values()]

    @property
    def reward_weights_feat(self) -> list[float]:
        """
        Return reward weights as a simplex-like conditioning vector in the observation.
        """
        weights = np.array(list(self.ts.reward_weights.values()), dtype=np.float32)
        denom = np.sum(np.abs(weights))
        if denom <= 1e-8:
            return [0.0 for _ in weights]
        return list(weights / denom)

class ObservationFunctionPT(DefaultObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, 
                 ts:                        TrafficSignal,
                 n_buses_per_lane:          int         = MAX_NUM_BUS_PER_LANE_IN_OBS,#this is the number of consecutive buses that gets passed to the agent
                 in_state_distance_factor:  float       = 1,    #buses that are x times the distance_normalize_range[1] or less away from the LSA get included in the state
                 delay_normalize_range:     list[float] | None = None, 
                 occupency_normalize_range: list[int]   | None = None,
                 ):
        """Initialize default observation function."""
        super().__init__(ts)
        delay_normalize_range = [-60,300] if delay_normalize_range is None else list(delay_normalize_range)
        occupency_normalize_range = [0,50] if occupency_normalize_range is None else list(occupency_normalize_range)
        self.buses:             BusStore    = self.env.buses
        self.n_buses_per_lane:  int         = n_buses_per_lane
        self.in_state_distance: float       = self.ts.state_range * in_state_distance_factor
        
        self.lanes:             list[str]       = list(self.ts.intersection_mapper.fixed_order_lanes)
        self.lane_set:          set[str]        = set(self.lanes)
        self.bus_obs_prefix:    dict[str,str]   = {lane: f"{lane}_bus" for lane in self.lanes}
        

        self.distance_normalize_range:      list[int | float]   = [0,self.in_state_distance]                             
        self.delay_normalize_range:         list[int | float]   = delay_normalize_range
        self.occupency_normalize_range:     list[int]           = occupency_normalize_range
        
        self.n_observations_per_bus:    int = self.n_observations_per_bus_total
        self.n_bus_observations:        int = self._total_bus_obs_len
        
        self.bus_observation: np.ndarray[Any,Any] = np.zeros(self.n_bus_observations) # this array includes the normalized observations of the buses

    def __call__(self) -> obs_type:
        return super().__call__()
    
    def _init_additional_obs(self)->None:
        super()._init_additional_obs()
        self.add_distance   = self.env.add_distance
        self.add_delay      = self.env.add_delay
        self.add_occupancy  = self.env.add_occupancy
        
    
    def _get_dict_obs(self) -> dict[str, list[float] | tuple[float,...] | float | int]:
        return_dict = super()._get_dict_obs()

        return_dict = return_dict | self._get_bus_dict_obs()
        
        return return_dict

    def _observation_len(self) -> int:
        """Return the observation length."""
        n_obs = super()._observation_len()
        n_obs = n_obs + self._total_bus_obs_len
        return n_obs
    
    def _get_bus_observation(self, bus: None | Bus) -> dict[str, float]:
        """Returns the observation of a bus given its ID."""
        if bus is not None:
            obs: dict[str, float] = {}
            if self.add_distance:  # normalized_distance (inverted as in original)
                obs["distance"] = abs(
                    clipped_min_max_normalization(
                        bus.distance,
                        self.distance_normalize_range[0],
                        self.distance_normalize_range[1],
                    ) - 1
                )

            if self.add_delay:  # normalized_delay
                obs["delay"] = clipped_min_max_normalization(
                    bus.delay,
                    self.delay_normalize_range[0],
                    self.delay_normalize_range[1],
                )

            if self.add_occupancy:  # normalized_occupancy
                obs["occupancy"] = clipped_min_max_normalization(
                    bus.occupancy,
                    self.occupency_normalize_range[0],
                    self.occupency_normalize_range[1],
                )

            return obs
        else:
            dummy_obs: dict[str, float] = {}

            if self.add_distance:
                dummy_obs["distance"] = 0.0

            if self.add_delay:
                dummy_obs["delay"] = 0.0

            if self.add_occupancy:
                dummy_obs["occupancy"] = 0.0

            return dummy_obs

    def get_bus_in_obs(
        self,
        empty_id: str | None = None,
        empty_dist: float | None = None,
        empty_bus: Bus | None = None,
    ) -> dict[str, tuple[str | None, float | None, Bus | None]]:
        heaps: dict[str, list[tuple[float, str, Bus]]] = {lane: [] for lane in self.lanes}

        n = self.n_buses_per_lane
        lane_set = self.lane_set

        # One pass: keep top-n closest per lane
        for veh_id, v in self.ts.buses_in_state.items():
            dist = v[0]
            lane = v[2]
            bus_obj = v[3]
            if lane not in lane_set:
                continue

            h = heaps[lane]
            nd = -dist  # store negative dist so h[0] is "farthest among kept"

            item = (nd, veh_id, bus_obj)

            if len(h) < n:
                heapq.heappush(h, item)
            else:
                # Replace farthest kept bus if this one is closer
                if nd > h[0][0]:
                    heapq.heapreplace(h, item)

        # Pre-fill output with empty keys for all lanes/slots
        out: dict[str, tuple[str | None, float | None, Any | None]] = {}
        for lane in self.lanes:
            prefix = self.bus_obs_prefix[lane]
            for i in range(n):
                out[prefix + str(i)] = (empty_id, empty_dist, empty_bus)

        # Overwrite with actual closest buses (sorted by ascending distance)
        for lane in self.lanes:
            h = heaps[lane]
            if not h:
                continue

            # Convert to (dist, veh_id, bus_obj) and sort by dist
            closest_sorted = sorted(((-nd, veh_id, bus_obj) for nd, veh_id, bus_obj in h),
                                    key=lambda t: t[0])

            prefix = self.bus_obs_prefix[lane]
            for i, (dist, veh_id, bus_obj) in enumerate(closest_sorted):
                out[prefix + str(i)] = (veh_id, dist, bus_obj)

        return out
    
    def _get_bus_dict_obs(self)-> dict[str, float]:
        bus_in_obs:     dict[str, tuple[None | str, None | float, Bus | None]] = self.get_bus_in_obs(empty_id=None, empty_dist=None)

        return {f"{lane_bus}_{obs_name}":obs for lane_bus,(_,_,bus_object) in bus_in_obs.items() for obs_name,obs in self._get_bus_observation(bus_object).items()}
    
    @property
    def n_observations_per_bus_total(self)->int:
        return int(self.add_distance) + int(self.add_delay) + int(self.add_occupancy)
    
    @property
    def _total_bus_obs_len(self)->int:
        return len(self.ts.intersection_mapper.fixed_order_lanes) * self.n_observations_per_bus_total * self.n_buses_per_lane
