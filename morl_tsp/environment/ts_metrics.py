# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from __future__ import annotations

from abc import ABC
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from morl_tsp import config
from morl_tsp.environment.bus import Bus
from morl_tsp.environment.intersection_mapper import IntersectionMapper
from morl_tsp.util.utils import cvar_tail_mean

if TYPE_CHECKING:
    from env import SumoEnvironment

class TrafficSignalMetrics(ABC):
    '''
    This class is inherited by the TrafficSignal class to provide metrics for the traffic signal. The metrics are used to evaluate the performance and calculate the reward for the traffic signal.

    The typehints below have the purpose that no errors are marked by VS Code 
    '''
    env: SumoEnvironment
    sumo: Any
    id: str
    lanes: list[str]
    state_range: int
    free_flow_time_per_lane: dict[str, float]
    intersection_mapper: IntersectionMapper
    # Vehicle related variables
    vehicle_tl_dict:    dict[str, Any]
    vids_in_state:      dict[str, tuple[float, float, str]]
    buses_in_state:     dict[str, tuple[float, float, str, Bus]]
    cars_in_state:      dict[str, tuple[float, float, str]]
    #lane_to_vids
    lane_to_vids:       dict[str, list[str]]
    lane_to_vids_car:   dict[str, list[str]]
    lane_to_vids_bus:   dict[str, list[str]]
    #crossing_times
    crossing_times:     dict[str, float]
    crossing_times_car: dict[str, float]
    crossing_times_bus: dict[str, float]
    # delays
    car_delays:     list[float]
    bus_delays:     list[float]
    vehicle_delays: list[float]
    car_ppd:        list[float]
    bus_ppd:        list[float]
    vehicle_ppd:    list[float]

    def __init__(self) -> None:
        self.state_range:   int = config.STATE_RANGE
        self.max_q_in_state:float = self.state_range / (config.DEFAULT_VEHICLE_LENGTH + config.MIN_GAP) #max number of vehicles that can be in the state range
        '''Initialize the traffic signal vid related variables'''
        self.last_vids:set[str] = set() #old vehicle ids from last time step
        self.vid_enter_time: dict[str,float | int] = {} #mapping vehicle id to the time that they enter the relevant range of the traffic light system
        self.vid_enter_exit_time:  dict[str,list[float | int]] = {}#mapping vehicle id to the time that they cross the traffic light system
        self.vid_enter_lane: dict[str, str] = {}  # vehicle id → lane at entry (for per-vehicle delay)
        pass

    '''
    New lane free state functions
    '''
    def get_route_densities(self)->dict[str,float]:
        '''
        Returns a dict with a fixed order that maps the lanes to the normalized density of vehicles in the lane
        '''
        lane_counts = Counter([t[2] for t in self.vids_in_state.values()])
        return {
            lane_id: 0 if lane_id not in lane_counts else lane_counts[lane_id] / self.max_q_in_state
            for lane_id in self.intersection_mapper.fixed_order_lanes
        }

    def get_route_queue_lengths(self)->dict[str,float]:
        '''
        Returns a dict with a fixed order that maps the lanes to the normalized queue length of vehicles in the lane
        '''
        vids_in_state_stopped = {vid:(distance,speed,lane) for vid,(distance,speed,lane) in self.vids_in_state.items() if speed < 1}
        
        lane_distances:dict[str,list[float]] = defaultdict(list)
        for distance, _, lane in vids_in_state_stopped.values():
            lane_distances[lane].append(distance)
        lane_to_queue_distances: dict[str,list[float]] = {lane: [distances[0]] + [current for prev, current in zip(distances[:-1],distances[1:], strict=True) if current - prev <= 10] 
                                                        for lane,distances in lane_distances.items() 
                                                        if min(distances) <= 5}
        lane_to_queue_len_norm =  {lane:len(queque_distances)/self.max_q_in_state for lane,queque_distances in lane_to_queue_distances.items()}
        
        return {
            lane_id: 0 if lane_id not in lane_to_queue_len_norm else lane_to_queue_len_norm[lane_id]
            for lane_id in self.intersection_mapper.fixed_order_lanes
        }
    
    def get_time_since_enter_state_per_route(self,
                                             vids: list[str] | None = None,
                                             agg_func:None | Callable[[Sequence[float]], float] = None,
                                             per_person: bool = False
                                             )->dict[str,list[float]] | dict[str,float]:
        '''
        Returns a dict that maps the lanes to the time since enter of vehicles in the lane
        If agg_func is provided, it will be applied to the list of times since enter for each lane to return a single value per lane.

        agg_func: function to aggregate the list of times since enter per lane (e.g., sum, max, min, np.mean)

        e.g. if agg_func = None:
            'D2TL_2': [25.0],
            'C2TL_0': [],
            'C2TL_1': [36.0, 41.0, 63.0, 52.0, 48.0, 65.0, 66.0, 44.0, 30.0, 69.0, 33.0],
            'C2TL_2': [51.0, 68.0, 65.0, 72.0, 67.0, 75.0],
            'B2TL_0': [22.0, 81.0, 80.0, 162.0, 86.0, 59.0, 5.0],
            'B2TL_1': [90.0, 84.0],
        '''
        lane_to_vids = self.lane_to_vids if vids is None else {lane:[vid for vid in lane_vids if vid in vids] for lane,lane_vids in self.lane_to_vids.items()}
        time_since_enter:dict[str,list[float]] = {lane: [(self.env.sim_step - self.vid_enter_time[vid]) * (self._get_occupancy(vid) if per_person else 1.0) for vid in vids] for lane,vids in lane_to_vids.items()}
        
        if agg_func is None:
            return time_since_enter
        else:
            return {lane: float(agg_func(v)) if len(v) > 0 else 0.0 for lane, v in time_since_enter.items()}

    def get_delay_per_route(self,
                            vids: list[str] | None = None,
                            normalize: bool = False,
                            per_person: bool = False
                            )->dict[str,list[float]]:
        '''
        Returns a dict that maps the lanes to the norm delay of vehicles in the lane
        '''
        time_since_enter:dict[str,list[float]] = self.get_time_since_enter_state_per_route(vids=vids, per_person=per_person,agg_func=None) # type: ignore
        if normalize:
            return {lane:[max(0,(t - self.free_flow_time_per_lane[lane])/(self.free_flow_time_per_lane[lane]+1e-6)) for t in times_since_enter] for lane,times_since_enter in time_since_enter.items()}
        else:
            return {lane:[max(0,(t-self.free_flow_time_per_lane[lane])) for t in times_since_enter] for lane,times_since_enter in time_since_enter.items()}
    
    def get_all_delays(self,
                       vids: list[str] | None = None,
                       normalize: bool = False,
                       per_person: bool = False
                       )->list[float]:
        return [delay for delays_per_route in self.get_delay_per_route(vids=vids, normalize=normalize, per_person=per_person).values() for delay in delays_per_route]

    def cvar_episode_crossing_times(self, alpha: float = 0.10, by_vid: Literal["car", "bus"] | None = None) -> float:
        crossing_time:dict[str,float] = self.crossing_times if by_vid is None else self.crossing_times_bus if by_vid.lower() == "bus" else self.crossing_times_car
        return cvar_tail_mean(list(crossing_time.values()), alpha)
    
    def get_route_densities_list(self)->list[float]:
        return list(self.get_route_densities().values())
    
    def get_route_queue_lengths_list(self)->list[float]:
        return list(self.get_route_queue_lengths().values())

    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        speed_list = [speed for _,speed,_ in self.vids_in_state.values()]
        return sum(speed_list) / len(speed_list) if len(speed_list) > 0 else 1.0
    
    def get_total_queued_vehicles(self) -> int:
        """Returns the total number of queued vehicles in the intersection."""
        return sum(1 for _, speed, _ in self.vids_in_state.values() if speed < 1.0)
    
    def get_all_vehicle_accelerations(self) -> list[float]:
        return [self.sumo.vehicle.getAcceleration(vid) for vid in self.vids_in_state.keys()]

    '''
    Crossing time related functions
    '''
    def get_mean_accumulated_waiting_time_per_lane(self,
                                                   by_vid: Literal["car", "bus"] | None = None
                                                   ) -> dict[str,float]:
        """
        Returns the mean total accumulated waiting time per lane as a dict. Meaning that the total accumulated waiting time of all vehicles in the lane is divided by the number of vehicles in the lane. 
        """
        lane_to_vids:dict[str,list[str]] = self.lane_to_vids if by_vid is None else self.lane_to_vids_bus if by_vid.lower() == "bus" else self.lane_to_vids_car
        return {lane: sum([self.sumo.vehicle.getWaitingTime(vid) for vid in vids])/len(vids) if len(vids) > 0 else 0.0
                for lane, vids in lane_to_vids.items() if "placeholder" not in lane}

    def get_max_accumulated_waiting_time_per_lane(self,
                                                  by_vid: Literal["car", "bus"] | None = None
                                                  ) -> dict[str,float]:
        """
        Returns the mean total accumulated waiting time per lane as a dict. Meaning that the total accumulated waiting time of all vehicles in the lane is divided by the number of vehicles in the lane. 
        """
        lane_to_vids:dict[str,list[str]] = self.lane_to_vids if by_vid is None else self.lane_to_vids_bus if by_vid.lower() == "bus" else self.lane_to_vids_car
        return {lane: max([self.sumo.vehicle.getWaitingTime(vid) for vid in vids], default=0.0) if len(vids) > 0 else 0.0
                for lane, vids in lane_to_vids.items() if "placeholder" not in lane}

    def get_accumulated_waiting_time_per_lane(self) -> list[float]:
        """Returns the accumulated waiting time per lane.

        Returns:
            list[float]: list of accumulated waiting time of each intersection lane.
        """
        return [sum([self.sumo.vehicle.getWaitingTime(vid) for vid in vids])
                for vids in self.lane_to_vids.values()]

    '''
    Crossing time related functions
    '''
    def _get_crossing_time(self)->dict[str,int | float]:
        '''
        return a dictionary mapping vehicle ids to the time it took them to cross the traffic light from the reward range
        '''
        return {k:v[1]-v[0] for k,v in self.vid_enter_exit_time.items()} 
    
    def get_agg_crossing_time_info(self,
                                   by_vid: Literal["car", "bus"] | None = None
                                   )->dict[str,int|float|None]: 
        '''
        return a list of the crossing time of all vehicles that crossed the traffic light
        '''
        #if we only want the calculation for car or buses, filter the vehicle id
        crossing_times:list[float] = list(self.crossing_times.values() if by_vid is None else self.crossing_times_bus.values() if by_vid.lower() == "bus" else self.crossing_times_car.values())
        return {"max"   : max(crossing_times, default=None), 
                "mean"  : sum(crossing_times)/len(crossing_times) if crossing_times else None, 
                "median": np.median(crossing_times) if crossing_times else None} # type: ignore

    def get_agg_delay_info(self,
                           by_vid: Literal["car", "bus"] | None = None
                           ) -> dict[str, int | float | None]:
        '''Per-vehicle delay = crossing_time - free_flow_time for the entry lane.'''
        if by_vid is None:
            ct = self.crossing_times
        elif by_vid.lower() == "bus":
            ct = self.crossing_times_bus
        else:
            ct = self.crossing_times_car

        delays: list[float] = []
        for vid, crossing_time in ct.items():
            lane = self.vid_enter_lane.get(vid)
            ff = self.free_flow_time_per_lane.get(lane, 0.0) if lane else 0.0
            delays.append(max(0.0, crossing_time - ff))

        return {
            "max":    max(delays, default=None),
            "mean":   sum(delays) / len(delays) if delays else None,
            "median": float(np.median(delays)) if delays else None,
        }

    '''
    Delay calculation related functions
    '''
    def _get_occupancy(self, vid:str) -> float:
        '''
        get the occupancy of vehicles in the same order as the vehicle ids provided. increase the bus occupancy by the bus_baseline param if include_bus_baseline = True
        '''
        if not vid.lower().startswith(self.env.bus_vid_prefix.lower()):
            return self.env.default_occupancy
        try:
            if vid in self.env.buses:
                return self.env.buses[vid].occupancy
            return self.env.buses.get_or_create(vid).occupancy
        except Exception:
            return self.env.bus_default_occupancy

    '''
    Intersection Reward Functions
    '''
    def get_total_vehicle_emissions(self) -> float:
        if len(self.vids_in_state) == 0:
            return 0.0
        return sum(self.get_vehicle_emissions(vid) for vid in self.vids_in_state)

    def get_vehicle_emissions(self, vid: str) -> float:
        if hasattr(self.env, "iz_emission_model") and "_" in vid:
            try:
                return self.get_iz_vehicle_emissions(vid)
            except Exception:
                pass
        try:
            return float(self.sumo.vehicle.getCO2Emission(vid))
        except Exception:
            return 0.0

    def get_iz_vehicle_emissions(self, vid: str) -> float:
        # make sure to only run this method if intersection zoo is being used otherwise the vehicle_emissions_type will not work with the vid splitting
        human_or_rl:Literal["human","rl"] = vid.split("_")[0] # type: ignore
        return self.env.iz_emission_model.get_emissions_single_condition(
            vehicle_emissions_type= vid.split("_")[1],
            speed = self.sumo.vehicle.getSpeed(vid),
            accel = self.sumo.vehicle.getAcceleration(vid),
            road_grade = self.sumo.vehicle.getSlope(vid),
            condition = self.env.iz_config.temperature_humidity, # type: ignore
            emission_condition = self.env.iz_config.regular if human_or_rl == "human" else self.env.iz_config.electric,
            is_rl = False if human_or_rl == "human" else True,
            )
    
    def get_iz_vehicle_idle_emissions(self, vid: str) -> float:
        # make sure to only run this method if intersection zoo is being used otherwise the vehicle_emissions_type will not work with the vid splitting
        human_or_rl:Literal["human","rl"] = vid.split("_")[0] # type: ignore
        return self.env.iz_emission_model.get_emissions_single_condition(
            vehicle_emissions_type= vid.split("_")[1],
            speed = 0.0,
            accel = 0.0,
            road_grade = self.sumo.vehicle.getSlope(vid),
            condition = self.env.iz_config.temperature_humidity, # type: ignore
            emission_condition = self.env.iz_config.regular if human_or_rl == "human" else self.env.iz_config.electric,
            is_rl = False if human_or_rl == "human" else True,
            )
    
    def get_idle_emissions_per_route(self,
                                ) -> dict[str,float]:
        '''
        return the idle emissions of a lane by summing up the idle emissions of all vehicles in the lane
        '''
        return {lane: sum([self.get_iz_vehicle_idle_emissions(vid) for vid in vids]) for lane, vids in self.lane_to_vids.items()}

    '''
    Logging Bus info
    '''
    def mean_bus_dwell_time(self)->float:
        dwell_times = [bus.dwell_time for bus in self.env.buses.values() if bus.dwell_time is not None]
        if len(dwell_times) == 0:
            return 0
        return sum(dwell_times) / len(dwell_times)

    def bus_dwell_times(self)->list[float]: 
        if len(self.env.buses) == 0:
            return []
        return [bus.dwell_time for bus in self.env.buses.values() if bus.dwell_time is not None]

    def _get_average_speed_by_type(self,
                                   by_vid: Literal["car", "bus"] | None = None) -> float | None:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        vids:list[str] = list(self.vids_in_state if by_vid is None else self.buses_in_state if by_vid.lower() == "bus" else self.cars_in_state)

        if len(vids) == 0:
            return None

        return sum([self.vids_in_state[vid][1] for vid in vids]) / len(vids)
    def get_bus_speed(self)->float | None:
        return self._get_average_speed_by_type(by_vid="bus")
    def get_car_speed(self)->float | None:
        return self._get_average_speed_by_type(by_vid="car")
