# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from morl_tsp import config
from morl_tsp.environment.bus import Bus
from morl_tsp.environment.intersection_mapper import IntersectionMapper
from morl_tsp.environment.phase_controllers import BasePhaseController
from morl_tsp.environment.reward import Rewards
from morl_tsp.environment.ts_metrics import TrafficSignalMetrics

if TYPE_CHECKING:
    from env import SumoEnvironment

    from morl_tsp.environment.observations import ObservationFunctionPT

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")


#original documentation for env https://lucasalegre.github.io/sumo-rl/documentation/sumo_env/
class TrafficSignal(TrafficSignalMetrics):
    """This class represents a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    # Observation Space
    The default observation for each traffic signal agent is a vector:

    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]

    - ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
    - ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
    - ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
    - ```lane_i_queue``` is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

    You can change the observation space by implementing a custom observation class. See :py:class:`morl_tsp.environment.observations.ObservationFunction`.

    # Action Space
    Action space is discrete, corresponding to which green phase is going to be open for the next delta_time seconds.

    # Reward Function
    The default reward function is 'diff_waiting_time'. You can change the reward function by implementing a custom reward function and passing to the constructor of :py:class:`morl_tsp.environment.env.SumoEnvironment`.
    """
    
    def __init__(
        self,
        env : SumoEnvironment,
        phase_controller_class: type[BasePhaseController],
        tls_id:         str,
        delta_time:     int,
        yellow_time:    int,
        min_green:      int,
        max_green:      int,
        reward_weights: dict[str, float],
        reward_kwargs:  dict[str, dict[str,Any]],
        reward_clip:    tuple[float,float],
        reward_scales: dict[str, tuple[float,float]],
        reward_norm_with_previous: bool,
        obs_kwargs: dict[str, Any],
    ):
        """Initializes a TrafficSignal object.

        Args:
            env (SumoEnvironment): The environment this traffic signal belongs to.
            tls_id (str): The id of the traffic signal.
            delta_time (int): The time in seconds between actions.
            yellow_time (int): The time in seconds of the yellow phase.
            min_green (int): The minimum time in seconds of the green phase.
            max_green (int): The maximum time in seconds of the green phase.
            reward_weights: dict[str, float],
            reward_clip:    tuple[float,float],
            sumo (Sumo): The Sumo instance.
            reward_kwargs:  dict[str, dict[str,Any]],
        """
        super().__init__()

        self.id             = tls_id
        self.env            = env
        self.sumo           = env.sumo #traci, documentation on the sumo website
        self.last_reward    = None
        self.reward_weights = reward_weights
        self.reward_kwargs  = reward_kwargs
        self.reward_clip    = reward_clip
        self.reward_scales  = reward_scales
        self.reward_norm_with_previous = reward_norm_with_previous
        self.obs_kwargs     = obs_kwargs

        self.state_range:  int = config.STATE_RANGE
        
        self.vids_in_state:dict[str,tuple[float,float,str]] = {}

        '''Phase controller related variables'''
        self.phase_controller: BasePhaseController = phase_controller_class(self, delta_time, yellow_time, min_green, max_green) 
        self._green_phase_last_seen: dict[int, float] = {}

        '''Lanes related variables'''
        self.intersection_mapper: IntersectionMapper = IntersectionMapper(self.id, 
                                                      self.sumo)
        self.lanes = [lane for lane in self.intersection_mapper.fixed_order_lanes if "placeholder" not in lane]
        
        self.speed_limit_per_lane:      dict[str, float] = {lane:self.env.sumo.lane.getMaxSpeed(lane) if "empty_placeholder" not in lane else 0 for lane in self.intersection_mapper.fixed_order_lanes}
        self.free_flow_time_per_lane:   dict[str, float] = {lane:self.state_range/speed_limit      if "empty_placeholder" not in lane else 0 for lane,speed_limit in self.speed_limit_per_lane.items()}

        '''Observation related variables'''
        self.observation_fn: ObservationFunctionPT = self.env.observation(self, **self.obs_kwargs) # type: ignore

        '''Initialize the observation and action spaces - for the traffic signal agent''' 
        self.observation_space: spaces.Box = self.observation_fn.observation_space()
        self.action_space: spaces.Discrete[np.int64] = spaces.Discrete(self.phase_controller.num_green_phases)

        '''Initialize the reward related variables'''
        self.rewards = Rewards(self, self.reward_weights, self.reward_kwargs, self.reward_clip, self.reward_norm_with_previous,self.reward_scales)

        '''Initialize IntersectionZoo metrics'''
        self.co2_emission_history: list[float] = []
    
    def __getattr__(self, name: str):
        """
        Called only if `name` isn't found on self via the normal
        lookup (i.e. not an attribute on Wrapper itself). We
        delegate it to the wrapped object.
        """
        try:
            return getattr(self.phase_controller, name)
        except AttributeError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' - Original error: {e}") from e

    def __dir__(self) -> list[str]:
        """
        Make dir(self) list both Wrapper’s own attributes and
        the wrapped object’s attributes, for better IDE/autocomplete
        support.
        """
        return sorted({
            *super().__dir__(),
            *dir(self.phase_controller),
        })

    def update_scenario(self,
                        tls_id: str,) -> None:
        '''
        This method is called when the net file or traffic light changes but it is still controlled by the same agent.
        '''
        self.id = tls_id

        '''Lanes related variables'''
        self.intersection_mapper = IntersectionMapper(self.id,self.sumo)
        self.lanes = [lane for lane in self.intersection_mapper.fixed_order_lanes if "placeholder" not in lane]

    '''
    Phase controller related properties and methods
    '''
    @property
    def time_to_observe(self) -> bool:
        """Returns True if the traffic signal should act in the current step."""
        return self.phase_controller.time_to_observe # type: ignore
    
    @property
    def green_phase_last_seen(self) -> dict[int, float]:
        """Returns a dictionary mapping green phase indices to the last simulation step they were active."""
        return {k:self.env.sim_step - v for k,v in self._green_phase_last_seen.items()}
    
    def set_next_phase(self, new_phase: int) -> None:
        """Sets the next phase of the traffic signal.

        This method is used to set the next phase of the traffic signal and update the phase queue.
        It is called by the phase controller when it is time to change the phase.

        Args:
            new_phase (int): The index of the new phase to set.
        """
        self.phase_controller.set_next_phase(new_phase)

    def update(self) -> None:
        """Updates the traffic signal state.

        Is called on every step of the simulation.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.phase_controller.update()
        self.update_internals()

    def update_internals(self) ->None:
        if self.env.sim_step % 1 == 0:
            self.co2_emission_history.append(self.get_total_vehicle_emissions())
        self._green_phase_last_seen[self.current_phase] = self.env.sim_step

    '''
    Functions related to keeping an up to date list of vehicles that are within the reward range of the traffic light and are heading for the traffic light and their time of entrance
    '''
    def update_vehicle_dicts(self)->None:
        self.vehicle_tl_dict    = self._get_vehicle_dict()
        self.vids_in_state      = self._get_vids_in_state()
        self.buses_in_state:dict[str, tuple[float,float,str,Bus]] = {vid: (distance,speed,in_lane,self.env.buses[vid]) for vid, (distance,speed,in_lane) in self.vids_in_state.items() if self.env.bus_vid_prefix.lower() in vid.lower() and vid in self.env.buses}
        self.cars_in_state: dict[str, tuple[float,float,str]]     = {vid: (distance,speed,in_lane) for vid, (distance,speed,in_lane) in self.vids_in_state.items() if self.env.bus_vid_prefix.lower() not in vid.lower()}
        self.lane_to_vids:  dict[str,list[str]] = self._get_lane_to_vids()
        self.lane_to_vids_car:dict[str,list[str]] = {lane: [vid for vid in vids if self.env.bus_vid_prefix.lower() not in vid.lower()] for lane, vids in self.lane_to_vids.items()}
        self.lane_to_vids_bus:dict[str,list[str]] = {lane: [vid for vid in vids if self.env.bus_vid_prefix.lower() in vid.lower()] for lane, vids in self.lane_to_vids.items()}
        self._update_enter_time()
        self.crossing_times:    dict[str, float] = self._get_crossing_time()
        self.crossing_times_car:dict[str, float] = {vid:v for vid,v in self.crossing_times.items() if self.env.bus_vid_prefix.lower() not in vid.lower()}
        self.crossing_times_bus:dict[str, float] = {vid:v for vid,v in self.crossing_times.items() if self.env.bus_vid_prefix.lower()     in vid.lower()}
        #get delays and ppd
        self.car_delays:        list[float] = self.get_all_delays(vids=list(self.cars_in_state.keys()))
        self.bus_delays:        list[float] = self.get_all_delays(vids=list(self.buses_in_state.keys()))
        self.vehicle_delays:    list[float] = self.car_delays + self.bus_delays
        self.car_ppd:           list[float] = self.get_all_delays(vids=list(self.cars_in_state.keys()), per_person=True)
        self.bus_ppd:           list[float] = self.get_all_delays(vids=list(self.buses_in_state.keys()), per_person=True)
        self.vehicle_ppd:       list[float] = self.car_ppd + self.bus_ppd

    def _get_vehicle_dict(self)->dict[str, tuple[str, int, str]]:
        '''
        get a dictionary mapping vehicle id to the closest traffic light id and their distance to that traffic light
        for all vids in simulation
        '''
        return {vid: (tlsID, distance, self.intersection_mapper.get_in_lane_for_vehicle(vid))
                    for vid in self.env.vehicleIDList
                    for tlsID, _, distance, _  in self.sumo.vehicle.getNextTLS(vid) if tlsID == self.id}

    def _get_vids_in_state(self)->dict[str, tuple[float,float,str]]:
        """
        update traffic_signal.vids_in_state that map vehicle ids of vehicles within the reward range of the traffic light to their distance from the traffic light
        """
        return {vid: (distance,self.sumo.vehicle.getSpeed(vid),in_lane) for vid, (tlsID, distance, in_lane) in self.vehicle_tl_dict.items() if tlsID == self.id and distance < self.state_range}
    
    def _update_enter_time(self)->None:
        '''
        update traffic signal.vid_enter_time mapping vehicles ids of vehicles within the reward range to when they enter the reward range
        (reward range = some predefined distance to the traffic lights)
        '''
        current_vids = set(self.vids_in_state.keys())

        new_vids = current_vids - self.last_vids
        old_vids = self.last_vids - current_vids

        for vid in old_vids:
            self.vid_enter_exit_time[vid] = [self.vid_enter_time[vid],self.env.sim_step]
            del self.vid_enter_time[vid]
            
        for vid in new_vids:
            self.vid_enter_time[vid] = self.env.sim_step
            # Track entry lane for per-vehicle delay computation
            _, _, lane = self.vids_in_state[vid]
            self.vid_enter_lane[vid] = lane
        
        self.last_vids = current_vids
        return None

    def _get_lane_to_vids(self) -> dict[str, list[str]]:
        '''Returns a dictionary mapping lane ids to the list of vehicle ids in that lane - they are in the correct order'''
        return {lane: [vid for vid, (_, _, lane_id) in self.vids_in_state.items() if lane_id == lane] 
                for lane in self.intersection_mapper.fixed_order_lanes} # if "placeholder" not in lane

    '''
    RL related methods
    '''
    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self)->NDArray[np.float64]:
        """Computes the reward of the traffic signal."""
        self.last_reward = self.rewards.compute_reward()
        return self.last_reward

    '''
    Properties
    '''
    @property
    def co2_emission(self) -> float:
        '''Returns the total CO2 emission of vehicles within the reward range of the traffic light'''
        return sum(self.co2_emission_history)
