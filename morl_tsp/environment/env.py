# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

"""SUMO Environment for Traffic Signal Control."""
import atexit
import logging
import os
import pickle
import random
import sys
import warnings
from pathlib import Path
from signal import SIGINT, signal
from types import FrameType
from typing import Any, Literal, NoReturn

import gymnasium as gym
import numpy as np
import sumolib
from dotenv import load_dotenv
from numpy.typing import NDArray

from morl_tsp import config
from morl_tsp.environment import observations, phase_controllers
from morl_tsp.util.backend import LIBSUMO, TRACI_SUPPORTS_CONNECTION_LABELS, traci
from morl_tsp.util.utils import cleanup_decorator, nseconds_from_str

from .bus import Bus, BusDataManager, BusStore
from .env_metrics import EnvMetrics
from .observations import ObservationFunction, ObservationFunctionPT
from .phase_controllers import (
    BasePhaseController,
    BasicPhaseController,
)
from .traffic_signal import TrafficSignal
from .typing import obs_type, reward_type, trajectory_type, action_type

if "SUMO_HOME" not in os.environ:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
if tools not in sys.path:
    sys.path.append(tools)

load_dotenv()

logging.basicConfig(filename='env.log', 
                    filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s', 
                    level=logging.DEBUG)

class SumoEnvironment(gym.Env[obs_type, action_type], EnvMetrics):
    """SUMO Environment for Traffic Signal Control.

    Class that implements a gym.Env interface for traffic signal control using the SUMO simulator.
    See https://sumo.dlr.de/docs/ for details on SUMO.
    See https://gymnasium.farama.org/ for details on gymnasium.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (tuple[int, int] | None): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The time in seconds the simulation must end. Default: 3600
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Maximum green time in a phase (enforced by BasicPhaseController). Default: 60 seconds.
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict): String with the name of the reward function used by the agents, a reward function, or dictionary with reward functions assigned to individual traffic lights by their keys.
        observation (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
        ##### New Imports #####
        additional_files (str): these files include additional elements like bus stops
        sumo_cfg (str): The sumo cfg is an alternative method to start the simulation - some parameters need to have specific values that match the xml file, for example delta_time = <step-length value="4"/> and begin_time matching the configured SUMO begin time in seconds.
        tls_id: str | None = None, #allows you to chose a different tls_id
        actions: list[str] | Literal["parameters"] | None= None, #None means that the traffic signal builds the phases/actions itself, parameters means that the list in the parameters.json will be used
             ["GggGGgrrrrrrGggGGgrrrrrrrrrrrrr",
              "rrrrrGrrrrrrrrrrrGrrrrrrrrrrrrr",
              "rrrrrrGggGGgrrrrrrGggGGgrrrrrrr",
              "rrrrrrrrrrrGrrrrrrrrrrrGrrrrrrr",
              "GggGGrrrrrrrGggGGrrrrrrrrrrrrrr",
              "rrrrrrGggGGrrrrrrrGggGGrrrrrrrr"]
        composed_phase_dict: dict[int,int] | None = None, #if the actions are composed of multiple phases this dict will specify which phase (value) follows which phase (key), so when phase (key) is active phase (value) will be actived afterwards
            {0:1,2:3}
        minor_phase_list: list[int] | None = None, #each value of the list is the index of a minor phase from the list of actions
            [1,3]
        minor_phase_min_length: int | None = 5, #the length of the minor phases, None means it is the same length as the major phases given by min_green

    """ 
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self,
        net_file:           None | str = None,
        route_file:         None | str | list[str] = None,
        flow_file:          None | str = None,
        scenarios:          None | list[tuple[str,str]] = None, # list of (net_file, route_file) tuples - this is mostly used for
        vehicles_file:      None | str = None, # file contains vehicle type definitions and is added to the route files in the start of the simulation
        begin_time:         int | str = "14:30:00",
        num_seconds:        int = 20000,
        max_depart_delay:   int = 100,
        waiting_time_memory:int = 1000,
        time_to_teleport:   int = 300,
        delta_time:         int = 1,
        single_agent:       bool = True,
        add_system_info:    bool = False,
        add_per_agent_info: bool = True,
        sumo_seed:          Literal["random"] | int = 0,
        sumo_warnings:      bool = False,
        additional_sumo_cmd:str | None = None,
        ##### Traffic Signal control related variables #####
        phase_controller:   type[BasePhaseController] | Literal["BasicPhaseController", "RuleBasedTSPPhaseController"] = BasicPhaseController, 
        fixed_ts:           bool = False, #if true the obs phase will be wrong
        actuated_ts:        bool = False, #if true the traffic signals will be actuated
        random_ts:          bool = False, #if true the traffic signals action will be chosen randomly
        yellow_time:        int = 3,
        min_green:          int = 10,
        max_green:          int = 50,
        tsp_detection_range_m: float | None = None,
        tsp_max_eta_seconds: float = 25.0,
        tsp_min_speed_mps: float = 2.0,
        tsp_min_delay_seconds: float = 0.0,
        tsp_enable_early_green: bool = False,
        tsp_enable_green_extension: bool = True,
        tsp_early_green_eta_seconds: float = 18.0,
        tsp_force_switch_remaining_seconds: float = 2.0,
        tsp_green_extension_seconds: float = 8.0,
        tsp_skip_buses_at_stops: bool = True,
        tsp_min_bus_speed_for_priority_mps: float = 0.5,
        tsp_min_override_gap_seconds: float | None = None,
        ##### New Arguments #####
        additional_files:   str | None = None,
        sumo_cfg :          str | None = None,
        save_state:         str | None = None, #if None, then the state is not saved. If a string, then the state is saved to the given path
        step_length:        float         = 0.25,  
        agent_id:           str | None = "3040", #allows you to chose a different tls_id
        random_start_time:  None | tuple[int,int] = None, # None if the start time is fixed, else a tuple with (min_start_time, max_start_time) in seconds
        random_seed:        int | None = 3422,
        run_up_time:        int = 0,  # in seconds | Populates the streets for x seconds befor the simulation starts
        init_duration:      int = 72000, # duration for which the simulation will be initialised in this time the number of unique tlsIndex values for buses is logged and saved
        init_start_time:    int | str = "05:30:00",
        out_history_name:   str | None   = None,
        dataset_split:      float = 0.8, # the split rate of the route file into train and test data
        #Display related variables
        use_gui:            bool = False,
        virtual_display:    tuple[int, int] = (3200, 1800),
        render_mode:        str | None = None,
        #Eval Mode related variables
        eval_mode:           bool  = False, # if true the simulation will be run in eval mode which causes it to start at 0 and end when all vehciles are gone
        eval_start_time:     int   = 27_000, # the time at which the simulation starts in eval mode
        eval_duration:       int   = 10_800, # the duration of the simulation in eval mode
        #Observation related variables
        observation:    ObservationFunction | Literal["ObservationFunction", "ObservationFunctionPT", "EmptyObservationFunction"] = ObservationFunctionPT, # type: ignore
        obs_kwargs:           dict[str, dict[str,Any]] | None = None, # additional arguments to pass to the observation function
        obs_every_delta_time: bool = True, # if true the agent observs the environment every delta_time seconds instead of every time he can apply an action
        add_phase_id:         bool = True,
        add_min_green:        bool = True,
        add_phase_last_seen:  bool = True,
        add_density:          bool = True,
        add_queue:            bool = True,
        add_co2:              bool = False,
        add_cvar_delay:       bool = False,
        add_mean_delay:       bool = False,
        add_reward_weights_to_obs: bool = False,
        time_of_day:          Literal["simple","circular",None] = None, #simple: Time of day is included in the state, circular: Time of day is included in the state as a circular variable (normalized sin, cos)
        #Bus observation related variables
        add_distance:       bool = False,
        add_delay:          bool = False,
        add_occupancy:      bool = False,
        #Action related variables
        actions:                    list[str] | Literal["parameters"] | None    = None, #None means that the traffic signal builds the phases/actions itself, parameters means that the list in the parameters.json will be used # type: ignore
        composed_phase_dict:        dict[int,int] | None                        = None, #if the actions are composed of multiple phases this dict will be used to map the composed phase to the individual phases # type: ignore
        minor_phase_list:           list[int] | None                            = None, #each value of the list is the index of a minor phase from the list of actions # type: ignore
        minor_phase_min_length:     int | None                                  = 5, #the length of the minor phases
        no_yellow_between_phase:    dict[int,tuple[int]] | None                 = None,# {1:(0,), 4:(3,)} # if the phase/action moves from one phase to another you get no yellow phase between the two phases
        red_between_phase_groups:   list[tuple[int]] | None                     = None, # [(0,1,2),(3,4,5)] # if the phase/action moves from one phase group to another you get an all red phase 
        all_red_phase_length:       int | None                                  = 4, # the length of the all red phase
        yellow_after_red_phase_len: int | None                                  = 1, # the length of the yellow phase after an all red phase
        #Reward related variables - at some point i should instead pass the reward class directly or i should pass a dict with the parameters that the reward class needs
        reward_fn:          str | None   = None,
        reward_weights:     dict[str, float] | None = None,
        reward_kwargs:      dict[str, dict[str,Any]] | None = None, # additional arguments to pass to the reward function
        reward_scales:      dict[str, tuple[float,float]] | None = None, # min and max values for each reward component for normalization
        reward_clip:        tuple[float,float] = (config.REWARD_MIN_CLIP, config.REWARD_MAX_CLIP),
        reward_norm_with_previous: bool = True,
        multi_objective: bool = False,
        ##### BUS RELATED VARIABLES #####
        bus_vid_prefix:  str = "Bus", # the string that identifies a bus in the vehicle id
        add_per_bus_info:    bool = True,
        bus_tls_indices:     list[int] | None = None,
        #Occupancy related variables
        default_occupancy:     float = config.DEF_OCCUPANCY,
        bus_default_occupancy: float = 10.0, # default occupancy of a bus if no real data is available
        #Dwell time
        dwell_times:            Literal["Fixed", "default", None] = "Fixed",
        scenario_sampling: Literal["sequential","random","curriculum"] = "sequential",
    ) -> None:
        obs_kwargs = {} if obs_kwargs is None else dict(obs_kwargs)
        reward_weights = {} if reward_weights is None else dict(reward_weights)
        reward_kwargs = {} if reward_kwargs is None else dict(reward_kwargs)
        reward_scales = {} if reward_scales is None else dict(reward_scales)

        assert single_agent is True, "Currently only single agent mode is supported, please set single_agent to True"
        assert scenarios is not None or net_file is not None, "Either scenarios or net_file must be provided"
        assert sum([fixed_ts, random_ts, actuated_ts]) <= 1, "Only one of fixed_ts, random_ts or actuated_ts can be true."
        assert sumo_cfg is None or eval_mode is False, "Currently eval_mode is only supported with a sumo_cfg file"
        if sumo_cfg is not None and (route_file is not None or additional_files is not None):
            raise ValueError(f"A sumo_cfg was passed the route_file and additional_files parameters should therefore be None. \n route_file: {route_file} \n additional_files: {additional_files}")
        assert dwell_times in ["Fixed", "default", None], "dwell_times must be one of ['Fixed', 'default', None]"
        assert actions is None or isinstance(actions, list) or actions == "parameters", "actions must be a list of strings, 'parameters' or None"
        assert reward_fn is not None or reward_weights != {}, "If no reward_fn is provided, reward_weights cannot be empty"
        assert sum((reward_fn is not None,reward_weights != {})) < 2, "Only one of reward_fn and reward_weights can be provided"
        assert reward_clip[0] < reward_clip[1], "reward_clip min must be smaller than reward_clip max"
        assert tsp_detection_range_m is None or tsp_detection_range_m >= 0.0, "tsp_detection_range_m must be >= 0 when provided"
        assert tsp_max_eta_seconds >= 0.0, "tsp_max_eta_seconds must be >= 0"
        assert tsp_min_speed_mps > 0.0, "tsp_min_speed_mps must be > 0"
        assert tsp_early_green_eta_seconds >= 0.0, "tsp_early_green_eta_seconds must be >= 0"
        assert tsp_force_switch_remaining_seconds > 0.0, "tsp_force_switch_remaining_seconds must be > 0"
        assert tsp_green_extension_seconds > 0.0, "tsp_green_extension_seconds must be > 0"
        assert tsp_min_bus_speed_for_priority_mps >= 0.0, "tsp_min_bus_speed_for_priority_mps must be >= 0"
        if tsp_min_override_gap_seconds is not None:
            assert tsp_min_override_gap_seconds >= 0.0, "tsp_min_override_gap_seconds must be >= 0 when provided"
        random.seed(random_seed)
        self.random_seed = random_seed
        self._py_random = random.Random(random_seed) if random_seed is not None else random.Random()
        '''Display related variables''' 
        assert render_mode is None or render_mode in ["human", "rgb_array"], "Invalid render mode."
        self.render_mode        = render_mode
        self.virtual_display    = virtual_display
        self.disp               = None
        self.use_gui            = use_gui and not LIBSUMO
        #Use the gui sumo if the render_mode is not None or use_gui is True
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary: str = sumolib.checkBinary("sumo-gui") # type: ignore
        else:
            self._sumo_binary: str = sumolib.checkBinary("sumo") # type: ignore

        '''SUMO related variables'''
        '''SUMO start related variables'''
        self.sumo_cfg               = sumo_cfg
        self._net: str              = net_file if net_file is not None else scenarios[0][0] if scenarios is not None else ""
        self.route_file: str | list[str] | None = route_file if isinstance(route_file, str) else self._get_sorted_route_files(route_file) if isinstance(route_file, list) else None
        self.flow_file              = flow_file
        self.scenarios              = scenarios
        self.scenario_sampling      = scenario_sampling
        self.scenario_idx           = 0
        self.vehicles_file          = vehicles_file
        self.save_state:str | None = save_state 
        self.dataset_split       = dataset_split
        if isinstance(route_file, str):
            self._route:str    = route_file
        elif scenarios is None:
            self._route_files_train_test_split() #creates self.train and self.test routes
        self.begin_time:            int  = begin_time if isinstance(begin_time, int) else nseconds_from_str(begin_time)
        self.original_begin_time:   int  = self.begin_time #saving the original begin time so that it can be used in the reset method, otherwise after few smaller episodes the begin time goes negative and training stops
        self.delta_time             = delta_time # seconds on sumo at each step
        # self.additional_files       = additional_files if not fixed_ts else ",".join([file for file in additional_files.split(",") if "actuated" not in file]) # if fixed_ts is true the actuated files are removed so that the original phases are used which are specified in the net file
        if additional_files is None:
            self.additional_files = None
        elif fixed_ts:
            # Only use files that do NOT contain 'actuated'
            self.additional_files = ",".join(
                [file for file in additional_files.split(",") if "actuated" not in file]
            )
        else:
            self.additional_files = additional_files

        self.step_length            = step_length
        self.num_seconds            = num_seconds
        self.sumo_warnings          = sumo_warnings
        self.additional_sumo_cmd    = additional_sumo_cmd
        self.random_start_time      = random_start_time
        self.sumo_seed              = sumo_seed
        '''Eval mode related variables'''
        self.eval_mode              = eval_mode
        self.eval_start_time        = eval_start_time
        self.eval_end_time          = eval_start_time + eval_duration
        '''SUMO simulation behaviour related variables'''
        self.time_to_teleport       = time_to_teleport
        self.max_depart_delay       = max_depart_delay
        self.waiting_time_memory    = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.run_up_time            = run_up_time
        '''SUMO related variables'''
        SumoEnvironment.CONNECTION_LABEL    += 1 # type: ignore
        self.sumo           = None
        self._sumo_started  = False
        self.label          = str(SumoEnvironment.CONNECTION_LABEL)

        '''Traffic signal related variables'''
        self.min_green      = min_green
        self.max_green      = max_green
        self.yellow_time    = yellow_time
        self.tsp_detection_range_m = tsp_detection_range_m
        self.tsp_max_eta_seconds = tsp_max_eta_seconds
        self.tsp_min_speed_mps = tsp_min_speed_mps
        self.tsp_min_delay_seconds = tsp_min_delay_seconds
        self.tsp_enable_early_green = tsp_enable_early_green
        self.tsp_enable_green_extension = tsp_enable_green_extension
        self.tsp_early_green_eta_seconds = tsp_early_green_eta_seconds
        self.tsp_force_switch_remaining_seconds = tsp_force_switch_remaining_seconds
        self.tsp_green_extension_seconds = tsp_green_extension_seconds
        self.tsp_skip_buses_at_stops = tsp_skip_buses_at_stops
        self.tsp_min_bus_speed_for_priority_mps = tsp_min_bus_speed_for_priority_mps
        self.tsp_min_override_gap_seconds = tsp_min_override_gap_seconds

        self.phase_controller_class: type[BasePhaseController] = phase_controller if not isinstance(phase_controller, str) else getattr(phase_controllers, phase_controller)
        self.fixed_ts       = fixed_ts
        self.actuated_ts    = actuated_ts
        self.random_ts      = random_ts

        '''RL related variables'''
        self.single_agent       = single_agent
        self.agent_id:  str | None = agent_id
        self.reward_fn: str | None = reward_fn
        self.reward_kwargs      = reward_kwargs
        self.reward_scales      = reward_scales
        self.reward_weights: dict[Any,Any]     = reward_weights if reward_weights != {} else {reward_fn : 1.0} if reward_fn is not None else {}
        self.reward_clip        = reward_clip
        self.reward_norm_with_previous = reward_norm_with_previous
        self.multi_objective = multi_objective
        self.episode            = 0
        self.reward_range       = (config.REWARD_MIN_CLIP, config.REWARD_MAX_CLIP)
        self.observation        = observation if not isinstance(observation, str) else getattr(observations, observation)
        self.obs_kwargs         = obs_kwargs
        self.obs_every_delta_time= obs_every_delta_time # if true the agent observs the environment every delta_time seconds instead of every time he can apply an action
        self.env_history: list[tuple[obs_type, np.ndarray[Any,Any], bool, bool, dict[str, float], int, dict[str, float | int | action_type]]] = []
        '''Info related variables'''
        self.add_system_info    = add_system_info
        self.add_per_agent_info = add_per_agent_info
        '''Logging related variables'''
        self.out_history_name   = out_history_name
        self.vehicle_tl_dict    = {}
        '''Observation related variables'''
        self.add_phase_id   = add_phase_id
        self.add_min_green  = add_min_green
        self.add_density    = add_density
        self.add_queue      = add_queue
        self.add_co2        = add_co2
        self.add_phase_last_seen    = add_phase_last_seen
        self.add_cvar_delay         = add_cvar_delay
        self.add_mean_delay         = add_mean_delay
        self.add_reward_weights_to_obs = add_reward_weights_to_obs
        #Bus Obs
        self.add_distance           = add_distance
        self.add_delay              = add_delay
        self.add_occupancy          = add_occupancy
        self.time_of_day            = time_of_day
        '''Action related variables'''
        self.actions:list[str] | Literal["parameters"] | None   = actions
        self.composed_phase_dict:       dict[int,int] | None    = composed_phase_dict 
        self.minor_phase_list:          list[int] | None        = minor_phase_list
        self.minor_phase_min_length:    int | None              = minor_phase_min_length
        self.no_yellow_between_phase:   dict[int,tuple[int]] | None = no_yellow_between_phase # {1:(0), 4:(3)} # if the phase/action moves from one phase to another you get no yellow phase between the two phases
        self.red_between_phase_groups:  list[tuple[int]] | None = red_between_phase_groups 
        self.all_red_phase_length:      int | None              = all_red_phase_length
        self.yellow_after_red_phase_len:int | None              = yellow_after_red_phase_len # the length of the yellow phase after an all red phase
        '''Bus related variables'''
        self.buses:BusStore     = BusStore(self, bus_factory=lambda bus_id: Bus(self, bus_id, dwell_times=dwell_times),) # only required for the initailisation of the env will be reset on each episode
        self.bus_vid_prefix     = bus_vid_prefix
        self.add_per_bus_info   = add_per_bus_info
        self.dwell_times        = dwell_times
        self.bus_data_manager   = BusDataManager(self) # manages the current list of active and inactive buses and loads the bus data 
        #occupancy related variables
        self.default_occupancy     = default_occupancy
        self.bus_default_occupancy = bus_default_occupancy
        '''Initialisation related variables'''
        '''Config file initialisation related variables''' #Currently does nothing
        self.init_duration      = init_duration
        self.init_start_time    = init_start_time
        '''Load the config file'''
        self.bus_tls_indices   = bus_tls_indices
        
        '''Start the Simulation in order to retrieve the traffic light ids and their information''' 
        self._init_start_simulation()

        signal(SIGINT, self._signal_handler)
        atexit.register(self._cleanup)

    def _signal_handler(self, sig: int, frame: FrameType | None) -> NoReturn:
        self._cleanup()
        sys.exit(0)

    def _init_start_simulation(self):
        '''Start the Simulation in order to retrieve the traffic light ids and their information''' 
        sumo_cmd = [self._sumo_binary, "-n", self._net]
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")

        if LIBSUMO or not TRACI_SUPPORTS_CONNECTION_LABELS:
            traci.start(sumo_cmd)  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start(sumo_cmd, label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label) # type: ignore
        
        self.tls_ids = list(conn.trafficlight.getIDList())
        
        if self.agent_id is not None:
            assert self.agent_id in self.tls_ids, f"The provided agent_id {self.agent_id} is not in the list of traffic lights in the network: {self.tls_ids}"
            self.tls_ids = [self.agent_id]
        self.tls_id:str = self.tls_ids[0]

        self.sumo = conn

        self.traffic_signals = self._get_tl_dict()
        
        conn.close()
        # The bootstrap connection is only used for metadata extraction.
        # Keep runtime state marked as not started to avoid double-closing.
        self.sumo = None
        self._sumo_started = False

    def _get_sorted_route_files(self,route_files:list[str])->list[str]:
        return list(dict(sorted({int(file_name.split("_")[-2]): file_name for file_name in route_files}.items())).values())

    def _route_files_train_test_split(self)->None:
        assert isinstance(self.route_file,list) and len(self.route_file) >= 2, "The route file must be a list of atleast two files to be splitted into train and test data" # type: ignore
        
        route_files = self.route_file.copy() 
        split_index = int(len(route_files) * self.dataset_split)

        self.train_routes:  list[str]   = route_files[:split_index]
        self.test_routes:   list[str]   = route_files[split_index:]
        self.test_route:    str         =  self._py_random.sample(self.test_routes,1)[0]

    def _get_route_path(self)->str:
        return ",".join([s for s in [self._route, self.flow_file, self.vehicles_file] if s is not None]) # type: ignore

    def _start_simulation(self):

        '''Init lists and dicts that are reset on each episode'''
        self.history: list[tuple[obs_type, np.ndarray[Any,Any], bool, bool, dict[str, float]]] = []
        self.trajectory:trajectory_type = []
        self.buses  = BusStore(self)

        self.observations   = {ts: [0.0] * self.traffic_signals[self.tls_id].observation_space.shape[0] for ts in self.tls_ids}
        self.rewards        = {ts: np.zeros(len(self.reward_weights), dtype=np.float64) for ts in self.tls_ids}
        
        '''Start the sumo simulation with the cfg file or with a sumo command'''
        if self.sumo_cfg is not None:
            sumo_cmd = [self._sumo_binary, "-c" ,self.sumo_cfg]
            if not self.sumo_warnings:
                sumo_cmd.append("--no-warnings")
        else:
            #If eval_mode is true the simulation starts at 0, if eval_mode is false and random_start_time is True, the simulation will start at a random time between 0 and 24 hours
            if self.eval_mode:
                self.begin_time = self.eval_start_time # dont use 0 since during the night not much happens
            elif self.random_start_time is not None:
                self.begin_time = self._compute_random_starting_time()
            else:
                self.begin_time:int = self.begin_time - self.run_up_time
            if self.scenarios is not None:
                idx = self.scenario_idx if len(self.scenarios) > 1 else 0
                self._net   = self.scenarios[idx][0]
                self._route = self.scenarios[idx][1]
            elif isinstance(self.route_file,list):
                self._route:str = self._py_random.sample(self.train_routes,1)[0] if not self.eval_mode else self.test_route #random.sample(self.test_routes, k=1)[0] The problem is if i random split the individual eval runs can not be compared which is a problem
            self.sim_max_time = self.num_seconds + self.begin_time  + self.run_up_time

            # construct the sumo starting command
            sumo_cmd = [
                self._sumo_binary,
                "-n",
                self._net,
                "-r",
                self._get_route_path(),
                "--max-depart-delay",
                str(self.max_depart_delay),
                "--waiting-time-memory",
                str(self.waiting_time_memory),
                "--time-to-teleport",
                str(self.time_to_teleport),
                "--step-length",
                str(self.step_length),
            ]
            
            ##### new code below #####
            if self.additional_files is not None:
                sumo_cmd.extend(["--additional-files", self.additional_files])
            if isinstance(self.save_state, str):
                if not os.path.exists(self.save_state):
                    os.makedirs(self.save_state)
                sumo_cmd.extend(["--fcd-output", os.path.join(self.save_state, f"fcd_{self.tls_id}.xml")])

            #     sumo_cmd.extend(["--save-state", self.save_state])
            ##### new code above #####
            if self.begin_time > 0: # type: ignore
                sumo_cmd.append(f"-b {self.begin_time}")
            if self.sumo_seed == "random":
                sumo_cmd.append("--random")
            else:
                sumo_cmd.extend(["--seed", str(self.sumo_seed)])
            if not self.sumo_warnings:
                sumo_cmd.append("--no-warnings")
            if self.additional_sumo_cmd is not None:
                sumo_cmd.extend(self.additional_sumo_cmd.split())
            if self.use_gui or self.render_mode is not None:
                sumo_cmd.extend(["--start", "--quit-on-end"])
                if self.render_mode == "rgb_array":
                    warnings.warn("use_gui is currently not supported", stacklevel=2)
        
        self.sumo_cmd = sumo_cmd # save the sumo_cmd for later use
        
        if LIBSUMO or not TRACI_SUPPORTS_CONNECTION_LABELS:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label) # type: ignore
        self._sumo_started = True

        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world") # type: ignore

    @cleanup_decorator # type: ignore
    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[obs_type, dict[str, float]]:
        """Reset the environment."""
        # Reset to configured baseline so run_up subtraction does not accumulate across episodes.
        self.begin_time:int = self.original_begin_time
        super().reset(seed=seed, **kwargs)

        if self.episode != 0:
            self.close()
            if self.out_history_name is not None: 
                self.save_history(self.out_history_name,self.episode)
            # Scenario changes are applied below via scenario_idx selection and a fresh
            # _start_simulation() + _get_tl_dict() rebuild for the new episode.
        elif self.episode == 0 and self.reward_scales == {}:
            self.reward_scales = self._init_reward_scales()
        self.episode += 1
        if self.scenarios is not None and len(self.scenarios) > 1:
            if self.scenario_sampling == "random":
                self.scenario_idx = self._py_random.randrange(len(self.scenarios))
            elif self.scenario_sampling == "curriculum":
                # Monotonic curriculum: consume scenarios in order, then keep using the last.
                self.scenario_idx = min(self.episode - 1, len(self.scenarios) - 1)
            else:
                # Deterministic sequential sampling with wraparound.
                self.scenario_idx = (self.episode - 1) % len(self.scenarios)

        if seed is not None:
            self.sumo_seed = seed
            self._py_random.seed(seed)
        self._start_simulation()
        self.traffic_signals = self._get_tl_dict()
        if self.run_up_time > 0:
            self._run_up_simulation()

        self._update_vehicle_info()
        
        self.traffic_signals[self.tls_id].next_action_time = self.sim_step
        self.traffic_signals[self.tls_id].rewards.previous_reward = self.traffic_signals[self.tls_id].compute_reward() # compute the initial reward - for normalising purposes - do it after the run uptime so that cars populate the streets
        if sum((self.fixed_ts, self.actuated_ts)) == 0:
            self.traffic_signals[self.tls_id].disable_fixed_ts()

        obs = self._compute_observations()[self.tls_id]

        self.trajectory.append((obs, np.zeros(len(self.reward_weights), dtype=np.float64), None, list(self.reward_weights.values())))
       
        return obs, self._compute_info()

    @cleanup_decorator # type: ignore
    def step(self,
             action: action_type
             )->tuple[obs_type, reward_type, bool, bool, dict[str, float]]:
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (dict | int): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        #Fixed/Acuated TS - No action, follow fixed TL defined in self.phases
        if action is None or action == {} or self.fixed_ts or self.actuated_ts:
            for _ in range(int(self.delta_time/self.step_length)):
                self._sumo_step()
                self._update_vehicle_info()
                if self.single_agent:
                    self.traffic_signals[self.tls_id].next_action_time = self.sim_step
                self.traffic_signals[self.tls_id].update()
        #RL TS or random TS
        else:
            self._apply_actions(action if not self.random_ts else self.action_space.sample()) #added the option to select a raodm action
            self._run_steps()

        observations:   obs_type = self._compute_observations()[self.tls_id]
        rewards:        NDArray[np.float64] = self._compute_rewards()[self.tls_id]
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones["__all__"]  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        self.trajectory.append((observations, rewards, action, list(self.reward_weights.values())))# this is a_{t-1}
        self.history.append((observations, rewards, terminated, truncated, info))
        self.env_history.append((observations, rewards, terminated, truncated, info, self.episode, self.get_step_info()))

        reward: reward_type = rewards if self.multi_objective else float(np.sum(rewards))
        return observations, reward, terminated, truncated, info

    @property
    def sim_step(self) -> float:
        """Return current simulation second on SUMO."""
        return float(self.sumo.simulation.getTime())

    def _update_vehicle_info(self)->None:
        self.vehicleIDList:tuple[str,...] = self.sumo.vehicle.getIDList() #get all vehicle ids
        self.bus_data_manager.update_bus_dict() #updates the value of the bus data
        #update the time that the vehicle enter the bound of relevance for each traffic lights
        for agent in self.traffic_signals:
            self.traffic_signals[agent].update_vehicle_dicts()

    def _run_steps(self):
        time_to_observe = False if not self.obs_every_delta_time else True
        i = 0
        while not time_to_observe or i < int(self.delta_time/self.step_length):
            i = i + 1
            self._sumo_step()
            self._update_vehicle_info()
            for ts in self.tls_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_observe and not self.obs_every_delta_time:
                    time_to_observe = True

    def _apply_actions(self, actions: action_type):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {tls_id : greenPhase}
        """
        if self.single_agent:
            if self.traffic_signals[self.tls_id].time_to_observe:
                self.traffic_signals[self.tls_id].set_next_phase(actions)

    def _compute_dones(self):
        dones = {tls_id: False for tls_id in self.tls_ids}
        #if eval mode is active the simulation will end when all vehicles are gone
        if self.eval_mode:
            dones["__all__"] = self.sim_step >= self.eval_end_time
        else:
            dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones
    
    def get_done(self):
        return self._compute_dones()["__all__"]

    def _compute_observations(self) -> dict[str, obs_type]:
        self.observations.update(
            {ts: self.traffic_signals[ts].compute_observation() for ts in self.tls_ids if self.traffic_signals[ts].time_to_observe or self.obs_every_delta_time}
        )
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_observe or self.obs_every_delta_time}  # type: ignore

    def _compute_rewards(self) -> dict[str, NDArray[np.float64]]:
        self.rewards.update(
            {ts: self.traffic_signals[ts].compute_reward() for ts in self.tls_ids if self.traffic_signals[ts].time_to_observe or self.obs_every_delta_time} 
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_observe or self.obs_every_delta_time}

    def _sumo_step(self):
        #take the sumo step
        self.sumo.simulationStep()

    def close(self):
        """Close the environment and stop the SUMO simulation."""
        if not self._sumo_started:
            self.sumo = None
            return

        should_close = True
        if LIBSUMO and hasattr(traci, "isLoaded"):
            try:
                should_close = bool(traci.isLoaded())
            except Exception:
                should_close = False

        if should_close:
            try:
                if not LIBSUMO and TRACI_SUPPORTS_CONNECTION_LABELS:
                    traci.switch(self.label) # type: ignore
                traci.close()
            except Exception:
                pass

        if self.disp is not None:
            self.disp.stop()
            self.disp = None

        self.sumo = None
        self._sumo_started = False

    def _cleanup(self):
        """Clean up the environment."""

    def cleanup(self):
        """Clean up the environment."""
        self._cleanup()

    def __del__(self):
        """Close the environment and stop the SUMO simulation."""
        self._cleanup()

    def render(self) -> None | NDArray[np.uint8]:
        """Render the environment.

        If render_mode is "human", the environment will be rendered in a GUI window using pyvirtualdisplay.
        """
        if self.render_mode == "human":
            return  # sumo-gui will already be rendering the frame
        elif self.render_mode == "rgb_array":
            img = self.disp.grab() # type: ignore
            return np.array(img) # type: ignore

    def save_history(self, out_history_name: str, episode: int):
        Path(out_history_name).mkdir(parents=True, exist_ok=True)
        with open(f"{out_history_name}_{episode}.pkl", "wb") as f:
            pickle.dump(self.history, f)

    # # Below functions are for discrete state space
    # def encode(self, state, tls_id):
    #     """Encode the state of the traffic signal into a hashable object."""
    #     phase = int(np.where(state[: self.traffic_signals[tls_id].num_green_phases] == 1)[0])
    #     min_green = state[self.traffic_signals[tls_id].num_green_phases]
    #     density_queue = [discretize_density(d) for d in state[self.traffic_signals[tls_id].num_green_phases + 1 :]]
    #     # tuples are hashable and can be used as key in python dictionary
    #     return tuple([phase, min_green] + density_queue)

    def _compute_random_starting_time(self) -> int:
        assert self.random_start_time is not None, "random_start_time must be set to compute a random starting time"
        return self._py_random.randint(self.random_start_time[0], self.random_start_time[1] - self.num_seconds)
    
    def _get_tl_dict(self,
                     reward_weights: dict[Any,Any] | None = None,
                     reward_clip: tuple[float, float] | None = None) -> dict[str,TrafficSignal]:
        reward_weights = reward_weights if reward_weights is not None else self.reward_weights
        reward_clip = reward_clip if reward_clip is not None else self.reward_clip
        traffic_signals = {
            ts: TrafficSignal(
                self,
                self.phase_controller_class,
                ts,
                self.delta_time,
                self.yellow_time, 
                self.min_green, 
                self.max_green, 
                reward_weights,
                self.reward_kwargs,
                reward_clip,
                self.reward_scales,
                self.reward_norm_with_previous,
                self.obs_kwargs,
            )
            for ts in self.tls_ids
        }
        return traffic_signals
    
    '''
    Properties
    '''
    @property
    def observation_space(self)->gym.spaces.Box: # type: ignore
        """Return the observation space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.tls_id].observation_space

    @property
    def action_space(self): # type: ignore
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        return self.traffic_signals[self.tls_id].action_space
    
    @property
    def reward_space(self) -> gym.spaces.Box:
        """Return the vector reward space for MORL algorithms."""
        dim = len(self.reward_weights)
        low = np.full(dim, self.reward_clip[0], dtype=np.float64)
        high = np.full(dim, self.reward_clip[1], dtype=np.float64)
        return gym.spaces.Box(low=low, high=high, dtype=np.float64)
    
    '''
    Simulation run up and initialization methods
    '''
    def _run_up_simulation(self):
        for _ in range(int(self.run_up_time/self.step_length)):
            self._sumo_step()
            self._update_green_phase_last_seen()

    def _init_reward_scales(self) -> dict[str,tuple[float,float]]:
        reward_run_up_history: list[np.ndarray[Any,Any]] = []
        self._start_simulation()

        start_time: float = self.sim_step
        duration:   int   = (self.run_up_time if self.run_up_time > 0 else 3_600)

        self.traffic_signals = self._get_tl_dict(reward_weights={k:1 for k in self.reward_weights},reward_clip=(float("-inf"),float("inf"))) # create a tl dict with reward weights of 1 and no reward clipping so that the rewards are not clipped during the run up phase and the scales are computed correctly

        while self.sim_step < start_time + duration*3:
            if self.sim_step == start_time + duration*2:
                self.traffic_signals[self.tls_id].disable_fixed_ts()
            elif self.sim_step > start_time + duration:
                self._apply_actions(self.action_space.sample())
            self._run_steps()
            
            self._update_green_phase_last_seen()
            
            reward_run_up_history.append(self.traffic_signals[self.tls_id].compute_reward())

        self.close()

        reward_scales:dict[str,tuple[float,float]] = {k:(min([r[i] for r in reward_run_up_history]),max([r[i] for r in reward_run_up_history])) for i,k in enumerate(self.reward_weights.keys())}
        return {k: v for k,v in reward_scales.items() if v[0] != v[1]} # remove the reward scales where min and max are the same
    
    def _update_green_phase_last_seen(self):
        tl_state:str = self.sumo.trafficlight.getRedYellowGreenState(self.tls_id) 
        if tl_state in [p.state for p in self.traffic_signals[self.tls_id].green_phases]:
            self.traffic_signals[self.tls_id]._green_phase_last_seen[[i for i,p in enumerate(self.traffic_signals[self.tls_id].green_phases) if p.state == tl_state][0]] = self.sim_step
