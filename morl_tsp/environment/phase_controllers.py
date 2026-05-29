# SPDX-FileCopyrightText: Copyright (c) Lucas Alegre and SUMO-RL contributors
# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import libsumo
    from env import SumoEnvironment

    from morl_tsp.environment.traffic_signal import TrafficSignal

logging.basicConfig(filename='phase_controller.log', 
                    filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s', 
                    level=logging.DEBUG)

class BasePhaseController(ABC):
    """Base class for Sumo Phase Controllers.

    This class is used to control the traffic signal phases in a SUMO simulation.
    It provides methods to set the next phase, update the traffic signal state,
    and manage the phase queue.
    """

    def __init__(self, 
                 ts         :TrafficSignal,
                 delta_time :int,
                 yellow_time:int,
                 min_green  :int,
                 max_green  :int,) -> None: 
        """Initializes the base phase controller.

        Args:
            ts (TrafficSignal): The TrafficSignal instance to control.
        """
        self.ts   = ts
        self.env:SumoEnvironment  = ts.env
        self.sumo = ts.sumo
        self.id   = self.ts.id

        '''Traffic signal properties'''
        #Timings
        self.delta_time     :int = delta_time
        self.yellow_time    :int = yellow_time
        self.min_green      :int = min_green
        self.max_green      :int = max_green

        self.all_red_phase_length   :int | None = self.env.all_red_phase_length
        self.minor_phase_min_length :int | None = self.env.minor_phase_min_length
        #Phases
        self.actions             :list[str] | Literal["parameters"] | None    = self.env.actions #None means that the traffic signal builds the phases/actions itself, parameters means that the list in the parameters.json will be used 
        self.composed_phase_dict :dict[int,int] | None                        = self.env.composed_phase_dict #if the actions are composed of multiple phases this dict will be used to map the composed phase to the individual phases
        self.minor_phase_list    :list[int]|None                              = self.env.minor_phase_list

        self.red_between_phase_groups_dict: dict[int,int]|None  = {val: idx for idx, phase_group in enumerate(self.env.red_between_phase_groups)for val in phase_group} if self.env.red_between_phase_groups is not None else None

        '''Traffic signal state properties'''
        self.green_phase:int    = 0
        self.current_phase:int  = 0 # the same as green_phase but it includes the all red phase if it is present

        '''Original traffic signal'''
        self.original_programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        self.original_logic = self.original_programs[0]

        # Episode-level stability accounting (used for eval reporting).
        self.phase_change_count: int = 0
        self.forced_max_green_count: int = 0
        self.infeasible_action_count: int = 0
        self.last_action_infeasible: bool = False
        self._pending_infeasible_penalty: bool = False
        self.action_call_count: int = 0
        self._controller_step_count: int = 0
        self._max_green_exceed_step_count: int = 0
        self._green_run_durations_seconds: list[float] = []
        self._current_green_run_seconds: float = 0.0

        self._build_phases()

    @property
    @abstractmethod
    def time_to_observe(self) -> bool:
        """Returns True if the traffic signal should act in the current step."""
        pass

    @abstractmethod
    def set_next_phase(self, new_phase: int):
        """Sets the next phase of the traffic signal.

        Is called when the traffic signal should change its phase.

        Args:
            new_phase (int): The new phase to set.
        """
        pass

    @abstractmethod
    def update(self):
        """Updates the traffic signal state.

        Is called on every step of the simulation.
        """
        pass

    @abstractmethod
    def can_change_phase(self) -> Literal[0, 1]:
        """Returns 1 if the traffic signal can change its phase.

        This is used for the observation space. to indicate to the agent that he can change the phase.
        """
        pass

    def _register_action_call(self) -> None:
        self.action_call_count += 1

    def _register_infeasible_action(self) -> None:
        self.infeasible_action_count += 1
        self.last_action_infeasible = True
        self._pending_infeasible_penalty = True

    def _register_phase_change(self) -> None:
        self.phase_change_count += 1

    def _register_forced_max_green(self) -> None:
        self.forced_max_green_count += 1

    def _tick_controller_step(self) -> None:
        self._controller_step_count += 1

    def _tick_green_hold(self, step_seconds: float) -> None:
        self._current_green_run_seconds += max(0.0, float(step_seconds))
        if self.max_green > 0 and self._current_green_run_seconds > float(self.max_green) + 1e-9:
            self._max_green_exceed_step_count += 1

    def _finalize_green_hold(self) -> None:
        if self._current_green_run_seconds > 0.0:
            self._green_run_durations_seconds.append(float(self._current_green_run_seconds))
            self._current_green_run_seconds = 0.0

    def _max_green_reached(self) -> bool:
        return self.max_green > 0 and self._current_green_run_seconds >= float(self.max_green) - 1e-9

    def _next_distinct_phase(self, current_phase: int) -> int:
        if self.num_green_phases <= 1:
            return int(current_phase)
        return int((int(current_phase) + 1) % int(self.num_green_phases))

    def consume_pending_infeasible_penalty(self) -> bool:
        pending = bool(self._pending_infeasible_penalty)
        self._pending_infeasible_penalty = False
        return pending

    def get_episode_stability_metrics(self) -> dict[str, float]:
        run_durations = list(self._green_run_durations_seconds)
        if self._current_green_run_seconds > 0.0:
            run_durations.append(float(self._current_green_run_seconds))

        mean_green_duration = (
            float(sum(run_durations) / len(run_durations)) if len(run_durations) > 0 else 0.0
        )
        sim_minutes = (
            (float(self._controller_step_count) * float(self.env.step_length)) / 60.0
            if self._controller_step_count > 0
            else 0.0
        )
        phase_change_rate = (
            float(self.phase_change_count) / sim_minutes if sim_minutes > 0.0 else 0.0
        )
        infeasible_action_rate = (
            float(self.infeasible_action_count) / float(self.action_call_count)
            if self.action_call_count > 0
            else 0.0
        )
        max_green_exceed_step_fraction = (
            float(self._max_green_exceed_step_count) / float(self._controller_step_count)
            if self._controller_step_count > 0
            else 0.0
        )

        return {
            "phase_change_count": float(self.phase_change_count),
            "phase_change_rate_per_min": float(phase_change_rate),
            "mean_green_duration_seconds": float(mean_green_duration),
            "infeasible_action_rate": float(infeasible_action_rate),
            "forced_max_green_count": float(self.forced_max_green_count),
            "max_green_exceed_step_fraction": float(max_green_exceed_step_fraction),
            "max_green_seconds": float(self.max_green),
        }

    '''
    Phase/Action related methods
    '''

    def _build_phases(self):

        if self.env.fixed_ts or self.env.actuated_ts:
            self.num_green_phases = len(self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            self.green_phases = self._get_green_phases()
            self.num_phases:        int = self.num_green_phases if self.env.red_between_phase_groups is None else self.num_green_phases + 1 #if there are no red phases between the green phases, the number of phases is equal to the number of green phases, otherwise we have one more phase for the red phase
            self.all_phases, self.yellow_dict = self._get_transition_phases(self.green_phases)
            self.phase_len = len(self.all_phases[0].state)  # length of the phase string
            return None

        ##### Build the green Phases
        if self.env.actions is None:
            self.green_phases = self._get_green_phases()
        else:
            self.green_phases = self._get_custom_green_phases()
        self.num_green_phases:  int = len(self.green_phases)
        self.num_phases:        int = self.num_green_phases if self.env.red_between_phase_groups is None else self.num_green_phases + 1 #if there are no red phases between the green phases, the number of phases is equal to the number of green phases, otherwise we have one more phase for the red phase

        ##### Build the yellow Phases
        self.all_phases, self.yellow_dict = self._get_transition_phases(self.green_phases)
        self.phase_len = len(self.all_phases[0].state)  # length of the phase string

    def disable_fixed_ts(self):
        '''
        This function disables the fixed traffic signal control
        '''
        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    def _get_green_phases(self
                          )->list[libsumo.TraCIPhase]: 
        '''
        This function returns the green phases of the traffic signal

        Return:
        green_phases: list of green phases
            [Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrGrGrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrGrrrrrrrrrrrGrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgGrGrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrrrrrrGrrrrrrrrrrrGrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0)]
        '''
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        green_phases:list[libsumo.TraCIPhase] = []
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                green_phases.append(self.sumo.trafficlight.Phase(60, state))

        return green_phases

    def _get_custom_green_phases(self,
                                 phases:list[str] | None  = None  # type: ignore
                                 )->list[libsumo.TraCIPhase]:
        '''
        This function returns the custom actions of the traffic signal

        Parameters:
        phases: list[str] | None 
            if None the actions from the environment are used
            list of custom actions
            ["GggGGgrrrrrrGggGGgrrrrrrrrrrrrr",
             "rrrrrGrrrrrrrrrrrGrrrrrrrrrrrrr",
             "rrrrrrGggGGgrrrrrrGggGGgrrrrrrr",
             "rrrrrrrrrrrGrrrrrrrrrrrGrrrrrrr"]

        Return:
        green_phases: list of green phases
            [Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrGrGrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrGrrrrrrrrrrrGrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgGrGrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrrrrrrGrrrrrrrrrrrGrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0)]
        '''

        if phases is None:
            assert self.env.actions is not None, "Custom actions are not defined - set actions in the environment - _get_custom_green_phases should only be called if env.actions is not None if you dont want custom actions use _get_green_phases()"
            
            if isinstance(self.env.actions,list):
                phases:list[str] = self.env.actions
            else:
                raise ValueError("actions should be a list of strings")

        green_phases:list[libsumo.TraCIPhase] = [self.sumo.trafficlight.Phase(60, phase) for phase in phases]

        return green_phases

    def _get_transition_phases(self,
                               green_phases:list[libsumo.TraCIPhase],  
                               )->tuple[list[libsumo.TraCIPhase],dict[tuple[int,int],int]]: 
        '''
        This function creates the yellow phases between the green phases and a dict that defines how to go from one green phase to another

        Parameters:
        green_phases: list of green phases
            [Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrGrGrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrGrrrrrrrrrrrGrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgGrGrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrrrrrrGrrrrrrrrrrrGrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0)]

        Returns:
        all_phases: list[traci.libsumo.TraCIPhase] 
            list of all phases
            [Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrGrGrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='GggGGgrrrrrrGggGGgrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrGrrrrrrrrrrrGrrrrrrrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgGrGrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrGggGGgrrrrrrGggGGgrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=60.0, state='rrrrrrrrrrrGrrrrrrrrrrrGrrrrrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=3.0, state='GggGGgrrrrrrGggGGgrrrrrrryryrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=3.0, state='yyyyygrrrrrryyyyygrrrrrrryryrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=3.0, state='yyyyyyrrrrrryyyyyyrrrrrrryryrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=3.0, state='yyyyyyrrrrrryyyyyyrrrrrrryryrrr', minDur=-1073741824.0, maxDur=-1073741824.0),
             Phase(duration=3.0, state='yyyyyyrrrrrryyyyyyrrrrrrryryrrr', minDur=-1073741824.0, maxDur=-1073741824.0),

        yellow_dict: dict[tuple[int,int],int]
            dictionary that specifies which yellow phase to use when transitioning from one green phase to another
            {(0, 1): 6,
             (0, 2): 7,
             (0, 3): 8,
             (0, 4): 9,
             (0, 5): 10,
             (1, 0): 11,
             (1, 2): 12,
             (1, 3): 13,
             (1, 4): 14,
             (1, 5): 15,
             (2, 0): 16,
        '''

        yellow_dict: dict[tuple[int,int],int] = {}
        all_phases:list[libsumo.TraCIPhase] = green_phases.copy()
        for i, p1 in enumerate(green_phases):
            for j, p2 in enumerate(green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                yellow_dict[(i, j)] = len(all_phases)
                all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        return all_phases, yellow_dict


class BasicPhaseController(BasePhaseController):
    def __init__(self, 
                 ts:TrafficSignal,
                 delta_time :int,
                 yellow_time:int,
                 min_green  :int,
                 max_green  :int) -> None: 
        """Initializes the basic phase controller.

        Args:
            sumo (libsumo.Sumo): The Sumo instance to control.
        """
        super().__init__(ts,
                         delta_time,
                         yellow_time,
                         min_green,
                         max_green)
        self.next_action_time   :float = self.env.begin_time

        self.is_yellow          :bool = False
        self.is_red_yellow      :bool = False #is true if we go from red to yellow
        self.is_red             :bool = False
        self.next_phase_is_red  :bool = False

        self.obs_every_delta_time :bool = self.env.obs_every_delta_time

        self.time_since_last_phase_change:float = 0

    @property
    def time_to_observe(self)->bool:
        """Returns True if the traffic signal should act in the current step."""
        if self.obs_every_delta_time:
            return self.next_action_time <= self.env.sim_step
        else:
            return self.time_since_last_phase_change > self.yellow_time + self.min_green

    def update(self):
        """Updates the traffic signal state.

        Is called on every step of the simulation.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self._tick_controller_step()
        self.time_since_last_phase_change += self.env.step_length
        
        if self.next_phase_is_red and self.time_since_last_phase_change >= self.yellow_time:
            #if the next phase has to be red and the yellow time is over, set the light to red
            all_red_phase = "".join(["r" for _ in range(self.phase_len)])
            self.sumo.trafficlight.setRedYellowGreenState(self.id, all_red_phase)
            self.is_red             = True
            self.is_yellow          = False
            self.next_phase_is_red  = False
            self.time_since_last_phase_change = 0

        if self.is_red and self.time_since_last_phase_change >= self.env.all_red_phase_length:
            #if it is red and the red time is over, set the light to yellow
            yellow_phase = "".join(["r" if c.lower() != "g" else "y" for c in self.all_phases[self.green_phase].state])
            self.sumo.trafficlight.setRedYellowGreenState(self.id, yellow_phase)
            self.is_red         = False
            self.is_yellow      = True
            self.is_red_yellow  = True
            self.time_since_last_phase_change = 0

        if self.is_red_yellow and self.time_since_last_phase_change >= self.env.yellow_after_red_phase_len:
            #if it is red yellow and the time is over, set the light to green
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_red_yellow = False
            self.is_yellow     = False

        if self.is_yellow and self.time_since_last_phase_change >= self.yellow_time:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

        #updates the min green time depending of the current phase - minor phases have a shorter min green time
        if self.env.minor_phase_list is not None:
            self.min_green = self.env.min_green if self.green_phase not in self.env.minor_phase_list else self.env.minor_phase_min_length

        # Green-hold accounting for stability metrics and max-green enforcement diagnostics.
        if not self.is_yellow and not self.is_red and not self.is_red_yellow and not self.next_phase_is_red:
            self._tick_green_hold(step_seconds=self.env.step_length)

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Is called whenever time_to_observe is True.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        Variables:
            new_phase: int
                The new phase to be set.
            green_phase: int
                The current phase of the traffic signal.
        """
        new_phase = int(new_phase)
        requested_phase = int(new_phase)
        self._register_action_call()
        self.last_action_infeasible = False

        forced_by_max_green = False
        if (
            not self.is_yellow
            and not self.is_red
            and not self.is_red_yellow
            and new_phase == self.green_phase
            and self._max_green_reached()
        ):
            forced_phase = self._next_distinct_phase(self.green_phase)
            if forced_phase != self.green_phase:
                new_phase = forced_phase
                forced_by_max_green = True
                self._register_forced_max_green()

        if self.is_red:
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            pass
        elif (self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green):
            if (
                requested_phase != self.green_phase
                and not forced_by_max_green
                and self.time_since_last_phase_change < self.yellow_time + self.min_green
                and not self.is_yellow
            ):
                self._register_infeasible_action()
            #if its the same phase or it is not yet time to change the phase - do nothing but update the next action time
            self.next_action_time = self.env.sim_step + self.delta_time
        elif self.is_yellow:
            #if the phase is yellow and it is time to change the phase do nothing
            pass
        else:
            self._finalize_green_hold()
            self._register_phase_change()
            #else update the phase
            #if the current phase is a composed phase, the new phase will be the second part of the composed phase instead of the selcted action
            if isinstance(self.env.composed_phase_dict,dict) and self.green_phase in self.env.composed_phase_dict.keys():
                new_phase = int(self.env.composed_phase_dict[self.green_phase])

            yellow_phase = self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state

            if self.red_between_phase_groups_dict is not None:
                if self.red_between_phase_groups_dict[self.green_phase] != self.red_between_phase_groups_dict[new_phase]:
                    self.next_phase_is_red = True
                    yellow_phase = "".join(["r" if c.lower() != "g" else "y" for c in self.sumo.trafficlight.getRedYellowGreenState(self.id)])

            #set the new yellow phase
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, yellow_phase
            )
            self.green_phase    = new_phase
            self.current_phase  = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def can_change_phase(self) -> Literal[0, 1]:
        """Returns 1 if the traffic signal can change its phase.

        This is used for the observation space. to indicate to the agent that he can change the phase.
        """
        return [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1][0] # type: ignore


class RuleBasedTSPPhaseController(BasicPhaseController):
    """
    Simple rule-based bus TSP baseline that overlays BasicPhaseController.

    This controller keeps the same safety/timing behavior as BasicPhaseController
    and only overrides the requested action when a bus-priority rule triggers.
    """

    def __init__(
        self,
        ts: TrafficSignal,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
    ) -> None:
        super().__init__(
            ts=ts,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
        )

        default_detection_range = min(float(getattr(self.ts, "state_range", 120.0)), 120.0)
        detection_range_raw = getattr(self.env, "tsp_detection_range_m", default_detection_range)
        detection_range = default_detection_range if detection_range_raw is None else detection_range_raw
        self.tsp_detection_range_m: float = max(
            0.0, float(detection_range)
        )
        self.tsp_max_eta_seconds: float = max(
            0.0, float(getattr(self.env, "tsp_max_eta_seconds", 25.0))
        )
        self.tsp_min_speed_mps: float = max(
            0.1, float(getattr(self.env, "tsp_min_speed_mps", 2.0))
        )
        self.tsp_min_delay_seconds: float = float(
            getattr(self.env, "tsp_min_delay_seconds", 0.0)
        )
        self.tsp_enable_early_green: bool = bool(
            getattr(self.env, "tsp_enable_early_green", False)
        )
        self.tsp_enable_green_extension: bool = bool(
            getattr(self.env, "tsp_enable_green_extension", True)
        )
        self.tsp_early_green_eta_seconds: float = max(
            0.0, float(getattr(self.env, "tsp_early_green_eta_seconds", 18.0))
        )
        self.tsp_force_switch_remaining_seconds: float = max(
            0.5, float(getattr(self.env, "tsp_force_switch_remaining_seconds", 2.0))
        )
        self.tsp_green_extension_seconds: float = max(
            float(self.delta_time),
            float(getattr(self.env, "tsp_green_extension_seconds", float(self.delta_time) + 4.0)),
        )
        self.tsp_skip_buses_at_stops: bool = bool(
            getattr(self.env, "tsp_skip_buses_at_stops", True)
        )
        self.tsp_min_bus_speed_for_priority_mps: float = max(
            0.0, float(getattr(self.env, "tsp_min_bus_speed_for_priority_mps", 0.5))
        )
        override_gap_raw = getattr(self.env, "tsp_min_override_gap_seconds", None)
        override_gap_default = float(self.delta_time)
        if override_gap_raw is None:
            self.tsp_min_override_gap_seconds: float = override_gap_default
        else:
            self.tsp_min_override_gap_seconds = max(
                float(self.env.step_length), float(override_gap_raw)
            )
        self._last_tsp_override_step: float = float("-inf")

        self.tsp_override_count: int = 0
        self.tsp_hold_count: int = 0

        self._tls_index_to_green_phases = self._build_tls_index_to_green_phases()

    def _build_tls_index_to_green_phases(self) -> dict[int, list[int]]:
        if len(self.green_phases) == 0:
            return {}

        phase_state_len = len(self.green_phases[0].state)
        tls_idx_to_phases: dict[int, list[int]] = {}
        for tls_index in range(phase_state_len):
            phases_for_tls_idx = [
                phase_idx
                for phase_idx, phase in enumerate(self.green_phases)
                if tls_index < len(phase.state) and phase.state[tls_index].lower() == "g"
            ]
            if phases_for_tls_idx:
                tls_idx_to_phases[tls_index] = phases_for_tls_idx
        return tls_idx_to_phases

    def _phase_hops(self, source_phase: int, target_phase: int) -> int:
        num_green = max(1, int(self.num_green_phases))
        return int((int(target_phase) - int(source_phase)) % num_green)

    def _can_request_phase_change_now(self) -> bool:
        if self.is_red:
            return True
        if self.is_yellow or self.is_red_yellow or self.next_phase_is_red:
            return False
        return bool(self.time_since_last_phase_change >= self.yellow_time + self.min_green)

    def _iter_bus_candidates(self):
        buses_in_state = getattr(self.ts, "buses_in_state", None)
        if not isinstance(buses_in_state, dict) or len(buses_in_state) == 0:
            return

        for bus_state in buses_in_state.values():
            if not isinstance(bus_state, tuple) or len(bus_state) < 4:
                continue

            try:
                distance = float(bus_state[0])
                speed = float(bus_state[1])
            except (TypeError, ValueError):
                continue

            if distance < 0.0 or distance > self.tsp_detection_range_m:
                continue

            bus_data = bus_state[3]
            if self.tsp_skip_buses_at_stops and bool(
                getattr(bus_data, "at_bus_stop", False)
            ):
                continue
            if abs(speed) < self.tsp_min_bus_speed_for_priority_mps:
                continue
            try:
                delay = float(getattr(bus_data, "delay", 0.0))
            except (TypeError, ValueError):
                delay = 0.0

            if delay < self.tsp_min_delay_seconds:
                continue

            tls_index = getattr(bus_data, "tlsIndex", None)
            if tls_index is None:
                continue

            try:
                tls_index_int = int(tls_index)
            except (TypeError, ValueError):
                continue

            target_phases = self._tls_index_to_green_phases.get(tls_index_int)
            if not target_phases:
                continue

            eta_seconds = distance / max(abs(speed), self.tsp_min_speed_mps)
            if (
                eta_seconds > self.tsp_max_eta_seconds
                and distance > 0.5 * self.tsp_detection_range_m
            ):
                continue

            yield {
                "distance": distance,
                "speed": speed,
                "delay": delay,
                "eta_seconds": eta_seconds,
                "tls_index": tls_index_int,
                "target_phases": target_phases,
            }

    def _select_priority_bus(self):
        best_candidate = None
        best_score: tuple[float, float, float] | None = None
        for candidate in self._iter_bus_candidates():
            score = (
                float(candidate["eta_seconds"]),
                float(candidate["distance"]),
                -float(candidate["delay"]),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    @staticmethod
    def _is_green_state(phase_state: str) -> bool:
        s = phase_state.lower()
        return ("y" not in s) and any(ch == "g" for ch in s)

    def _get_tsp_priority_phase(self) -> int | None:
        candidate = self._select_priority_bus()
        if candidate is None:
            return None

        target_phases = list(candidate["target_phases"])
        if self.green_phase in target_phases:
            return int(self.green_phase)

        return int(
            min(
                target_phases,
                key=lambda phase: (self._phase_hops(self.green_phase, phase), int(phase)),
            )
        )

    def _apply_fixed_tsp_overlay(self) -> None:
        candidate = self._select_priority_bus()
        if candidate is None:
            return

        current_state = str(self.sumo.trafficlight.getRedYellowGreenState(self.id))
        if not self._is_green_state(current_state):
            return

        tls_index = int(candidate["tls_index"])
        if tls_index >= len(current_state):
            return

        remaining_seconds = max(
            0.0,
            float(self.sumo.trafficlight.getNextSwitch(self.id)) - float(self.env.sim_step),
        )
        if (
            float(self.env.sim_step) - self._last_tsp_override_step
            < self.tsp_min_override_gap_seconds - 1e-9
        ):
            return
        serves_now = current_state[tls_index].lower() == "g"

        if serves_now and self.tsp_enable_green_extension:
            if (
                float(candidate["eta_seconds"]) <= self.tsp_max_eta_seconds
                and remaining_seconds + 1e-9 < self.tsp_green_extension_seconds
            ):
                self.sumo.trafficlight.setPhaseDuration(
                    self.id, float(self.tsp_green_extension_seconds)
                )
                self.tsp_override_count += 1
                self._last_tsp_override_step = float(self.env.sim_step)
            return

        if not self.tsp_enable_early_green:
            return

        if (
            float(candidate["eta_seconds"]) <= self.tsp_early_green_eta_seconds
            and remaining_seconds > self.tsp_force_switch_remaining_seconds + 1e-9
        ):
            self.sumo.trafficlight.setPhaseDuration(
                self.id, float(self.tsp_force_switch_remaining_seconds)
            )
            self.tsp_override_count += 1
            self._last_tsp_override_step = float(self.env.sim_step)

    def update(self):
        if self.env.fixed_ts:
            self._tick_controller_step()
            self._apply_fixed_tsp_overlay()
            return
        super().update()

    def set_next_phase(self, new_phase: int):
        resolved_phase = int(new_phase)
        tsp_phase = self._get_tsp_priority_phase()

        if tsp_phase is not None:
            if tsp_phase == self.green_phase:
                resolved_phase = int(self.green_phase)
                self.tsp_hold_count += 1
            elif self._can_request_phase_change_now():
                resolved_phase = int(tsp_phase)
                self.tsp_override_count += 1
            else:
                resolved_phase = int(self.green_phase)
                self.tsp_hold_count += 1

        super().set_next_phase(resolved_phase)

    def get_episode_stability_metrics(self) -> dict[str, float]:
        metrics = super().get_episode_stability_metrics()
        metrics.update(
            {
                "tsp_override_count": float(self.tsp_override_count),
                "tsp_hold_count": float(self.tsp_hold_count),
                "tsp_detection_range_m": float(self.tsp_detection_range_m),
                "tsp_max_eta_seconds": float(self.tsp_max_eta_seconds),
                "tsp_early_green_eta_seconds": float(self.tsp_early_green_eta_seconds),
                "tsp_green_extension_seconds": float(self.tsp_green_extension_seconds),
                "tsp_min_override_gap_seconds": float(self.tsp_min_override_gap_seconds),
                "tsp_min_bus_speed_for_priority_mps": float(self.tsp_min_bus_speed_for_priority_mps),
            }
        )
        return metrics
