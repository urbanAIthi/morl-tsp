# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import traci

from morl_tsp.util.utils import check_common

if TYPE_CHECKING:
    from morl_tsp.environment.env import SumoEnvironment


#currently only support single agent learning
class Bus:
    
    def __init__(self,
                 env,
                 bus_id :str,
                 log:bool = True,
                 dwell_times: Literal["Fixed", "default", None] = "Fixed",
                 ) -> None:
        
        assert dwell_times in ["Fixed", "default", None], "dwell_times must be one of ['Fixed', 'default', None]"

        self.env                    = env
        self.id         :str        = bus_id
        self.log        :bool       = log
        self.line       :str|None   = self._get_line()
        self.tl_id      :str        = self._get_rl_tl_id()
        self.crossing_lane :str     = env.traffic_signals[self.tl_id].intersection_mapper.get_in_lane_for_vehicle(bus_id)
        self.timetable  :dict[str,float]    = self._get_timetable()
        self.occupancy  :int        = self._get_occupancy()
        self.max_speed  :float      = env.sumo.vehicle.getMaxSpeed(self.id)
        self.speed_factor:float     = env.sumo.vehicle.getSpeedFactor(self.id)
        self.done       :bool       = False
        if log:
            self.history = []

        self.stops      :list[tuple]        = env.sumo.vehicle.getStops(bus_id)
        self.o_stops    :list[tuple]        = self.stops
        self.delay_dict :dict[str,float]    = {stop_lane:0.0 for stop_lane in self.timetable} #maps the stop id to the delay # type: ignore
        self.delay      :float  = self._get_schedule_devation() #this default value is important if the bus does not spawn at a bus stop
        self._set_stops()

        self.update()


    def __call__(self)->dict[str,float |int | str]:
        return self.info_dict

    '''
    Get info about the Bus
    ''' 
    def _get_occupancy(self)->int:
        '''returns the occupancy of the bus'''
        raw = self.env.sumo.vehicle.getParameter(self.id, "occupancy")
        try:
            if raw in ("", None):
                raise ValueError("empty occupancy")
            return int(float(raw))
        except (TypeError, ValueError):
            return int(getattr(self.env, "bus_default_occupancy", 10))
    
    def _get_schedule_devation(self)->float:
        '''returns the schedule deviation of the bus'''
        raw = self.env.sumo.vehicle.getParameter(self.id, "scheduleDeviation")
        try:
            if raw in ("", None):
                raise ValueError("empty schedule deviation")
            return float(raw)
        except (TypeError, ValueError):
            return 0.0


    def _get_line(self)->str|None:
        line_tokens = [s for s in self.id.split(":") if len(s) == 2 and s.isdigit()]
        if len(line_tokens) == 1:
            return line_tokens[0]
        elif len(line_tokens) > 1:
            raise ValueError(f"Bus ID {self.id} contains multiple 2 digit numbers that are not separated by ':'")
        else:
            return None
        
    def _get_rl_tl_id(self)->str:
        '''get the id of the tl that is controlled with rl'''
        def _check_common(list1, list2)->list: #returns the copmmon entries from to lists as a list
            return list(set(list1) & set(list2))

        common_rl_tl = _check_common(self.env.traffic_signals.keys(),
                                    [tlsID for (tlsID,_,_,_) in self.env.sumo.vehicle.getNextTLS(self.id)])
        if len(common_rl_tl) != 1:
            raise ValueError(f"The Bus class currently only supports a single RL TL, but Bus {self.id} has multiple TL: {common_rl_tl}")
        return common_rl_tl[0]
    
    def _get_timetable(self)->dict[str,float]: # type: ignore
        return {x.stoppingPlaceID: x.until if hasattr(x,"until") and x.until > 0 else self.env.sumo.simulation.getTime() for x in self.env.sumo.vehicle.getStops(self.id)} 
        
    '''
    Bus info related functions
    '''
    def update(self)->None:
        '''updates the information of the bus if the bus is still within the simulation'''
        def get_next_tls()->list[tuple[str,int,float,str]]:
            '''returns a list of tuples that include information about all of the tl that are going to be passed'''
            #This exception is raised if the bus is not in the simulation anymore
            try:
                return [(tlsID,tlsIndex,distance,state) for (tlsID,tlsIndex,distance,state) in self.env.sumo.vehicle.getNextTLS(self.id) if tlsID == self.tl_id]
            except traci.exceptions.TraCIException:
                return []
        
        self.next_tls = get_next_tls()
        
        if self.next_tls:
            #set only during in the initial update
            if not hasattr(self,"tlsIndex"): 
                self.tlsIndex :int          = self._get_tls_index()

            self.speed      :float          = self.env.sumo.vehicle.getSpeed(self.id)
            self.distance   :float          = self._get_distance_to_stop_line()
            self.stops      :list[tuple]    = self.env.sumo.vehicle.getStops(self.id)
            self.stop       :str | None     = self.stops[0].stoppingPlaceID if len(self.stops) > 0 else None    #next bus stop  # type: ignore
            self.dwell_time :float           = self.stops[0].duration        if len(self.stops) > 0 else 0.0    # the dwell time of the next stop ticks down while the bus is at the stop # type: ignore
            self.at_bus_stop:bool           = bool(self.env.sumo.vehicle.getStopState(self.id) // 16)           # True if the bus is currently stopped at a bus stop otherwise its false 
            if self.timetable and self.stops:
                self._update_delay() #updates self.delay: float and self.delay_dict: dict
            if self.log:
                self.history.append(self.info)
        else:
            self.done = True

    def _get_distance_to_stop_line(self)->float:
        '''retuns the distance from the bus to the stop line of the rl controlled tl'''
        return [distance for (_,_,distance,_) in self.next_tls][0]
    
    def _get_tls_index(self)->int:
        '''returns the tls index which indicates through which path the vehicle is going to pass the intersection'''
        return [tlsIndex for (_,tlsIndex,_,_) in self.next_tls][0]

    @property
    def info(self)->tuple[float, int, float, float, int]:
        return self.distance,self.tlsIndex,self.delay,self.speed,self.occupancy
    
    @property
    def info_dict(self)->dict[str,float |int | str]:
        info_dict =  {"distance"        : self.distance,
                      "tlsIndex"        : self.tlsIndex,
                      "crossing_lane"   : self.crossing_lane,
                      "delay"           : self.delay,
                      "speed"           : self.speed,
                      "occupancy"       : self.occupancy,
                      "dwell_time"      : self.dwell_time}
        return info_dict

    def _update_delay(self)->None: #-> delay: float, delay_dict: dict
        '''updates the current delay and the delay dict if the bus just arrived at an intersection'''
        arrival = getattr(self.stops[0],"arrival",None)
        if arrival is not None and arrival > 0:
            if self.delay_dict[self.stop] is None: # type: ignore
                self.delay_dict[self.stop] = arrival - self.timetable[self.stop] # type: ignore
                self.delay = self.delay_dict[self.stop] # type: ignore
    
    '''
    Set dwell times related functions
    '''
    def _set_stops(self)->None:
        
        next_stops = self.env.sumo.vehicle.getStops(self.id)

        dwell_times = None

        if self.env.dwell_times is None:
            dwell_times = [(x.stoppingPlaceID,10) for x in next_stops]

        elif self.env.dwell_times == "Fixed":
            dwell_times = [(x.stoppingPlaceID,15) for x in next_stops]

        elif self.env.dwell_times == "default":
            return None

        if dwell_times is not None:
            for bus_stop, duration in dwell_times:
                try:
                    self.env.sumo.vehicle.setBusStop(vehID = self.id,
                                                     stopID = bus_stop,
                                                     duration = duration,
                                                     until = [s for s in self.o_stops if s.stoppingPlaceID == bus_stop][0].until) # type: ignore
                except Exception as e:
                        logging.debug(f"Could not set dwell time for bus {self.id} at stop {bus_stop}: {e}")
@dataclass
class BusStore(MutableMapping[str, Bus]):
    env: object
    bus_factory: None | Callable[[str], Bus] = None

    _active: dict[str, Bus] = field(default_factory=dict, init=False, repr=False)
    _inactive: dict[str, Bus] = field(default_factory=dict, init=False, repr=False)
    
    processed_buses: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.bus_factory is None:
            self.bus_factory = lambda bus_id: Bus(self.env, bus_id)  # type: ignore[name-defined]

    # --- Mapping interface (ACTIVE only) ---
    def __getitem__(self, key: str) -> Bus:
        return self._active[key]

    def __setitem__(self, key: str, value: Bus) -> None:
        self._active[key] = value
        self._inactive.pop(key, None)

    def __delitem__(self, key: str) -> None:
        del self._active[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._active)

    def __contains__(self, key: object) -> bool:
        return key in self._active

    def __len__(self) -> int:
        return len(self._active)

    def __repr__(self) -> str:
        return f"BusStore(n_active={len(self._active)}, n_inactive={len(self._inactive)}, active={list(self._active.keys())})"

    def get_or_create(self, key: str) -> Bus:
        """Return an ACTIVE bus. Reactivate if needed. Create if missing."""
        if key in self._active:
            return self._active[key]
        if key in self._inactive:
            self._active[key] = self._inactive.pop(key)
            return self._active[key]
        bus = self.bus_factory(key)  # type: ignore[operator]
        self._active[key] = bus
        return bus

    def deactivate(self, key: str, *, strict: bool = False) -> None | Bus:
        """Move ACTIVE -> INACTIVE."""
        bus = self._active.pop(key, None)
        if bus is None:
            if strict:
                raise KeyError(key)
            return None
        self._inactive[key] = bus
        return bus

    @property
    def inactive_count(self) -> int:
        return len(self._inactive)

class BusDataManager:
    """
    Manages retrieval and updating of bus information from the active SUMO simulation.
    """

    def __init__(self, env):
        """
        Initializes the BusDataManager with a reference to the parent environment.

        The env object is expected to have the following attributes:
        -------------------------------------------------------------------------
        sumo : object
            The SUMO (simulation) interface or controller (e.g., traci).
        vehicleIDlist : list[str]
            list of all current vehicle IDs in the simulation.
        traffic_signals : dict
            dictionary of traffic signals in your simulation environment.
        buses : object
            Custom container/object with methods like get_keys(), add_item(), move_to_inactive().
        processed_buses : list[str]
            list of bus IDs that have already been processed or are not relevant.
        _check_common : Callable
            A function that checks some “commonality” between sets of TLS IDs 
            (akin to your original `_check_common` method).
        -------------------------------------------------------------------------
        """
        self.env = env

    @property
    def buses(self) -> BusStore:
        return self.env.buses

    def _get_new_buses(self) -> list[str]:
        """
        Find buses in 'vehicleIDlist' that match the given prefix and
        are not already processed, but do drive through the RL TL.

        Returns
        -------
        list[str]
            A list of new bus IDs that meet the criteria.
        """
        bus_id_list = []

        seen_buses = self.buses.processed_buses | set(self.buses.keys())

        # For each bus-like ID not yet processed, check if it drives through known traffic signals
        for bus_id in [vid for vid in self.env.vehicleIDList if self.env.bus_vid_prefix.lower() in vid.lower() and vid not in seen_buses]:
            # If there's a common TLS among traffic_signals and next_tls, log the bus
            if check_common(self.env.traffic_signals.keys(), [tlsID for (tlsID, _, _, _) in self.env.sumo.vehicle.getNextTLS(bus_id)]):
                bus_id_list.append(bus_id)
                self.buses.processed_buses.add(bus_id)

        return bus_id_list

    def update_bus_dict(self):
        """
        Updates the internal bus dictionary:
        - Finds new buses.
        - Adds them to the 'active' set if not already present.
        - Updates each active bus (calls 'bus.update()').
        - Moves any 'done' buses to the 'inactive' set.
        """
        # Detect newly discovered buses
        bus_id_list = self._get_new_buses()

        # Add new entries
        for bus_id in bus_id_list:
            if bus_id not in self.buses:
                self.buses.get_or_create(bus_id)

        # Keep track of buses that should be removed
        drop_list = []
        for bus in self.buses.values():
            bus.update()
            if bus.done:
                drop_list.append(bus.id)

        # Move dropped buses from active to inactive
        for bus_id in drop_list:
            self.buses.deactivate(bus_id)
