# SPDX-FileCopyrightText: 2026 Philip-Roman Adam
# SPDX-License-Identifier: MIT AND AGPL-3.0-or-later

import logging
import math

from morl_tsp.config import MAX_NUM_LANES

logger = logging.getLogger(__name__)

class IntersectionMapper:
    '''
    This class takes a sumo connection and a traffic signal id and computes the following:
    1. Fixed ordered lanes (12 positions) based on canonical direction and lane index.
    2. Unique incoming edge IDs (tls_in_edges) from the fixed ordered lanes.
    3. Mapping of each tls_in_edge to the set of possible out_edge IDs based on lane connection information.
    4. Mapping of (in_edge, out_edge) to the list of lanes from the fixed_order_lanes that provide that connection.
    5. Mapping of (in_edge, out_edge) to the best lane (the one with the least connections).
    6. Given a vehicle id, determines which lane on the current ("from") edge provides a valid connection
         to the next ("to") edge in the vehicle's route.
    '''

    def __init__(self, 
                 tls_id, 
                 sumo):
        """
        tls_id: Traffic light system id.
        sumo: Reference to the TraCI SUMO simulation instance.
        """
        self.tls_id = tls_id
        self.sumo = sumo
        # Get unique controlled lanes.
        lanes:list[str] = list(set(self.sumo.trafficlight.getControlledLanes(self.tls_id)))
        self.fixed_order_lanes, self.lane_angles, self.canonical_mapping = self._get_ordered_lanes(lanes)
        self.tls_in_edges = self._compute_tls_in_edges()
        self.tls_out_edges = self._compute_tls_out_edges()
        # Compute mapping of (in_edge, out_edge) -> lane(s) from the fixed order list.
        self.in_out_lane_mapping = self._compute_in_out_lane_mapping()
        # Additional dict: map (in_edge, out_edge) -> best lane (with least connections)
        self.best_lane_mapping = self._compute_best_lane_mapping()
        self._missing_mapping_warned: set[tuple[str, str]] = set()

    @staticmethod
    def _lane_to_edge_id(lane_id: str) -> str:
        """
        Convert lane id to edge id while preserving underscores in edge names.
        SUMO lane ids are typically '<edge_id>_<lane_index>'.
        """
        edge_id, sep, lane_idx = lane_id.rpartition("_")
        if sep and lane_idx.isdigit():
            return edge_id
        return lane_id

    @staticmethod
    def _edge_base(edge_id: str) -> str:
        """
        Normalize edge ids for matching routes vs lane links when '#segment'
        suffixes differ across sources.
        """
        return edge_id.split("#", 1)[0]
    
    def _is_vehicle_lane(self, 
                         lane_id:str) -> bool:
        """
        Returns True if the lane is intended for vehicles.
        Assumes that sumo.lane.getAllowed(lane_id) returns a list of allowed vehicle types.
        Exception: If the vehicle_type is not defined in any .xml files, then simply assume that the lane is a vehicle lane.
        """
        allowed: tuple = self.sumo.lane.getAllowed(lane_id)
        vehicle_types = {"passenger", "car", "truck", "bus"}

        #exception: if there are no allowed vehicle types, assume its a vehicle lane. this is workaround only for intersectionzoo data. check for other cases as per need.
        if not allowed:
            return True

        return any(vtype in allowed for vtype in vehicle_types)
    
    def _get_ordered_lanes(self, 
                          lanes:list) -> tuple[list[str],dict[str,float],dict[str,float]]:
        """
        1. Filter out lanes not used by vehicles.
        2. Limit to at most 12 lanes.
        3. Compute each lane’s angle from its shape (using the first two points).
        4. Map each lane’s angle to the nearest canonical direction (0°, 90°, 180°, 270°).
        5. Create a fixed-order list of 12 positions (filling with None if needed) where the order is:
           - Sorted first by canonical direction and then by the numeric lane index.
        """
        # Filter out non-vehicle lanes.
        vehicle_lanes = [lane for lane in lanes if self._is_vehicle_lane(lane)]
        # Limit to at most 12 lanes.
        vehicle_lanes = vehicle_lanes[:MAX_NUM_LANES]
    
        # Compute angles for each lane.
        lane_angles = {}
        for lane in vehicle_lanes:
            shape = self.sumo.lane.getShape(lane)
            if len(shape) >= 2:
                (x1, y1), (x2, y2) = shape[0], shape[1]
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360
                lane_angles[lane] = angle
            else:
                lane_angles[lane] = None
    
        # Map each lane’s angle to the nearest canonical direction.
        canonical_dirs = [0, 90, 180, 270]
        canonical_mapping = {}
        for lane, angle in lane_angles.items():
            if angle is not None:
                best_dir = min(canonical_dirs, key=lambda d: min(abs(angle - d), 360 - abs(angle - d)))
                canonical_mapping[lane] = best_dir
            else:
                canonical_mapping[lane] = None
    
        # Sort lanes: first by canonical direction, then by lane index (assumed to be the trailing number in "edgeID_index").
        def lane_sort_key(lane):
            canonical = canonical_mapping[lane]
            try:
                lane_index = int(lane.split('_')[-1])
            except Exception:
                lane_index = 0
            return (canonical, lane_index)
    
        ordered_lanes:list[str] = sorted(vehicle_lanes, key=lane_sort_key)
        # Create a fixed list with 12 positions (fill missing slots with "empty_placeholder").
        fixed_list:list[str] = [f"empty_placeholder_{i}" for i in range(MAX_NUM_LANES)]
        for i, lane in enumerate(ordered_lanes):
            fixed_list[i] = lane
    
        return fixed_list, lane_angles, canonical_mapping
    
    def _compute_tls_in_edges(self)->list[str]:
        """
        Computes and returns a list of unique incoming edge IDs (tls_in_edges) from the fixed_order_lanes.
        Assumes each lane ID is formatted as "edgeID_index".
        """
        in_edges = list({
            self._lane_to_edge_id(lane)
            for lane in self.fixed_order_lanes
            if "empty_placeholder" not in lane
        })
        assert in_edges, "No vehicle lanes found in fixed_order_lanes. Check _get_ordered_lanes() logic."
        return in_edges
    
    def _compute_tls_out_edges(self)->dict[str,list[str]]:
        """
        Computes a dictionary mapping each tls_in_edge (from the controlled lanes) to the set of
        possible out_edge IDs based on lane connection information.
        """
        out_edges_mapping = {}
        for in_edge in self.tls_in_edges:
            lanes_on_in_edge = [lane for lane in self.fixed_order_lanes 
                                 if "empty_placeholder" not in lane and self._lane_to_edge_id(lane) == in_edge]
            possible_out_edges = set()
            for lane in lanes_on_in_edge:
                connections = self.sumo.lane.getLinks(lane)
                for conn in connections:
                    if isinstance(conn, dict):
                        dest_lane = conn.get('to')
                    else:
                        dest_lane = conn[0] if len(conn) > 0 else None
                    if dest_lane:
                        out_edge = self._lane_to_edge_id(dest_lane)
                        possible_out_edges.add(out_edge)
            out_edges_mapping[in_edge] = list(possible_out_edges)
        return out_edges_mapping
    
    def _compute_in_out_lane_mapping(self)->dict[tuple[str,str],list[str]]:
        """
        Computes a dictionary where each key is a tuple (in_edge, out_edge) and the value is a list
        of lanes (from the fixed_order_lanes) that allow a vehicle to move from the in_edge to the out_edge.
        """
        mapping = {}
        for lane in self.fixed_order_lanes:
            if "empty_placeholder" in lane:
                continue
            in_edge = self._lane_to_edge_id(lane)
            connections = self.sumo.lane.getLinks(lane)
            for conn in connections:
                if isinstance(conn, dict):
                    dest_lane = conn.get('to')
                else:
                    dest_lane = conn[0] if len(conn) > 0 else None
                if dest_lane is None:
                    continue
                out_edge = self._lane_to_edge_id(dest_lane)
                key = (in_edge, out_edge)
                if key not in mapping:
                    mapping[key] = []
                mapping[key].append(lane)
        return mapping
    
    def _compute_best_lane_mapping(self)->dict[tuple[str,str],str]:
        """
        Computes an additional dictionary mapping each (in_edge, out_edge) tuple to a single lane.
        The selected lane is the one with the fewest number of connections (as determined by len(self.sumo.lane.getLinks(lane))).
        """
        best_mapping = {}
        for key, lanes in self.in_out_lane_mapping.items():
            best_lane = min(lanes, key=lambda lane: len(self.sumo.lane.getLinks(lane)))
            best_mapping[key] = best_lane
        return best_mapping

    def _get_best_lane_for_pair(self, in_edge: str, out_edge: str) -> str | None:
        """
        Resolve the best lane for an (in_edge, out_edge) pair, allowing fuzzy
        matching when route edges and lane-link edges differ by '#'-suffixes.
        """
        best_lane = self.best_lane_mapping.get((in_edge, out_edge))
        if best_lane is not None:
            return best_lane

        in_base = self._edge_base(in_edge)
        out_base = self._edge_base(out_edge)
        for (mapped_in, mapped_out), lane in self.best_lane_mapping.items():
            if self._edge_base(mapped_in) == in_base and self._edge_base(mapped_out) == out_base:
                return lane
        return None

    def _lane_connects_to_out_edge(self, lane_id: str, out_edge: str) -> bool:
        """
        Returns True if a lane has a direct connection to the requested out-edge.
        Uses base edge ids so '#segment' suffixes do not break matching.
        """
        out_base = self._edge_base(out_edge)
        for conn in self.sumo.lane.getLinks(lane_id):
            if isinstance(conn, dict):
                dest_lane = conn.get('to')
            else:
                dest_lane = conn[0] if len(conn) > 0 else None
            if not dest_lane:
                continue
            dest_edge = self._lane_to_edge_id(dest_lane)
            if self._edge_base(dest_edge) == out_base:
                return True
        return False

    def _get_fallback_lane(self, vehicle_id: str, in_edge: str | None = None) -> str:
        """
        Pick a robust fallback lane instead of raising. This keeps training
        running on networks where not all turn relations are represented in the
        reduced lane set used for observations.
        """
        current_lane = ""
        try:
            current_lane = self.sumo.vehicle.getLaneID(vehicle_id)
        except Exception:
            current_lane = ""

        if in_edge is None and current_lane:
            in_edge = self._lane_to_edge_id(current_lane)

        if in_edge is not None:
            candidates = [
                lane for lane in self.fixed_order_lanes
                if "empty_placeholder" not in lane and self._lane_to_edge_id(lane) == in_edge
            ]
            if candidates:
                if current_lane in candidates:
                    return current_lane
                return min(candidates, key=lambda lane: len(self.sumo.lane.getLinks(lane)))

        if current_lane and "empty_placeholder" not in current_lane:
            return current_lane

        non_placeholder = [lane for lane in self.fixed_order_lanes if "empty_placeholder" not in lane]
        return non_placeholder[0] if non_placeholder else self.fixed_order_lanes[0]
    
    def get_in_lane_for_vehicle(self, 
                                vehicle_id:str)->str:
        """
        Given a vehicle id, determines which lane on the current ("from") edge provides a valid connection
        to the next ("to") edge in the vehicle's route.
        """
        route_edges: list[str] = list(self.sumo.vehicle.getRoute(vehicle_id))
        if len(route_edges) < 2:
            return self._get_fallback_lane(vehicle_id)

        in_edge_id = next((i for i, edge in enumerate(route_edges[:-1]) if edge in self.tls_in_edges), None)
        if in_edge_id is None:
            return self._get_fallback_lane(vehicle_id)

        in_edge = route_edges[in_edge_id]
        out_edge = route_edges[in_edge_id + 1]

        # Prefer the vehicle's real lane when it is already on the incoming
        # edge and supports the requested turn. This keeps per-lane metrics
        # faithful instead of collapsing all lanes to the "best" lane.
        try:
            current_lane = self.sumo.vehicle.getLaneID(vehicle_id)
        except Exception:
            current_lane = ""
        if current_lane and "empty_placeholder" not in current_lane:
            current_edge = self._lane_to_edge_id(current_lane)
            if (
                self._edge_base(current_edge) == self._edge_base(in_edge)
                and self._lane_connects_to_out_edge(current_lane, out_edge)
            ):
                return current_lane

        in_lane = self._get_best_lane_for_pair(in_edge, out_edge)
        if in_lane is not None:
            return in_lane

        missing_key = (in_edge, out_edge)
        if missing_key not in self._missing_mapping_warned:
            logger.warning(
                "No lane mapping for tls=%s edge pair (%s -> %s); using fallback lane.",
                self.tls_id,
                in_edge,
                out_edge,
            )
            self._missing_mapping_warned.add(missing_key)
        return self._get_fallback_lane(vehicle_id, in_edge=in_edge)
'''
# Example usage:
if __name__ == "__main__":
    tls_id = "3050"  # Replace with your actual traffic light id.
    mapper = IntersectionMapper(tls_id, traci)
    
    print("Fixed ordered lanes (12 positions):")
    for i, lane in enumerate(mapper.fixed_order_lanes):
        print(f"Slot {i}: {lane}")
    
    print("\nTLS In Edges:")
    print(mapper.tls_in_edges)
    
    print("\nTLS Out Edges mapping (in_edge -> possible out_edges):")
    for in_edge, out_edges in mapper.tls_out_edges.items():
        print(f"{in_edge} -> {out_edges}")
    
    print("\nIn-Out Lane Mapping ((in_edge, out_edge) -> lanes):")
    in_out_mapping = mapper.in_out_lane_mapping
    for key, lanes in in_out_mapping.items():
        print(f"{key}: {lanes}")
    
    print("\nBest Lane Mapping ((in_edge, out_edge) -> best lane):")
    best_mapping = mapper.best_lane_mapping
    for key, lane in best_mapping.items():
        print(f"{key}: {lane}")
    
    # Example for mapping a vehicle's route to its connection lane.
    vehicle_id = "63000_149"  # Replace with an actual vehicle id.
    lane_for_route = mapper.get_in_lane_for_vehicle(vehicle_id)
    if lane_for_route:
        print(f"\nVehicle {vehicle_id} uses lane {lane_for_route} for the connection.")
    else:
        print(f"\nCould not determine the lane for vehicle {vehicle_id}.")'
'''
