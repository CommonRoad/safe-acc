from typing import List, Dict, Tuple, Set
import numpy as np
import warnings

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType
from commonroad.geometry.shape import Shape
from commonroad.scenario.trajectory import State

from common.util_motion import safe_distance_profile_based
from common.vehicle import Vehicle, Maneuver, StateLongitudinal, StateLateral
from common.configuration import LaneCategory
from common.lane import Lane
from acc.safety_layer import SafetyLayer


class ObstacleSelection:
    """
    Class to extract dynamic obstacles in the field of view of the ego vehicle within the left,
    same, and right lane.
    """
    def __init__(self, scenario: Scenario, safety_layer: SafetyLayer, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, simulation_param: Dict, acc_param: Dict):
        """
        :param scenario: CommonRoad scenario
        :param safety_layer: safety layer object
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param simulation_param: dictionary with parameters of the simulation environment
        :param acc_param: dictionary with parameters of the acc related algorithms
        """
        self._scenario = scenario
        self._vehicle_dict = {}
        self._cut_out = {}
        self._safety_layer = safety_layer

        self._dt = simulation_param.get("dt")
        self._rel_obs_ids = simulation_param.get("other_vehicle_plots")
        self._plotting_profiles = simulation_param.get("plotting_profiles")

        self._fov = ego_vehicle_param.get("fov")
        self._a_max_ego = ego_vehicle_param.get("a_max")
        self._v_min_ego = ego_vehicle_param.get("v_min")
        self._v_max_ego = ego_vehicle_param.get("v_max")
        self._a_corr_ego = ego_vehicle_param.get("a_corr")
        self._t_react = ego_vehicle_param.get("t_react")
        self._j_min_ego = ego_vehicle_param.get("j_min")
        self._j_max_ego = ego_vehicle_param.get("j_max")
        self._a_min_ego = ego_vehicle_param.get("a_min")

        self._th_theta = other_vehicles_param.get("th_theta")
        self._th_offset = other_vehicles_param.get("th_offset")
        self._n_cutin = other_vehicles_param.get("n_cutin")
        self._a_min_other = other_vehicles_param.get("a_min")
        self._j_min_other = other_vehicles_param.get("j_min")
        self._j_max_other = other_vehicles_param.get("j_max")
        self._v_min_other = other_vehicles_param.get("v_min")
        self._v_max_other = other_vehicles_param.get("v_max")

        self._const_dist_offset = acc_param.get("common").get("const_dist_offset")
        self._emergency_profile = acc_param.get("emergency").get("emergency_profile")
        self._time_leave = round(acc_param.get("common").get("time_leave") / self._dt)

    @property
    def vehicle_dict(self) -> Dict[int, Vehicle]:
        return self._vehicle_dict

    @vehicle_dict.setter
    def vehicle_dict(self, vehicle_dict: Dict[int, Vehicle]):
        self._vehicle_dict = vehicle_dict

    def update_vehicle_dict(self, time_step: int, ego_lane: Lane, left_lane: Lane, right_lane: Lane) -> List[int]:
        """
        Update vehicle dictionary with new obstacle information from current time step

        :param time_step: current time step
        :param ego_lane: lane of ego vehicle (merge of current lanelet with successor lanelets)
        :param left_lane: left lane of ego vehicle (merge of current left adjacent lanelet with its successor lanelets)
        :param right_lane: right lane of ego vehicle (merge of current right adjacent lanelet
        """
        ego_lane_veh_ids = self.extract_obstacles_on_lane(ego_lane, LaneCategory.SAME, time_step)
        right_lane_veh_ids = self.extract_obstacles_on_lane(right_lane, LaneCategory.RIGHT, time_step)
        left_lane_veh_ids = self.extract_obstacles_on_lane(left_lane, LaneCategory.LEFT, time_step)

        return left_lane_veh_ids, right_lane_veh_ids, ego_lane_veh_ids

    def extract_obstacles_on_lane(self, lane: Lane, lane_category: LaneCategory, time_step: int) -> List[int]:
        """
        Evaluates whether a obstacle is located on a lane.
        :param lane: lane for which the obstacles should be returned
        :param lane_category: category of the lane with respect to the ego lane
        :param time_step: current time step
        :return: a list of vehicle IDs which are located on the lane
        """
        veh_ids = []
        if lane is None:
            return veh_ids
        for l_id in lane.contained_lanelets:
            lanelet = self._scenario.lanelet_network.find_lanelet_by_id(l_id)
            if lanelet.dynamic_obstacles_on_lanelet.get(time_step) is not None:
                for obstacle_id in lanelet.dynamic_obstacles_on_lanelet[time_step]:
                    obstacle = self._scenario.obstacle_by_id(obstacle_id)
                    veh_ids.append(obstacle_id)
                    if time_step == self._scenario.obstacle_by_id(obstacle_id).initial_state.time_step:
                        state_cr = obstacle.initial_state
                        state_cr_prev = None
                    elif self._vehicle_dict.get(obstacle_id) is None or \
                            self._vehicle_dict.get(obstacle_id).states_cr.get(time_step - 1) is None:
                        state_cr = obstacle.prediction.trajectory.state_at_time_step(time_step)
                        state_cr_prev = None
                    else:
                        state_cr = obstacle.prediction.trajectory.state_at_time_step(time_step)
                        state_cr_prev = self._vehicle_dict[obstacle_id].states_cr[time_step - 1]
                    try:
                        state_lon, state_lat = lane.create_curvilinear_state(state_cr, state_cr_prev, self._dt)
                        self.add_vehicle_to_dict(lane_category, obstacle.obstacle_shape, obstacle_id,
                                                 obstacle.obstacle_type,
                                                 state_cr, state_lat, state_lon, {lanelet.lanelet_id}, time_step)
                    except ValueError:
                        warnings.warn('Vehicle at the border of the projection domain. Vehicle will be neglected.')
                        continue
        return veh_ids

    def add_vehicle_to_dict(self, lane_category: LaneCategory, shape: Shape, obstacle_id: int,
                            obstacle_type: ObstacleType, state_cr: State, state_lat: StateLateral,
                            state_lon: StateLongitudinal, lanelet: Set[int], time_step: int):
        """
        Update vehicle dictionary with new obstacle information from current time step

        :param lane_category: enum indicating left, same and right lane
        :param shape: obstacle shape
        :param obstacle_id: CommonRoad ID of obstacle
        :param obstacle_type: CommonRoad obstacle type
        :param state_cr: CommonRoad state
        :param state_lat: lateral state in curvilinear coordinate system
        :param state_lon: longitudinal state in curvilinear coordinate system
        :param lanelet: lanelet the vehicle is on
        :param time_step: time step to which vehicle should be added
        """
        if self._vehicle_dict.get(obstacle_id) is not None:
            self._vehicle_dict[obstacle_id].append_state_lon(state_lon, time_step)
            self._vehicle_dict[obstacle_id].append_state_lat(state_lat, time_step)
            self._vehicle_dict[obstacle_id].append_state_cr(state_cr, time_step)
            self._vehicle_dict[obstacle_id].append_lane_category(lane_category, time_step)
            self._vehicle_dict[obstacle_id].append_lanelet_assignment(lanelet, time_step)
        else:
            vehicle = Vehicle(state_lon, state_lat, shape, lane_category, state_cr, obstacle_id,
                              obstacle_type, set(self._scenario.lanelet_network.find_lanelet_by_position(
                                        [np.array([state_cr.position[0], state_cr.position[1]])])[0]))
            self._vehicle_dict[obstacle_id] = vehicle

    def cutin_prev(self, vehicle: Vehicle, time_step: int):
        """
        Evaluation whether a vehicle in an adjacent lane performs a cut-in already for n_cutin time steps

        :param time_step: current time step
        :param vehicle: vehicle in adjacent lane
        :return: Boolean indicating if vehicle performed cut-in for enough time steps
        """
        if time_step >= self._n_cutin:
            for idx in range(self._n_cutin - 1):
                if vehicle.maneuver_list.get(time_step - idx - 1) is not None \
                        and vehicle.maneuver_list[time_step - idx - 1] == Maneuver.CUTIN:
                    continue
                else:
                    return False
            return True
        else:
            return False

    def detect_cutin_adjacent(self, time_step: int, vehicle_ids_right: List[int], vehicle_ids_left: List[int],
                              vehicle_ids_same: List[int], ego_vehicle: Vehicle, emg_idx) -> \
            Tuple[List[int], List[int]]:
        """
        Detects cut-in vehicles in adjacent lanes and updates vehicle maneuver

        :param time_step: current time step
        :param vehicle_ids_right: vehicle IDs of vehicles in ego vehicle's right lane
        :param vehicle_ids_left: vehicle IDs of vehicles in ego vehicle's left lane
        :param vehicle_ids_same: vehicle IDs of vehicles in ego vehicle's lane
        :param ego_vehicle: ego vehicle object
        :param emg_idx: execution index of emergency maneuver
        :return: lists with IDs of vehicles in adjacent lane performing a cut-in and updated vehicle list for same lane
        """
        vehicles_adjacent = []
        ego_state_lon = ego_vehicle.states_lon[time_step]
        for idx, veh_id in enumerate(vehicle_ids_left):
            if self._vehicle_dict[veh_id].states_lat[time_step].theta > self._th_theta \
                    and self._vehicle_dict[veh_id].states_lat[time_step].d < -self._th_offset:
                self.handle_cutin_vehicles(ego_state_lon, ego_vehicle, emg_idx, time_step, veh_id, vehicle_ids_same,
                                           vehicles_adjacent)
        for idx, veh_id in enumerate(vehicle_ids_right):
            if self._vehicle_dict[veh_id].states_lat[time_step].theta < -self._th_theta \
                    and self._vehicle_dict[veh_id].states_lat[time_step].d > self._th_offset:
                self.handle_cutin_vehicles(ego_state_lon, ego_vehicle, emg_idx, time_step, veh_id, vehicle_ids_same,
                                           vehicles_adjacent)

        return vehicles_adjacent, vehicle_ids_same

    def handle_cutin_vehicles(self, ego_state_lon: StateLongitudinal, ego_vehicle: Vehicle, emg_idx: int,
                              time_step: int, veh_id: int, vehicle_ids_same: List[int], vehicles_adjacent: List[int]):
        """
        Processes cut-in vehicles whether they performed a cut-in in the previous time step or not

        :param ego_state_lon: longitudinal state of ego vehicle
        :param ego_vehicle: ego vehicle object
        :param emg_idx: execution index of emergency maneuver
        :param time_step: current time step
        :param veh_id: currently considered cutin vehicle
        :param vehicle_ids_same: vehicle IDs of vehicles in ego vehicle's lane
        :param vehicles_adjacent: vehicle IDs of vehicles in adjacent lanes of ego vehicle
        """
        self._vehicle_dict[veh_id].append_maneuver(Maneuver.CUTIN, time_step)
        if self.cutin_prev(self._vehicle_dict[veh_id], time_step):
            state_lon = self.vehicle_dict[veh_id].states_lon[time_step]
            safe_distance = safe_distance_profile_based(ego_vehicle.front_position(time_step), ego_state_lon.v,
                                                        ego_state_lon.a,
                                                        self.vehicle_dict[veh_id].rear_position(time_step),
                                                        state_lon.v, self._dt,
                                                        self._t_react, self._a_min_ego, self._a_max_ego,
                                                        self._j_max_ego, self._v_min_ego, self._v_max_ego,
                                                        self._a_min_other, self._v_min_other, self._v_max_other,
                                                        self._a_corr_ego, self._const_dist_offset,
                                                        self._emergency_profile, emg_idx)
            self.vehicle_dict[veh_id].append_safe_distance(safe_distance, time_step)
            if self._vehicle_dict[veh_id].rear_position(time_step) - ego_vehicle.front_position(time_step) \
                    <= safe_distance:
                vehicles_adjacent.append(veh_id)
            elif self._vehicle_dict[veh_id].rear_position(time_step) - ego_vehicle.front_position(time_step) \
                    > safe_distance and veh_id not in vehicle_ids_same:
                vehicle_ids_same.append(veh_id)
                self._vehicle_dict[veh_id].append_maneuver(Maneuver.LANE_FOLLOWING, time_step)
            elif self._vehicle_dict[veh_id].rear_position(time_step) - \
                    ego_vehicle.front_position(time_step) > safe_distance and veh_id in vehicle_ids_same:
                self._vehicle_dict[veh_id].append_maneuver(Maneuver.LANE_FOLLOWING, time_step)
        elif self._vehicle_dict[veh_id].maneuver_list.get(time_step - 1) is not None \
                and self._vehicle_dict[veh_id].maneuver_list.get(time_step - 1) == Maneuver.LANE_FOLLOWING:
            self._vehicle_dict[veh_id].append_maneuver(Maneuver.LANE_FOLLOWING, time_step)

    def detect_cutin_same(self, time_step: int, vehicle_ids_same_lane: List[int], vehicle_ids_cutin: List[int],
                          ego_vehicle: Vehicle, emg_idx: int) -> Tuple[List[int], List[int]]:
        """
        Updates vehicle maneuver

        :param time_step: current time step
        :param vehicle_ids_same_lane: vehicle IDs of vehicles in ego vehicle's lane
        :param vehicle_ids_cutin: vehicle IDs of vehicles in ego vehicle's adjacent lanes
        :param ego_vehicle: ego vehicle object
        :param emg_idx: execution index of emergency maneuver
        :return: lists with IDs of vehicles performing cut-in and driving in front of ego vehicle
        """
        vehicle_ids_same_lane_updated = []
        for idx, veh_id in enumerate(vehicle_ids_same_lane):
            if veh_id in vehicle_ids_cutin:
                continue
            if self._vehicle_dict[veh_id].maneuver_list.get(time_step) is Maneuver.CUTIN:
                continue
            ego_state_lon = ego_vehicle.states_lon[time_step]
            state_lon = self.vehicle_dict[veh_id].states_lon[time_step]
            safe_distance = safe_distance_profile_based(ego_vehicle.front_position(time_step), ego_state_lon.v,
                                                        ego_state_lon.a,
                                                        self.vehicle_dict[veh_id].rear_position(time_step),
                                                        state_lon.v, self._dt,  self._t_react,
                                                        self._a_min_ego, self._a_max_ego, self._j_max_ego,
                                                        self._v_min_ego, self._v_max_ego, self._a_min_other,
                                                        self._v_min_other, self._v_max_other, self._a_corr_ego,
                                                        self._const_dist_offset, self._emergency_profile, emg_idx)
            self.vehicle_dict[veh_id].append_safe_distance(safe_distance, time_step)
            if self._vehicle_dict[veh_id].rear_position(time_step) - ego_vehicle.front_position(time_step) \
                    <= safe_distance:
                if self.cutin_prev(self._vehicle_dict[veh_id], time_step):
                    self._vehicle_dict[veh_id].append_maneuver(Maneuver.CUTIN, time_step)
                    vehicle_ids_cutin.append(veh_id)
                else: # check if lane following
                    if (self._vehicle_dict[veh_id].maneuver_list.get(time_step - 1) is None and time_step == 0) or \
                            self._vehicle_dict[veh_id].maneuver_list.get(time_step - 1) is Maneuver.LANE_FOLLOWING \
                            or self._vehicle_dict[veh_id].maneuver_list.get(time_step - 1) is Maneuver.CUTIN:
                        self._vehicle_dict[veh_id].append_maneuver(Maneuver.LANE_FOLLOWING, time_step)
                        vehicle_ids_same_lane_updated.append(veh_id)
            else:
                self._vehicle_dict[veh_id].append_maneuver(Maneuver.LANE_FOLLOWING, time_step)
                if veh_id in vehicle_ids_cutin:
                    vehicle_ids_cutin.remove(veh_id)
                vehicle_ids_same_lane_updated.append(veh_id)

        return vehicle_ids_same_lane_updated, vehicle_ids_cutin

    def add_safe_distance(self, ego_vehicle: Vehicle, time_step: int, emg_idx: int):
        """
        Add safe distance to all vehicles which are relevant for visualization

        :param ego_vehicle: Vehicle
        :param time_step: current time step
        :param emg_idx: execution index of emergency maneuver
        """
        for veh_id in self._rel_obs_ids:
            if self._vehicle_dict.get(veh_id) is None:
                if self._plotting_profiles:
                    warnings.warn("Obstacle ID for visualization does not exist.")
                continue
            if self._vehicle_dict.get(veh_id).states_lon.get(time_step) is None:
                continue
            if self._vehicle_dict[veh_id].safe_distance_list.get(time_step) is None:
                safe_distance = safe_distance_profile_based(ego_vehicle.front_position(time_step),
                                                            ego_vehicle.states_lon[time_step].v,
                                                            ego_vehicle.states_lon[time_step].a,
                                                            self.vehicle_dict[veh_id].rear_position(time_step),
                                                            self.vehicle_dict[veh_id].states_lon[time_step].v,
                                                            self._dt, self._t_react, self._a_min_ego, self._a_max_ego,
                                                            self._j_max_ego, self._v_min_ego, self._v_max_ego,
                                                            self._a_min_other, self._v_min_other, self._v_max_other,
                                                            self._a_corr_ego, self._const_dist_offset,
                                                            self._emergency_profile, emg_idx)
                self._vehicle_dict[veh_id].safe_distance_list[time_step] = safe_distance

    def vehicles_fov(self, time_step: int, ego_s_position: float, vehicle_ids_left: List[int],
                     vehicle_ids_right: List[int], vehicle_ids_same: List[int]) -> Tuple[List[int], List[int],
                                                                                         List[int]]:
        """
        Extracts list with IDs of vehicles within the field of view of the ego vehicle at specific time step

        :param time_step: current time step
        :param ego_s_position: ego vehicle s-coordinate
        :param vehicle_ids_left: list of vehicles in left lane
        :param vehicle_ids_right: list of vehicles in right lane
        :param vehicle_ids_same: list of vehicles in ego vehicle's lane
        :return: lists with vehicle IDs for left, right, and ego vehicle's lane
        """
        vehicle_ids_left_updated = \
            [veh_id for veh_id in vehicle_ids_left
             if self._vehicle_dict[veh_id].states_lon.get(time_step) is not None
             and ego_s_position <= self._vehicle_dict[veh_id].rear_position(time_step) <= ego_s_position + self._fov]
        vehicle_ids_right_updated = \
            [veh_id for veh_id in vehicle_ids_right
             if self._vehicle_dict[veh_id].states_lon.get(time_step) is not None
             and ego_s_position <= self._vehicle_dict[veh_id].rear_position(time_step) <= ego_s_position + self._fov]
        vehicle_ids_same_updated = \
            [veh_id for veh_id in vehicle_ids_same
             if self._vehicle_dict[veh_id].states_lon.get(time_step) is not None
             and ego_s_position <= self._vehicle_dict[veh_id].rear_position(time_step) <= ego_s_position + self._fov]

        return vehicle_ids_left_updated, vehicle_ids_right_updated, vehicle_ids_same_updated

    def get_vehicles_from_ids(self, vehicle_ids: List[int]):
        """
        Extract vehicle objects from IDs

        :param vehicle_ids: list of vehicle IDs
        :return: list with vehicle objects
        """
        vehicles = []
        for veh_id in vehicle_ids:
            vehicles.append(self._vehicle_dict[veh_id])
        return vehicles

    def cut_out_detection(self, time_step: int, vehicle_ids_same_lane: List[int],
                          vehicle_ids_adjacent: List[int]) -> Tuple[List[int], List[int]]:
        """
        Detects vehicle on adjacent lane recently left ego vehicle's lane in the past and should
        still considered for comfort reasons

        :param time_step: current time step
        :param vehicle_ids_same_lane: vehicle IDs of vehicles in ego vehicle's lane
        :param vehicle_ids_adjacent: vehicle IDs of vehicles in ego vehicle's adjacent lanes
        :return: lists with IDs of vehicles driving in front of ego vehicle and in adjacent lanes
        """

        # delete not existing vehicles from cut out dictionary
        for veh_id in self._cut_out.keys():
            if veh_id in vehicle_ids_adjacent:
                continue
            else:
                del self._cut_out[veh_id]

        # detect relevant vehicles in adjacent lanes
        vehicle_ids_same_lane_updated = vehicle_ids_same_lane
        for idx, veh_id in enumerate(vehicle_ids_adjacent):
            if self._vehicle_dict.get(veh_id) is not None \
                    and self._vehicle_dict[veh_id].maneuver_list.get(time_step) is not None:
                if self._vehicle_dict[veh_id].maneuver_list[time_step] is Maneuver.LANE_FOLLOWING \
                        and self._cut_out.get(veh_id) is None:
                    vehicle_ids_same_lane.append(veh_id)
                    self._cut_out[veh_id] = time_step
                elif self._cut_out.get(veh_id) is not None and time_step - self._cut_out.get(veh_id) < self._time_leave:
                    vehicle_ids_same_lane.append(veh_id)

        return vehicle_ids_same_lane_updated, vehicle_ids_adjacent

    def extract_vehicles(self, time_step: int, ego_vehicle: Vehicle, emg_idx: int, ego_lane: Lane,
                         left_lane: Lane, right_lane: Lane) -> Tuple[List[Vehicle], List[Vehicle]]:
        """
        Returns lists for vehicles to consider for cut-in reaction and ACC calculation
        in front of the ego vehicle within the field of view

        :param time_step: current time step
        :param ego_vehicle: ego vehicle object
        :param emg_idx: execution index of emergency maneuver
        :param ego_lane: lane of ego vehicle (merge of current lanelet with successor lanelets)
        :param left_lane: left lane of ego vehicle (merge of current left adjacent lanelet with its successor lanelets)
        :param right_lane: right lane of ego vehicle (merge of current right adjacent lanelet
        with its successor lanelets)
        :return: list for left, right and ego lane containing vehicles to consider in the next process steps
        """
        # Update vehicle dictionary with new vehicle information from current time step
        # and return vehicle ids mapped to lanes
        vehicle_ids_left, vehicle_ids_right, vehicle_ids_same = \
            self.update_vehicle_dict(time_step, ego_lane, left_lane, right_lane)

        # Select all vehicles in front of the ego vehicle within field of view at specific time step
        vehicle_ids_left, vehicle_ids_right, vehicle_ids_same = \
            self.vehicles_fov(time_step, ego_vehicle.front_position(time_step), vehicle_ids_left,
                              vehicle_ids_right, vehicle_ids_same)

        # Cut-in detection and safe distance calculation for left and right lane
        vehicle_ids_adjacent, vehicle_ids_same = self.detect_cutin_adjacent(time_step, vehicle_ids_right,
                                                                            vehicle_ids_left,
                                                                            vehicle_ids_same, ego_vehicle, emg_idx)

        # Cut-out detection and consideration of those vehicles
        vehicle_ids_same, vehicle_ids_adjacent = \
            self.cut_out_detection(time_step, vehicle_ids_same, vehicle_ids_adjacent)

        # Cut-in detection and safe distance calculation for same lane
        vehicle_ids_same_lane, vehicle_ids_cutin = self.detect_cutin_same(time_step, vehicle_ids_same,
                                                                          vehicle_ids_adjacent, ego_vehicle, emg_idx)

        # Get vehicle objects from IDs
        vehicles_same_lane = self.get_vehicles_from_ids(vehicle_ids_same_lane)
        vehicle_ids_cutin = self.get_vehicles_from_ids(vehicle_ids_cutin)

        # Add safe distance to vehicles which are relevant for visualization
        self.add_safe_distance(ego_vehicle, time_step, emg_idx)

        return vehicles_same_lane, vehicle_ids_cutin
