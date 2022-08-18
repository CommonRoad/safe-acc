from timeit import default_timer as timer
from typing import Tuple, List, Dict
import numpy as np
import math
import os

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from common.obstacle_selection import ObstacleSelection
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.solution import VehicleType
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad.common.util import Interval

from commonroad_dc.boundary import boundary
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics

from common.configuration import *
from common.vehicle import Vehicle, StateLateral, StateLongitudinal
from commonroad.scenario.trajectory import State
from common.quadratic_program import QP
from acc.safe_acc import SafeACC
from acc.cruise_control import CruiseControl
from acc.recapturing_control import RecapturingControl
from acc.safety_layer import SafetyLayer
from acc.nominal_acc import NominalACC
from common.lane import Lane, get_lanes


class Simulation:
    """
    Class for the simulation of a CommonRoad scenario using the safe ACC as longitudinal ego vehicle controller
    """
    def __init__(self, simulation_param: Dict, road_network_param: Dict, acc_param: Dict, ego_vehicle_param: Dict,
                 other_vehicles_param: Dict, input_constr_param: Dict,
                 recapturing_data_nominal: Dict[Tuple[float, float, float, float], Tuple[float, float]],
                 recapturing_data_acc_bounded: Dict[Tuple[float, float, float, float], Tuple[float, float]],
                 recapturing_controllers: List[Dict[int, QP]]):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param road_network_param: dictionary with parameters for the road network
        :param acc_param: dictionary with parameters of the acc related algorithms
        :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
        :param other_vehicles_param: dictionary with general parameters of the other vehicles
        :param input_constr_param: dictionary with parameters for input constraints
        :param recapturing_data_nominal: dictionary with safe distance and clearance time for given ego
        and cut-in state for nominal recapturing controller
        :param recapturing_data_acc_bounded: dictionary with safe distance and clearance time for given ego
        and cut-in state for acceleration bounded recapturing controller
        :param recapturing_controllers: controllers for different time horizons
        """
        if simulation_param.get("verbose_mode") is True:
            print("Initialization")
        # Initialization of general parameters and variables
        self._dt = simulation_param.get("dt")
        self._verbose_mode = simulation_param.get("verbose_mode")
        self._time_measuring = simulation_param.get("time_measuring")
        self._ego_vehicle_param = ego_vehicle_param
        self._input_constr_param = input_constr_param
        self._comp_time = []
        self._ego_vehicle = None
        self._road_network_param = road_network_param

        # Initialization of CommonRoad related variables
        cr_file = simulation_param.get("commonroad_scenario_folder") + "/" + simulation_param.get(
            "commonroad_benchmark_id") + ".xml"
        self._scenario, self._planning_problem_set = \
            CommonRoadFileReader(cr_file).open()
        self._planning_problem = list(self._planning_problem_set.planning_problem_dict.values())[0]

        self._cc = create_collision_checker(self._scenario)
        _, self._road_boundary_sg_rectangles = \
            boundary.create_road_boundary_obstacle(self._scenario, method='obb_rectangles')
        self._vehicle_model_feasibilty = VehicleDynamics.KS(VehicleType(ego_vehicle_param.get("vehicle_type")))

        # Initialization of safe ACC and obstacle selection
        self._nominal_acc = NominalACC(simulation_param, acc_param.get("nominal_acc"), ego_vehicle_param)
        self._cruise_control = CruiseControl(simulation_param, acc_param.get("cruise_control"),
                                             ego_vehicle_param)
        self._recapturing_control = RecapturingControl(simulation_param, acc_param.get("cutin"), ego_vehicle_param,
                                                       other_vehicles_param, acc_param.get("emergency"),
                                                       acc_param.get("common"), recapturing_data_nominal,
                                                       recapturing_data_acc_bounded, recapturing_controllers)
        self._safety_layer = SafetyLayer(self._nominal_acc, simulation_param, ego_vehicle_param,
                                         acc_param, other_vehicles_param)
        self._safe_acc = SafeACC(self._cruise_control, self._safety_layer, self._recapturing_control,
                                 self._scenario, simulation_param, ego_vehicle_param, acc_param, other_vehicles_param)
        self._object_selection = ObstacleSelection(self._scenario, self._safety_layer, ego_vehicle_param,
                                                   other_vehicles_param, simulation_param, acc_param)

    @property
    def ego_vehicle(self) -> Vehicle:
        return self._ego_vehicle

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def planning_problem_set(self) -> PlanningProblemSet:
        return self._planning_problem_set

    @property
    def obstacles(self):
        return self._object_selection.vehicle_dict

    @property
    def safe_acc(self):
        return self._safe_acc

    @property
    def comp_time(self):
        return self._comp_time

    def get_max_scenario_duration(self) -> int:
        """
        Computes the maximum time step which occurs in the scenario (max. prediction time step of all obstacles or
        upper goal region time step)

        :return: Last time step of scenario
        """
        max_time = 0
        for obstacle in self._scenario.dynamic_obstacles:
            if obstacle.prediction.trajectory.state_list[-1].time_step > max_time:
                max_time = obstacle.prediction.trajectory.state_list[-1].time_step

        if max_time == 0:
            max_time = int(self._planning_problem.goal.state_list[0].time_step.end / 2)
        return max_time

    def simulate(self):
        """
        Simulates CommonRoad scenario using an ego vehicle equipped with the safe ACC
        """
        # Initialization lane related variables
        if self._planning_problem.initial_state.velocity == 0:
            initial_steering_angle = 0
        else:
            initial_steering_angle = 0  # we assume a standing vehicle has steering angle 0
        initial_ego_state = State(position=self._planning_problem.initial_state.position,
                                  velocity=self._planning_problem.initial_state.velocity,
                                  orientation=self._planning_problem.initial_state.orientation,
                                  steering_angle=initial_steering_angle, time_step=0)
        ego_lanelet_id = self._scenario.lanelet_network.find_lanelet_by_position(
                    [initial_ego_state.position])[0][0]
        ego_lane, left_lane, right_lane = get_lanes(self._scenario, ego_lanelet_id, self._road_network_param)

        # Initialization of ego vehicle object
        ego_state_lon, ego_state_lat = ego_lane.create_curvilinear_state(initial_ego_state, None, self._dt)
        ego_state_lat.d = 0
        self._ego_vehicle = Vehicle(ego_state_lon, ego_state_lat,
                                    Rectangle(parameters_vehicle2().l, parameters_vehicle2().w), LaneCategory.SAME,
                                    initial_ego_state, self._scenario.generate_object_id(), ObstacleType.CAR,
                                    ego_lane.contained_lanelets)

        # Start simulation
        for time_step in range(0, self.get_max_scenario_duration()):
            print("time step: " + str(time_step))
            # Load obstacles in the field of view of the ego vehicle at current time step
            obstacles_at_current_time_step = self._scenario.obstacles_by_position_intervals([Interval(
                self._ego_vehicle.states_cr[time_step].position[0] - self._ego_vehicle_param.get("fov"),
                self._ego_vehicle.states_cr[time_step].position[0] + self._ego_vehicle_param.get("fov")),
                Interval(self._ego_vehicle.states_cr[time_step].position[1] - self._ego_vehicle_param.get("fov"),
                         self._ego_vehicle.states_cr[time_step].position[1] + self._ego_vehicle_param.get("fov"))],
                time_step=time_step)
            self._scenario.assign_obstacles_to_lanelets([time_step],
                                                        {obs.obstacle_id for obs in obstacles_at_current_time_step})
            current_ego_lanelet_id = \
                self._scenario.lanelet_network.find_lanelet_by_position(
                    [self._ego_vehicle.states_cr[time_step].position])[0][0]
            if current_ego_lanelet_id != ego_lanelet_id \
                    and current_ego_lanelet_id != \
                    self._scenario.lanelet_network.find_lanelet_by_id(current_ego_lanelet_id).successor:
                ego_lane, left_lane, right_lane = get_lanes(self._scenario, ego_lanelet_id, self._road_network_param)
                ego_lanelet_id = current_ego_lanelet_id

            vehicles_fov_same_lane, vehicles_fov_cutin = \
                self._object_selection.extract_vehicles(time_step, self._ego_vehicle,
                                                        self.safe_acc.emergency_maneuver_activity[-1], ego_lane,
                                                        left_lane, right_lane)

            start_time = 0
            if self._time_measuring:
                start_time = timer()

            # Calculate ego vehicle input
            a_lon = self._safe_acc.execute(vehicles_fov_same_lane, vehicles_fov_cutin, self._ego_vehicle, time_step)

            # Apply input constraints
            a_lon = self.apply_input_constraints(a_lon, self._ego_vehicle.states_lon[time_step].v,
                                                 vehicles_fov_same_lane, time_step)
            if self._time_measuring:
                end_time = timer()
                self._comp_time.append(end_time - start_time)

            # Propagate ego vehicle
            self._ego_vehicle = self.propagate_ego_vehicle(self._ego_vehicle, self._dt, a_lon, ego_lane, time_step)

            # Collision and goal region check
            trajectory = Trajectory(time_step + 1, [self.ego_vehicle.state_list_cr[time_step + 1]])
            traj_pred = TrajectoryPrediction(trajectory=trajectory, shape=self._ego_vehicle.shape)
            co = create_collision_object(traj_pred)
            if self._cc.collide(co):
                print("collision")
                break
            if self._planning_problem.goal_reached(trajectory)[0] is True:
                print("goal reached")
                break

    def propagate_ego_vehicle(self, ego_vehicle: Vehicle, dt: float, a_lon: float,
                              ego_lane: Lane, time_step: int) -> Vehicle:
        """
        Propagates ego vehicle for one time step

        :param ego_vehicle: ego vehicle object
        :param dt: time step size
        :param a_lon: longitudinal input
        :param ego_lane: ego vehicle lane object
        :param time_step: current time step
        :returns updated ego vehicle object
        """
        x_lon = ego_vehicle.states_lon[time_step]

        # check whether velocity gets negative
        if x_lon.v + a_lon * dt < 0:
            dt = x_lon.v / a_lon
            v_new = 0.0
        else:
            v_new = a_lon * dt + x_lon.v

        # compute new longitudinal state
        s_new = 0.5 * a_lon * dt ** 2 + x_lon.v * dt + x_lon.s
        x_lon_new = StateLongitudinal(s_new, v_new, a_lon)

        # compute new lateral state (we assume ego vehicle follows exactly center line)
        d_new = 0
        theta_new = 0
        x_lat_new = StateLateral(d_new, theta_new)

        # create Cartesian state
        cartesian_coord = ego_lane.clcs.convert_to_cartesian_coords(s_new, d_new)
        theta_cart = np.interp(s_new, ego_lane.path_length, ego_lane.orientation)
        new_time_step = time_step + 1
        curvature = np.interp(s_new, ego_lane.path_length, ego_lane.curvature)
        x_cr_new = State(position=cartesian_coord, velocity=v_new, orientation=theta_cart,
                         steering_angle=math.atan(self._ego_vehicle_param.get("dynamics_param").l * curvature),
                         time_step=new_time_step)

        # add new states to ego vehicle object
        ego_vehicle.append_state_lon(x_lon_new, new_time_step)
        ego_vehicle.append_state_lat(x_lat_new, new_time_step)
        ego_vehicle.append_state_cr(x_cr_new, new_time_step)
        ego_vehicle.append_lane_category(LaneCategory.SAME, new_time_step)
        ego_vehicle.append_maneuver(Maneuver.LANE_FOLLOWING, new_time_step)
        ego_vehicle.append_safe_distance(0, new_time_step)
        ego_vehicle.append_jerk((a_lon - ego_vehicle.states_lon[time_step].a) / dt, time_step)

        return ego_vehicle

    def apply_input_constraints(self, a: float, v_cur: float, vehicles_same_lane: List[Vehicle],
                                time_step: int) -> float:
        """
        Applies environmental and vehicle constraints on calculated acceleration

        :param a: acceleration calculated  safe ACC
        :param v_cur: current velocity
        :param vehicles_same_lane: vehicles within the ego vehicle's field of view and lane at current time step
        :param time_step: current time step
        :return: constrained input acceleration for inner controller
        """
        # string stability
        string_stability_constr = self._ego_vehicle_param.get("a_max")
        for idx in range(self._input_constr_param.get("string_stability_horizon")):
            time_idx = time_step - idx
            for vehicle in vehicles_same_lane:
                if vehicle.states_lon.get(time_idx) is None:
                    continue
                string_stability_constr = max(abs(vehicle.states_lon.get(time_idx).a), string_stability_constr)

        if a > self._input_constr_param.get("string_stability_gamma") * string_stability_constr:
            a = self._input_constr_param.get("string_stability_gamma") * string_stability_constr

        # velocity constraint
        c = False
        if (v_cur <= self._ego_vehicle_param.get("v_min") and a <= 0) or \
                (v_cur >= self._ego_vehicle_param.get("v_max") and a >= 0):
            c = True

        # engine power constraint
        if v_cur > self._ego_vehicle_param.get("v_limit_engine"):
            a_max = self._ego_vehicle_param.get("a_max") * (self._ego_vehicle_param.get("v_limit_engine") / v_cur)
        else:
            a_max = self._ego_vehicle_param.get("a_max")

        # negative acceleration caused by drag
        a_dr = \
            - (1/(2 * self._ego_vehicle_param.get("dynamics_param").m)) * \
            self._input_constr_param.get("air_density") * self._ego_vehicle_param.get("drag_coefficient") * \
            self._ego_vehicle_param.get("frontal_area") * (v_cur + self._input_constr_param.get("v_wind"))**2

        # negative acceleration caused by incline
        a_i = -9.81 * np.sin(self._input_constr_param.get("road_incline_angle"))

        # acceleration constraint
        if c:
            a = 0
        elif not c and a <= self._ego_vehicle_param.get("a_min") + a_dr + a_i:
            a = self._ego_vehicle_param.get("a_min") + a_dr + a_i
        elif not c and a >= a_max + a_dr + a_i:
            a = a_max + a_dr + a_i
        else:
            a = a

        return a
