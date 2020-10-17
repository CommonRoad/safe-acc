from typing import Union, Dict, Tuple, Type, List
import numpy as np

from common.vehicle import Vehicle
from common.util_motion import ics
from common.quadratic_program import QP


class RecapturingControl:
    def __init__(self, simulation_param: Dict, cutin_config_param: Dict, ego_vehicle_param: Dict,
                 cutin_vehicle_param: Dict, emergency_param: Dict, acc_param: Dict, recapturing_data_nominal,
                 recapturing_data_acc_bounded, recapturing_controllers: List[Dict[int, QP]]):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param cutin_config_param: dictionary with parameters for the recapturing controller
        :param ego_vehicle_param: dictionary with physical parameters of the ACC vehicle
        :param cutin_vehicle_param: dictionary with physical parameters of the cut-in vehicle
        :param emergency_param: dictionary with parameters of the emergency controller
        :param acc_param: dictionary with parameters with general parameters of the ACC
        :param recapturing_data_nominal: dictionary with safe distance and clearance time for given ego
        and cut-in state for nominal recapturing controller
        :param recapturing_data_acc_bounded: dictionary with safe distance and clearance time for given ego
        and cut-in state for acceleration bounded recapturing controller
        :param recapturing_controllers: controllers for different time horizons
        """
        self._num_states = 3
        self._dt = simulation_param.get("dt")
        self._a_min_ego = ego_vehicle_param.get("a_min")
        self._j_max_ego = ego_vehicle_param.get("j_max")
        self._a_max_ego = ego_vehicle_param.get("a_max")
        self._v_min_ego = ego_vehicle_param.get("v_min")
        self._v_max_ego = ego_vehicle_param.get("v_max")
        self._t_react = ego_vehicle_param.get("t_react")
        self._a_min_cutin = cutin_vehicle_param.get("a_min")
        self._v_min_cutin = cutin_vehicle_param.get("v_min")
        self._v_max_cutin = cutin_vehicle_param.get("v_max")
        self._a_corr = ego_vehicle_param.get("a_corr")
        self._const_dist_offset = acc_param.get("const_dist_offset")
        self._emergency_profile = emergency_param.get("emergency_profile")
        self._verbose = simulation_param.get("verbose_mode")
        self._t_clear_min = cutin_config_param.get("t_clear_min")
        self._t_clear_max = cutin_config_param.get("t_clear_max")
        self._t_clear_step = cutin_config_param.get("t_clear_step")
        self._v_ego_step = cutin_config_param.get("v_ego_step")
        self._v_cutin_step = cutin_config_param.get("v_cutin_step")
        self._a_ego_step = cutin_config_param.get("a_ego_step")
        self._delta_s_step = cutin_config_param.get("delta_s_step")
        self._fov = ego_vehicle_param.get("fov")
        self._qp_nominal = recapturing_controllers[0]
        self._qp_acc_bounded = recapturing_controllers[1]
        self._recapturing_data_nominal = recapturing_data_nominal
        self._recapturing_data_acc_bounded = recapturing_data_acc_bounded
        self._remaining_steps_vehicles = {}

    @staticmethod
    def motion_equations(dt: float, cost_s: float, cost_v: float, cost_a: float,
                         cost_j: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialization of discrete motion model

        :param dt: time step size
        :param cost_s: velocity weight for quadratic program
        :param cost_v: velocity weight for quadratic program
        :param cost_a: acceleration weight for quadratic program
        :param cost_j: jerk weight for quadratic program

        :returns A/B-matrix for motion equation in state-space form and weight matrices Q and R
        """
        a_d = np.array([[1, dt, -0.5 * dt ** 2],
                        [0, 1, -dt],
                        [0, 0, 1]])
        b_d = np.array([[-(1 / 6) * dt ** 3],
                        [-0.5 * dt ** 2],
                        [dt]])
        q = np.eye(3) * np.transpose(np.array([cost_s, cost_v, cost_a]))
        r = np.array([cost_j])

        return a_d, b_d, q, r

    @staticmethod
    def state_constraint_format() -> Tuple[Tuple[Type[Union[float, int, None]], Type[Union[float, int, None]]], ...]:
        """
        Defines structure of constraints

        :returns set of tuples containing type of lower and upper constraint
        """
        return (float, None), (float, float), (float, float)

    def calculate_input_nominal(self, a_ego: float, v_ego: float, v_lead: float, s_ego: float,
                                s_lead: float, safe_distance: float, num_steps: int) -> Union[np.ndarray, float, None]:
        """
        Calculation of longitudinal vehicle input with nominal ACC (jerk constraints) for cut-in recapturing

        :param a_ego: acceleration of ego vehicle
        :param v_ego: velocity of ego vehicle
        :param v_lead: velocity of leading vehicle
        :param s_ego: front position of ego vehicle
        :param s_lead: rear position of leading vehicle
        :param safe_distance: necessary safe distance between ego and leading vehicle
        :param num_steps: number of time steps after which safe distance must be achieved
        :returns longitudinal jerk list for ego vehicle or None if no solution can be found
        """
        if v_ego == 0.0:  # if ego vehicle is standing, acceleration must be zero (provided acceleration is the one
            # during the last time step)
            a_ego = 0.0
        x_0 = np.array([[s_lead - s_ego], [v_lead - v_ego], [a_ego]])
        lb_safe_distance = np.ones((num_steps, 1)) * -np.infty
        lb_safe_distance[-1] = safe_distance
        self._qp_nominal[num_steps].update_qp_matrices_dyn(x_0, [safe_distance, 0, 0],
                                                           constraints=((lb_safe_distance, None),
                                                                        (np.ones((num_steps, 1)) *
                                                                         (v_lead - self._v_max_ego),
                                                                         np.ones((num_steps, 1)) * v_lead),
                                                                        (np.ones((num_steps, 1)) * self._a_min_ego,
                                                                         np.ones((num_steps, 1)) * self._a_max_ego)))
        try:
            jerk = self._qp_nominal[num_steps].solve()
        except ValueError:
            if self._verbose:
                print("Safety Recapturing Nominal Controller: Quadratic program found no solution")
                print(a_ego, v_ego, v_lead, s_ego, s_lead, safe_distance)
            return None

        return jerk

    def calculate_input_bounded(self, a_ego: float, v_ego: float, v_lead: float, s_ego: float,
                                s_lead: float, safe_distance: float, num_steps: int) -> Union[np.ndarray, float, None]:
        """
        Calculation of longitudinal vehicle input with nominal ACC (no jerk constraints) for cut-in recapturing

        :param a_ego: acceleration of ego vehicle
        :param v_ego: velocity of ego vehicle
        :param v_lead: velocity of leading vehicle
        :param s_ego: front position of ego vehicle
        :param s_lead: rear position of leading vehicle
        :param safe_distance: necessary safe distance between ego and leading vehicle
        :param num_steps: number of time steps after which safe distance must be achieved
        :returns longitudinal jerk list for ego vehicle or None if no solution can be found
        """
        if v_ego == 0.0:  # if ego vehicle is standing acceleration must be zero (provided acceleration is the one
            # during the last time step)
            a_ego = 0.0
        x_0 = np.array([[s_lead - s_ego], [v_lead - v_ego], [a_ego]])
        lb_safe_distance = np.ones((num_steps, 1)) * -np.infty
        lb_safe_distance[-1] = safe_distance
        self._qp_acc_bounded[num_steps].update_qp_matrices_dyn(x_0, [safe_distance, 0, 0],
                                                               constraints=((lb_safe_distance, None),
                                                                            (np.ones((num_steps, 1)) *
                                                                            (v_lead - self._v_max_ego),
                                                                             np.ones((num_steps, 1)) * v_lead),
                                                                            (np.ones((num_steps, 1)) * self._a_min_ego,
                                                                             np.ones((num_steps, 1))
                                                                             * self._a_max_ego)))
        try:
            jerk = self._qp_acc_bounded[num_steps].solve()
        except ValueError:
            if self._verbose:
                print("Safety Recapturing Acceleration Controller: Quadratic program found no solution")
                print(a_ego, v_ego, v_lead, x_ego, x_lead, safe_distance)
            return None

        return jerk

    @staticmethod
    def roundup(value: float, multiple_of: int) -> int:
        """
        Rounds value up to the multiple of given integer

        :param value: value to round
        :param multiple_of: value rounded number should be multiple of
        :returns rounded value
        """
        value_rest = value % multiple_of
        return int(value if not value_rest else value + multiple_of - value_rest)

    def extract_vehicle_data_nominal(self, v_ego: float, v_cutin: float, delta_s: float, a_ego: float) -> \
            Tuple[Union[int, None], Union[float, None]]:
        """
        Extracts clearance time and safe distance from recapturing data dictionary for nominal recapturing control

        :param v_ego: ego vehicle velocity
        :param v_cutin: cut-in vehicle velocity
        :param delta_s: distance between ego vehicle and cut-in vehicle
        :param a_ego: ego vehicle acceleration
        :returns clearance time and safe distance (None for both if value does not exist)
        """
        v_ego = self.roundup(v_ego, self._v_ego_step)
        if v_ego > self._v_max_ego:
            v_ego = self._v_max_ego
        v_cutin = self.roundup(v_cutin, self._v_cutin_step) - self._v_cutin_step
        if v_cutin > self._v_max_cutin:
            v_cutin = self._v_max_cutin
        delta_s = self.roundup(delta_s, self._delta_s_step) - self._delta_s_step
        if delta_s > self._fov:
            delta_s = self._fov
        a_ego = self.roundup(a_ego, self._a_ego_step)
        if a_ego > self._a_max_ego:
            a_ego = self._a_max_ego

        if self._recapturing_data_nominal.get((delta_s, v_ego, v_cutin, a_ego)) is not None:
            t_clear, s_safe = self._recapturing_data_nominal[(delta_s, v_ego, v_cutin, a_ego)]
            return t_clear, s_safe
        else:
            return None, None

    def extract_vehicle_data_acc_bounded(self, v_ego: float, v_cutin: float, delta_s: float,
                                         a_ego: float) -> Tuple[Union[int, None], Union[float, None]]:
        """
        Extracts clearance time and safe distance from recapturing data dictionary
        for acceleration bounded recapturing control

        :param v_ego: ego vehicle velocity
        :param v_cutin: cut-in vehicle velocity
        :param delta_s: distance between ego vehicle and cut-in vehicle
        :param a_ego: ego vehicle acceleration
        :returns clearance time and safe distance (None for both if value does not exist)
        """
        v_ego = self.roundup(v_ego, self._v_ego_step)
        if v_ego > self._v_max_ego:
            v_ego = self._v_max_ego
        v_cutin = self.roundup(v_cutin, self._v_cutin_step) - self._v_cutin_step
        if v_cutin > self._v_max_cutin:
            v_cutin = self._v_max_cutin
        delta_s = self.roundup(delta_s, self._delta_s_step) - self._delta_s_step
        if delta_s > self._fov:
            delta_s = self._fov
        a_ego = self.roundup(a_ego, self._a_ego_step)
        if a_ego > self._a_max_ego:
            a_ego = self._a_max_ego

        if self._recapturing_data_acc_bounded.get((delta_s, v_ego, v_cutin, a_ego)) is not None:
            t_clear, s_safe = self._recapturing_data_acc_bounded[(delta_s, v_ego, v_cutin, a_ego)]
            return t_clear, s_safe
        else:
            return None, None

    def calculate_accelerations(self, ego_vehicle: Vehicle, vehicles: List[Vehicle], time_step: int) -> List[float]:
        """
        Calculates input acceleration based on each cut-in vehicle

        :param ego_vehicle: ego vehicle object
        :param vehicles: list of cut-in vehicles
        :param time_step: time step size
        :returns list of accelerations
        """
        accelerations = []
        state_lon_ego = ego_vehicle.states_lon[time_step]
        recapturing_data_vehicles = {}
        for idx, veh in enumerate(vehicles):
            if self._remaining_steps_vehicles.get(veh.id) is None:
                num_steps, safe_distance = self.extract_vehicle_data_nominal(ego_vehicle.states_lon[time_step].v,
                                                                             veh.states_lon[time_step].v,
                                                                             veh.rear_position(time_step) -
                                                                             ego_vehicle.front_position(time_step),
                                                                             ego_vehicle.states_lon[time_step].a)
            else:
                num_steps = self._remaining_steps_vehicles.get(veh.id)[0]
                safe_distance = self._remaining_steps_vehicles.get(veh.id)[1]
            if num_steps is None or safe_distance is None:
                num_steps, safe_distance = self.extract_vehicle_data_acc_bounded(ego_vehicle.states_lon[time_step].v,
                                                                                 veh.states_lon[time_step].v,
                                                                                 veh.rear_position(time_step) -
                                                                                 ego_vehicle.front_position(time_step),
                                                                                 ego_vehicle.states_lon[time_step].a)
                if num_steps is None or safe_distance is None:
                    veh_acceleration = self._a_min_ego
                else:
                    jerk = self.calculate_input_bounded(state_lon_ego.a, state_lon_ego.v, veh.states_lon[time_step].v,
                                                        ego_vehicle.front_position(time_step),
                                                        veh.rear_position(time_step), safe_distance, num_steps)
                    if jerk is None:
                        veh_acceleration = self._a_min_ego
                    else:
                        veh_acceleration = jerk[0] * self._dt + state_lon_ego.a
            else:
                recapturing_data_vehicles[veh.id] = (num_steps - 1, safe_distance)

                jerk = self.calculate_input_nominal(state_lon_ego.a, state_lon_ego.v, veh.states_lon[time_step].v,
                                                    ego_vehicle.front_position(time_step), veh.rear_position(time_step),
                                                    safe_distance, num_steps)
                if jerk is None:
                    if ics(ego_vehicle.front_position(time_step), ego_vehicle.states_lon[time_step].v,
                           self._a_min_ego, veh.rear_position(time_step), veh.states_lon[time_step].v,
                           self._v_min_cutin, self._v_max_cutin, self._v_min_ego, self._v_max_ego, self._dt):
                        veh_acceleration = self._a_min_ego
                    else:
                        jerk = self.calculate_input_bounded(state_lon_ego.a, state_lon_ego.v,
                                                            veh.states_lon[time_step].v,
                                                            ego_vehicle.front_position(time_step),
                                                            veh.rear_position(time_step), safe_distance, num_steps)
                        if jerk is None:
                            veh_acceleration = self._a_min_ego
                        else:
                            veh_acceleration = jerk[0] * self._dt + state_lon_ego.a
                else:
                    veh_acceleration = jerk[0] * self._dt + state_lon_ego.a
            accelerations.append(veh_acceleration)
        self._remaining_steps_vehicles = recapturing_data_vehicles
        return accelerations
