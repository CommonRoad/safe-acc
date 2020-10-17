from typing import Union, Dict, Tuple, Type
import numpy as np
import math
from common.quadratic_program import QP


class NominalACC:
    """Controller for leading vehicles within the same lane as the ego vehicle"""

    def __init__(self, simulation_param: Dict, nominal_acc_param: Dict, ego_vehicle_param: Dict):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param nominal_acc_param: dictionary with parameters for the nominal ACC system
        :param ego_vehicle_param: dictionary with physical parameters of the ACC vehicle
        """
        self._num_states = 3
        self._dt = simulation_param.get("dt")
        self._num_steps = math.floor(nominal_acc_param.get("t_h") / self._dt)
        self._a_min = ego_vehicle_param.get("a_min")
        self._a_max = ego_vehicle_param.get("a_max")
        self._v_max = ego_vehicle_param.get("v_max")
        self._fov = ego_vehicle_param.get("fov")
        self._verbose = simulation_param.get("verbose_mode")
        a_d, b_d, q, r = self.motion_equations(self._dt, nominal_acc_param.get("cost_s"),
                                               nominal_acc_param.get("cost_v"), nominal_acc_param.get("cost_a"),
                                               nominal_acc_param.get("cost_j"))
        self._qp = QP(self._num_states, a_d, b_d, q, r, self._num_steps, nominal_acc_param.get("solver"))
        self._qp.create_constraint_matrices(ego_vehicle_param.get("j_min"), ego_vehicle_param.get("j_max"),
                                            self.state_constraint_format())

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

    def calculate_input(self, a_ego: float, v_ego: float, v_lead: float, s_ego: float,
                        s_lead: float, safe_distance: float) -> Union[float, None]:
        """
        Calculation of longitudinal vehicle input with nominal ACC

        :param a_ego: acceleration of ego vehicle
        :param v_ego: velocity of ego vehicle
        :param v_lead: velocity of leading vehicle
        :param s_ego: front position of ego vehicle
        :param s_lead: rear position of leading vehicle
        :param safe_distance: necessary safe distance between ego and leading vehicle
        :returns longitudinal input acceleration for ego vehicle or None if no solution can be found
        """
        if v_ego == 0.0:  # if ego vehicle is standing acceleration must be zero (provided acceleration is the one
            # during the last time step)
            a_ego = 0.0
        x_0 = np.array([[s_lead - s_ego], [v_lead - v_ego], [a_ego]])

        self._qp.update_qp_matrices_const(x_0, [safe_distance + 5, 0, 0], constraints=((safe_distance, None),
                                                                                       (v_lead - self._v_max, v_lead),
                                                                                       (self._a_min, self._a_max)))
        try:
            jerk = self._qp.solve()
        except:
            if self._verbose:
                print("Nominal ACC: Quadratic program found no solution")
                print(a_ego, v_ego, v_lead, s_ego, s_lead, safe_distance)
            return None
        acceleration = jerk[0] * self._dt + a_ego

        return acceleration
