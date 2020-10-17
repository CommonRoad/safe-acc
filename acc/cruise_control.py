from typing import Union, Dict, Tuple, Type
import numpy as np
import math
from common.quadratic_program import QP


class CruiseControl:
    """Cruise control in case no leading vehicle exists"""

    def __init__(self, simulation_param: Dict, cc_config_param: Dict, ego_vehicle_param: Dict):
        """
        :param simulation_param: dictionary with parameters of the simulation environment
        :param cc_config_param: dictionary with parameters for the cruise control
        :param ego_vehicle_param: dictionary with physical parameters of the ACC vehicle
        """
        self._num_states = 2
        self._dt = simulation_param.get("dt")
        self._num_steps = math.floor(cc_config_param.get("t_h") / self._dt)
        self._a_min = ego_vehicle_param.get("a_min")
        self._a_max = ego_vehicle_param.get("a_max")
        self._v_max = ego_vehicle_param.get("v_max")
        self._v_des = ego_vehicle_param.get("v_des")
        self._verbose = simulation_param.get("verbose_mode")
        a_d, b_d, q, r = self.motion_equations(self._dt, cc_config_param.get("cost_v"), cc_config_param.get("cost_a"),
                                               cc_config_param.get("cost_j"))
        self._qp = QP(self._num_states, a_d, b_d, q, r, self._num_steps, cc_config_param.get("solver"))
        self._qp.create_constraint_matrices(ego_vehicle_param.get("j_min"), ego_vehicle_param.get("j_max"),
                                            self.state_constraints())

    @staticmethod
    def motion_equations(dt: float,  cost_v: float, cost_a: float, cost_j: float) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialization of discretized motion model

        :param dt: time step size
        :param cost_v: velocity weight for quadratic program
        :param cost_a: acceleration weight for quadratic program
        :param cost_j: jerk weight for quadratic program
        :returns A/B-matrix for motion equation in state-space form and weight matrices Q and R
        """
        a_d = np.array([[1, -dt],
                        [0, 1]])
        b_d = np.array([[-0.5 * dt ** 2],
                        [dt]])
        q = np.eye(2) * np.transpose(np.array([cost_v, cost_a]))
        r = np.array([cost_j])

        return a_d, b_d, q, r

    @staticmethod
    def state_constraints() -> Tuple[Tuple[Type[Union[float, int, None]], Type[Union[float, int, None]]], ...]:
        """
         Defines structure of constraints

        :returns set of tuples containing type of lower and upper constraint
        """
        return (float, float), (float, float)

    def calculate_input(self, a_ego: float, v_ego: float) -> Union[float, None]:
        """
        Execution of Cruise Control for longitudinal control

        :param a_ego: acceleration of ego vehicle
        :param v_ego: velocity of ego vehicle
        :returns longitudinal input acceleration for ego vehicle or None if no solution can be found
        """
        if v_ego == 0.0:  # if ego vehicle is standing acceleration must be zero (provided acceleration is the one
            # during the last time step)
            a_ego = 0.0
        x_0 = np.array([[self._v_des - v_ego], [a_ego]])

        self._qp.update_qp_matrices_const(x_0, [0, 0], constraints=((self._v_des - self._v_max, self._v_des),
                                                                    (self._a_min, self._a_max)))
        try:
            jerk = self._qp.solve()
        except ValueError:
            if self._verbose:
                print("Cruise Control: Quadratic program found no solution")
                print(a_ego, v_ego, self._v_des)
            return None
        acceleration = jerk[0] * self._dt + a_ego
        return acceleration
