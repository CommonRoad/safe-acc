from typing import List, Dict

from acc.nominal_acc import NominalACC
from common.vehicle import Vehicle
from common.util_motion import safe_distance_profile_based, vehicle_dynamics_jerk, vehicle_dynamics_acc


class SafetyLayer:
    """
    Safe Layer of Nominal ACC"
    """
    def __init__(self, nominal_acc: NominalACC, simulation_param: Dict, ego_vehicle_param: Dict, acc_param: Dict,
                 other_vehicles_param: Dict):
        """
        :param nominal_acc: nominal ACC system
        :param simulation_param: dictionary with parameters of the simulation environment
        :param ego_vehicle_param: dictionary with physical parameters of the ACC vehicle
        :param acc_param: dictionary with parameters of the acc related algorithms
        :param other_vehicles_param: dictionary with general parameters of other vehicles
        """
        self._dt = simulation_param.get("dt")
        self._const_dist_offset = acc_param.get("common").get("const_dist_offset")
        self._t_react = ego_vehicle_param.get("t_react")
        self._j_min_ego = ego_vehicle_param.get("j_min")
        self._j_max_ego = ego_vehicle_param.get("j_max")
        self._v_min_other = other_vehicles_param.get("v_min")
        self._v_max_other = other_vehicles_param.get("v_max")
        self._a_min_ego = ego_vehicle_param.get("a_min")
        self._a_min_other = other_vehicles_param.get("a_min")
        self._j_min_other = other_vehicles_param.get("j_min")
        self._j_max_other = other_vehicles_param.get("j_max")
        self._a_max_ego = ego_vehicle_param.get("a_max")
        self._a_corr = ego_vehicle_param.get("a_corr")
        self._v_max_ego = ego_vehicle_param.get("v_max")
        self._v_min_ego = ego_vehicle_param.get("v_min")
        self._verbose = simulation_param.get("verbose_mode")
        self.emergency_step_counter = 0
        self.acc = nominal_acc
        self.activated_emergency_acc_profile = None
        self._emergency_profile = acc_param.get("emergency").get("emergency_profile")

    @property
    def emergency_profile(self) -> List[float]:
        return self._emergency_profile

    @emergency_profile.setter
    def emergency_profile(self, profile: List[float]):
        self._emergency_profile = profile

    def create_activated_emergency_profile(self, a_ego: float):
        """
        Creates emergency acceleration profile every time an emergency maneuver must be executed

        :param a_ego: acceleration of ego vehicle
        """
        self.activated_emergency_acc_profile = [a_ego + self.emergency_profile[0] * self._dt]
        for jerk in self.emergency_profile[1::]:
            self.activated_emergency_acc_profile.append(max(self._a_min_ego,
                                                            self.activated_emergency_acc_profile[-1] + jerk * self._dt))

    def validate_input(self, jerk: float, leading_vehicle: Vehicle, ego_vehicle: Vehicle,
                       time_step: int) -> bool:
        """
        Validates input with respect to safe distance at next time step

        :param jerk: calculated input jerk
        :param leading_vehicle: leading vehicle object acceleration is based on
        :param ego_vehicle: ego vehicle object
        :param time_step: current time step
        :returns boolean indicating if input is safe or not
        """
        if jerk is None:
            return False
        if jerk < 0:  # separation necessary to ensure safety of lower level controller
            # (no continuous integration of jerk)
            a_ego = max(min(ego_vehicle.states_lon[time_step].a + jerk * self._dt, self._a_max_ego), self._a_min_ego)
            s_ego, v_ego = vehicle_dynamics_acc(ego_vehicle.front_position(time_step),
                                                ego_vehicle.states_lon[time_step].v, a_ego, self._v_min_ego,
                                                self._v_max_ego, self._dt)
        else:
            s_ego, v_ego, a_ego = vehicle_dynamics_jerk(ego_vehicle.front_position(time_step),
                                                        ego_vehicle.states_lon[time_step].v,
                                                        ego_vehicle.states_lon[time_step].a, jerk, self._v_min_ego,
                                                        self._v_max_ego, self._a_min_ego, self._a_max_ego, self._dt)

        s_lead, v_lead = vehicle_dynamics_acc(leading_vehicle.rear_position(time_step),
                                              leading_vehicle.states_lon[time_step].v, self._a_min_other,
                                              self._v_min_other, self._v_max_other, self._dt)

        safe_distance = safe_distance_profile_based(s_ego, v_ego, a_ego, s_lead, v_lead, self._dt,
                                                    self._t_react, self._a_min_ego, self._a_max_ego, self._j_max_ego,
                                                    self._v_min_ego, self._v_max_ego, self._a_min_other,
                                                    self._v_min_other, self._v_max_other, self._a_corr,
                                                    self._const_dist_offset, self._emergency_profile, 0)

        if s_lead - s_ego > safe_distance:
            return True
        else:
            return False

    def calculate_acceleration(self, ego_vehicle: Vehicle, vehicles: List[Vehicle], time_step: int) -> List[float]:
        """
        Calculates input acceleration for each vehicle in field of view within the ego vehicle's lane
        For all vehicles which activate emergency maneuver a single input is returned

        :param ego_vehicle: calculated input acceleration
        :param vehicles: list of leading vehicle objects
        :param ego_vehicle: ego vehicle object
        :param time_step: current time step
        :returns list of accelerations
        """
        acceleration_list = []
        emergency_active = False    # emergency maneuver already activated this time step
        for veh in vehicles:
            a = self.acc.calculate_input(ego_vehicle.states_lon[time_step].a,
                                         ego_vehicle.states_lon[time_step].v, veh.states_lon[time_step].v,
                                         ego_vehicle.front_position(time_step), veh.rear_position(time_step),
                                         veh.safe_distance_list[time_step])
            if a is None or not self.validate_input(a, veh, ego_vehicle, time_step):
                if self._verbose:
                    if a is None:
                        print("Other vehicle ID: " + str(veh.id))
                    print("Emergency Maneuver active")
                emergency_active = True
            else:
                acceleration_list.append(a)

        if emergency_active:
            if self.emergency_step_counter == 0:
                self.create_activated_emergency_profile(ego_vehicle.states_lon[time_step].a)
                a = self.activated_emergency_acc_profile[self.emergency_step_counter]
                self.emergency_step_counter += 1
            else:
                a = self.activated_emergency_acc_profile[self.emergency_step_counter]
                self.emergency_step_counter += 1
            acceleration_list.append(a)
        else:
            self.emergency_step_counter = 0
        return acceleration_list
