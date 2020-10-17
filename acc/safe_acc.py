from typing import List, Dict, Tuple, Union

from commonroad.scenario.scenario import Scenario

from acc.cruise_control import CruiseControl
from acc.recapturing_control import RecapturingControl
from acc.safety_layer import SafetyLayer

from common.vehicle import Vehicle
from common.configuration import Strategy
from common.util_motion import emg_stopping_distance, ics


class SafeACC:
    """
    Safe ACC with reaction to cut-in vehicles
    """
    def __init__(self, cc: CruiseControl, safety_layer: SafetyLayer, cutin: RecapturingControl,
                 scenario: Scenario, simulation_param: Dict, ego_vehicle_param: Dict, acc_param: Dict,
                 other_vehicles_param: Dict):
        """
        :param cc: Cruise Control system object
        :param safety_layer: Safety layer object
        :param cutin: Recapturing controller object
        :param scenario: CommonRoad scenario
        :param simulation_param: dictionary with parameters of the simulation environment
        :param acc_param: dictionary with parameters of the safe ACC
        :param ego_vehicle_param: dictionary with physical parameters of the ACC vehicle
        :param other_vehicles_param: dictionary with general parameters of other vehicles
        """
        self._strategy_list = []
        self._num_vehicles = []  # (num same lane, num reduced same lane, num cut-in)
        self._cc = cc
        self._cutin = cutin
        self._safety_layer = safety_layer
        self._scenario = scenario
        self._emergency_maneuver_activity = [0]
        self._verbose = simulation_param.get("verbose_mode")
        self._dt = simulation_param.get("dt")
        self._a_min_ego = ego_vehicle_param.get("a_min")
        self._a_max_ego = ego_vehicle_param.get("a_max")
        self._a_corr = ego_vehicle_param.get("a_corr")
        self._j_max_ego = ego_vehicle_param.get("j_max")
        self._v_min_ego = ego_vehicle_param.get("v_min")
        self._v_max_ego = ego_vehicle_param.get("v_max")
        self._v_min_other = other_vehicles_param.get("v_min")
        self._v_max_other = other_vehicles_param.get("v_max")
        self._t_react = ego_vehicle_param.get("t_react")
        self._vehicle_reduction_active = acc_param.get("common").get("vehicle_reduction")

    @property
    def emergency_maneuver_activity(self) -> List[int]:
        return self._emergency_maneuver_activity

    @emergency_maneuver_activity.setter
    def emergency_maneuver_activity(self, emergency: List[int]):
        self._emergency_maneuver_activity = emergency

    @property
    def strategy_list(self) -> List[Strategy]:
        return self._strategy_list

    @property
    def num_vehicles(self) -> List[Tuple[Union[float, None], Union[float, None], Union[float, None]]]:
        return self._num_vehicles

    @strategy_list.setter
    def strategy_list(self, strategy_list: List[Strategy]):
        self._strategy_list = strategy_list

    def __getitem__(self, idx):
        return self._emergency_maneuver_activity[idx]

    def __setitem__(self, idx, value):
        self._emergency_maneuver_activity[idx] = value

    @staticmethod
    def strategy_selection(vehicles_same_lane: List[Vehicle], vehicles_cutin: List[Vehicle]) -> Strategy:
        """
        Selects strategy to execute in the following process steps

        :param vehicles_same_lane: vehicles within the ego vehicle's field of view and
        lane at current time step
        :param vehicles_cutin: vehicles within the ego vehicle's field of view performing
        a cut-in in the ego vehicle's lane  at current time step
        :returns strategy to execute
        """
        if len(vehicles_same_lane) == 0 and len(vehicles_cutin) == 0:
            return Strategy.CC
        elif len(vehicles_same_lane) > 0 and len(vehicles_cutin) == 0:
            return Strategy.ACC
        elif len(vehicles_same_lane) == 0 and len(vehicles_cutin) > 0:
            return Strategy.CUTIN
        elif len(vehicles_same_lane) > 0 and len(vehicles_cutin) > 0:
            return Strategy.ACC_AND_CUTIN

    def vehicle_reduction(self, vehicles_same_lane: List[Vehicle], ego_vehicle: Vehicle, time_step: int):
        """
        Excludes vehicles in ego vehicle's lane from further process steps

        :param vehicles_same_lane: vehicles within the ego vehicle's field of view and lane at current time step
        :param ego_vehicle: ego vehicle object
        :param time_step: Current time step
        """
        if len(vehicles_same_lane) <= 1:
            return

        vehicles_same_lane.sort(key=lambda veh: veh.states_lon[time_step].s)
        # exclude faster preceding vehicles:
        min_velocity = vehicles_same_lane[0].states_lon[time_step].v
        idx = 1
        while idx < len(vehicles_same_lane):
            if min_velocity <= vehicles_same_lane[idx].states_lon[time_step].v:
                vehicles_same_lane.remove(vehicles_same_lane[idx])
            else:
                min_velocity = vehicles_same_lane[idx].states_lon[time_step].v
                idx += 1

        # exclude vehicles behind stopping distance:
        if len(vehicles_same_lane) > 1:
            s_ego = ego_vehicle.front_position(time_step) + ego_vehicle.states_lon[time_step].v * self._dt \
                    + 0.5 * self._a_max_ego + self._dt**2
            v_ego = ego_vehicle.states_lon[time_step].v + self._a_max_ego + self._dt
            a_ego = self._a_max_ego
            s_stop = emg_stopping_distance(s_ego, v_ego, a_ego, self._dt, self._t_react,
                                           self._a_min_ego + self._a_corr, self._a_max_ego, self._j_max_ego,
                                           self._v_min_ego, self._v_max_ego, self._a_corr,
                                           self._safety_layer.emergency_profile)
            for vehicle in vehicles_same_lane:
                if len(vehicles_same_lane) == 1:
                    return
                if vehicle.rear_position(time_step) > s_stop:
                    vehicles_same_lane.remove(vehicle)

    def ics_same_lane(self, vehicles_same_lane: List[Vehicle], ego_vehicle: Vehicle, time_step: int) -> bool:
        """
        Evaluation if ego vehicle is in an inevitable collision state (ICS) for any vehicle in the same lane

        :param vehicles_same_lane: vehicles within the ego vehicle's field of view and lane at current time step
        :param ego_vehicle: ego vehicle object
        :param time_step: current time step
        :returns boolean indicating if ego vehicle is in an ICS
        """
        for veh in vehicles_same_lane:
            if ics(ego_vehicle.front_position(time_step), ego_vehicle.states_lon[time_step].v,
                   self._a_min_ego, veh.rear_position(time_step), veh.states_lon[time_step].v, self._v_min_other,
                   self._v_max_other, self._v_min_ego, self._v_max_ego, self._dt):
                return True
        return False

    def execute(self, vehicles_same_lane: List[Vehicle], vehicles_cutin: List[Vehicle],
                ego_vehicle: Vehicle, time_step: int):
        """
        Selects strategy, reduces vehicles to consider, and executes controllers according to selected strategy

        :param vehicles_same_lane: vehicles within the ego vehicle's field of view and lane at current time step
        :param vehicles_cutin: vehicles within the ego vehicle's field of view performing a cut-in in the
         ego vehicle's lane  at current time step
        :param ego_vehicle: ego vehicle object
        :param time_step: Current time step
        """
        accelerations = []
        num_same_lane_initial = len(vehicles_same_lane)
        if self.ics_same_lane(vehicles_same_lane, ego_vehicle, time_step):
            if self._verbose:
                print("ICS same lane")
            self.strategy_list.append(Strategy.ICS.value)
            self.num_vehicles.append((num_same_lane_initial, len(vehicles_same_lane), len(vehicles_cutin)))
            self.emergency_maneuver_activity.append(-1)
            return self._a_min_ego

        if self._vehicle_reduction_active:
            self.vehicle_reduction(vehicles_same_lane, ego_vehicle, time_step)
        strategy = self.strategy_selection(vehicles_same_lane, vehicles_cutin)
        self.strategy_list.append(strategy.value)
        self.num_vehicles.append((num_same_lane_initial, len(vehicles_same_lane), len(vehicles_cutin)))

        if strategy == Strategy.ACC or strategy == Strategy.ACC_AND_CUTIN:
            if self._verbose:
                print("ACC active")
            accelerations += self._safety_layer.calculate_acceleration(ego_vehicle, vehicles_same_lane,
                                                                       time_step)
        else:
            self._safety_layer.emergency_step_counter = 0

        if strategy == Strategy.CC:
            if self._verbose:
                print("Cruise Control active")
            accelerations.append(self._cc.calculate_input(ego_vehicle.states_lon[time_step].a,
                                                          ego_vehicle.states_lon[time_step].v))
        if strategy == Strategy.CUTIN or strategy == Strategy.ACC_AND_CUTIN:
            if self._verbose:
                print("Cutin recapturing active")
            accelerations += self._cutin.calculate_accelerations(ego_vehicle, vehicles_cutin, time_step)

        if self._safety_layer.emergency_step_counter > 0:
            self.emergency_maneuver_activity.append(self._safety_layer.emergency_step_counter)
        else:
            self.emergency_maneuver_activity.append(self._safety_layer.emergency_step_counter)
        return min(accelerations)
