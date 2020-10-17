from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3
from typing import Dict, Union
import ruamel.yaml
import enum
import math
from common.util_motion import emg_stopping_distance
from decimal import Decimal


@enum.unique
class LaneCategory(enum.Enum):
    """ Enum containing lane numbers with respect to the ego vehicle's lane."""
    SAME = 0
    LEFT = -1
    RIGHT = 1


@enum.unique
class Strategy(enum.Enum):
    """ Enum containing execution strategies of the safe ACC."""
    CUTIN = 1
    ACC = 2
    CC = 3
    ACC_AND_CUTIN = 4
    ICS = 5


@enum.unique
class Maneuver(enum.Enum):
    """ Enum containing possible maneuvers of a vehicle."""
    CUTIN = 1
    LANE_FOLLOWING = 2


def calc_v_max(ego_vehicle_param: Dict, simulation_param: Dict, acc_param: Dict) -> int:
    """
    Calculates safety based maximum allowed velocity rounded to next lower integer value

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param acc_param: dictionary with parameters of the acc related algorithms
    :returns maximum allowed velocity
    """
    s_ego = 0
    v_ego = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
    a_ego = 0  # ego vehicle is already at v_max
    dt = simulation_param.get("dt")
    t_react = ego_vehicle_param.get("t_react")
    a_min = ego_vehicle_param.get("a_min") + ego_vehicle_param.get("a_corr")
    a_max = ego_vehicle_param.get("a_max")
    a_corr = ego_vehicle_param.get("a_corr")
    j_max = ego_vehicle_param.get("j_max")
    v_min = ego_vehicle_param.get("v_min")
    v_max = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
    emergency_profile = acc_param.get("emergency").get("emergency_profile")
    stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt, t_react, a_min, a_max, j_max, v_min, v_max,
                                              a_corr, emergency_profile)
    dist_offset = ego_vehicle_param.get("fov") - stopping_distance - acc_param.get("common").get("const_dist_offset")
    while dist_offset <= 0 or dist_offset >= 0.5 \
            and not (v_max == ego_vehicle_param.get("dynamics_param").longitudinal.v_max and dist_offset > 0.5):
        if ego_vehicle_param.get("fov") - stopping_distance - acc_param.get("common").get("const_dist_offset") < 0:
            v_max -= 0.001
        else:
            v_max += 0.001
        if v_max > ego_vehicle_param.get("dynamics_param").longitudinal.v_max:
            v_max = ego_vehicle_param.get("dynamics_param").longitudinal.v_max
        if v_max < v_min:
            v_max = v_min
        stopping_distance = emg_stopping_distance(s_ego, v_ego, a_ego, dt, t_react, a_min, a_max, j_max, v_min, v_max,
                                                  a_corr, emergency_profile)
        dist_offset = ego_vehicle_param.get("fov") - stopping_distance - acc_param.get("common").get(
            "const_dist_offset")

    return math.floor(v_max)


def create_ego_vehicle_param(ego_vehicle_param: Dict, simulation_param: Dict, acc_param: Dict) -> Dict:
    """
    Update ACC vehicle parameters

    :param ego_vehicle_param: dictionary with physical parameters of the ego vehicle
    :param simulation_param: dictionary with parameters of the simulation environment
    :param acc_param: dictionary with parameters of the acc related algorithms
    :returns updated dictionary with parameters of ACC vehicle
    """
    if ego_vehicle_param.get("vehicle_type") == 1:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle1()
    elif ego_vehicle_param.get("vehicle_type") == 2:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle2()
    elif ego_vehicle_param.get("vehicle_type") == 3:
        ego_vehicle_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for ACC vehicle in config file defined.')

    v_max_safety = calc_v_max(ego_vehicle_param, simulation_param, acc_param)
    ego_vehicle_param["v_max"] = min(v_max_safety, ego_vehicle_param.get("dynamics_param").longitudinal.v_max)

    if not -1e-12 <= (Decimal(str(ego_vehicle_param.get("t_react"))) %
                      Decimal(str(simulation_param.get("dt")))) <= 1e-12:
        raise ValueError('Reaction time must be multiple of time step size.')

    return ego_vehicle_param


def create_other_vehicles_param(other_vehicles_param: Dict) -> Dict:
    """
    Update other vehicle's parameters

    :param other_vehicles_param: dictionary with physical parameters of the leading vehicle
    :returns updated dictionary with parameters of other vehicles
    """
    if other_vehicles_param.get("vehicle_number") == 1:
        other_vehicles_param["dynamics_param"] = parameters_vehicle1()
    elif other_vehicles_param.get("vehicle_number") == 2:
        other_vehicles_param["dynamics_param"] = parameters_vehicle2()
    elif other_vehicles_param.get("vehicle_number") == 3:
        other_vehicles_param["dynamics_param"] = parameters_vehicle3()
    else:
        raise ValueError('Wrong vehicle number for leading vehicle in config file defined.')

    return other_vehicles_param


def create_acc_param(acc_param: Dict, j_min: float) -> Dict:
    """
    Evaluate ACC controller parameters

    :param acc_param: dictionary with parameters of the acc related algorithms
    :param j_min: minimum jerk of ego vehicle
    :returns updated acc_param dictionary
    """
    emergency_profile = acc_param.get("emergency").get("emergency_profile")
    emergency_profile += [j_min] * acc_param.get("emergency").get("num_steps_fb")
    acc_param["emergency"]["emergency_profile"] = emergency_profile

    return acc_param


def load_yaml(file_name: str) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    with open(file_name, 'r') as stream:
        try:
            config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            return None
