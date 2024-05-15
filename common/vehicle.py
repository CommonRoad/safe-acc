from commonroad.geometry.shape import Shape, Rectangle
from typing import Union, Set, Dict, List
from common.configuration import Maneuver, LaneCategory
from commonroad.scenario.trajectory import State
from commonroad.scenario.obstacle import ObstacleType


class StateLongitudinal:
    """
    Longitudinal state in curvilinear coordinate system
    """
    def __init__(self, s: float, v: float, a: float):
        """
        :param s: longitudinal position in curvilinear coordinates
        :param v: longitudinal velocity in curvilinear coordinates
        :param a: longitudinal acceleration in curvilinear coordinates
        """
        self._s = s
        self._v = v
        self._a = a

    @property
    def s(self) -> float:
        return self._s

    @s.setter
    def s(self, value: float):
        self._s = value

    @property
    def v(self) -> float:
        return self._v

    @v.setter
    def v(self, value: float):
        self._v = value

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, value: float):
        self._a = value


class StateLateral:
    """
    Lateral state in curvilinear coordinate system
    """
    def __init__(self, d: float, theta: float):
        """
        :param d: lateral position in curvilinear coordinates
        :param theta: orientation of vehicle
        """
        self._d = d
        self._theta = theta

    @property
    def d(self) -> float:
        return self._d

    @d.setter
    def d(self, value: float):
        self._d = value

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value: float):
        self._theta = value


class Vehicle:
    """
    Representation of a vehicle with state lists and dictionaries over complete simulation horizon
    """
    def __init__(self, state_lon: StateLongitudinal, state_lat: StateLateral, shape: Union[Shape, Rectangle],
                 lane_category: LaneCategory, cr_state: State, vehicle_id: int, obstacle_type: ObstacleType,
                 lanelet_assignment: Set[int]):
        """
        :param state_lon: initial longitudinal state of vehicle
        :param state_lat: initial lateral state of vehicle
        :param shape: CommonRoad rectangle representing shape of vehicle
        :param lane_category: enum value indicating left, right or same lane
        :param cr_state: initial CommonRoad state of vehicle
        :param vehicle_id: id of vehicle
        :param obstacle_type: type of the vehicle, e.g. parked car, car, bus, ...
        :param lanelet_assignment: initial lanelet assignment
        """
        self._states_lon = {cr_state.time_step: state_lon}
        self._states_lat = {cr_state.time_step: state_lat}
        self._states_cr = {cr_state.time_step: cr_state}
        self._jerk_list = {}
        self._safe_distance_list = {}
        self._shape = shape
        self._lane_category_list = {cr_state.time_step: lane_category}
        self._id = vehicle_id
        self._obstacle_type = obstacle_type
        self._maneuver_list = {}
        self._lanelet_assignment = {cr_state.time_step: lanelet_assignment}
        # ADDED
        self.ego_lanelet_id = None
        self.ego_as_obstacle = None
        self.ego_lane = None
        self.left_lane = None
        self.right_lane = None
        self.emergency_step_counter = 0
        self.emergency_active = False

    @property
    def shape(self) -> Rectangle:
        return self._shape

    @property
    def lane_number_list(self) -> Dict[int, LaneCategory]:
        return self._lane_category_list

    @property
    def id(self) -> int:
        return self._id

    @property
    def maneuver_list(self) -> Dict[int, Maneuver]:
        return self._maneuver_list

    @property
    def states_lon(self) -> Dict[int, StateLongitudinal]:
        return self._states_lon

    @property
    def states_lat(self) -> Dict[int, StateLateral]:
        return self._states_lat

    @property
    def states_cr(self) -> Dict[int, State]:
        return self._states_cr

    @property
    def state_list_cr(self) -> List[State]:
        state_list = []
        for state in self._states_cr.values():
            state_list.append(state)
        return state_list

    @property
    def safe_distance_list(self) -> Dict[int, float]:
        return self._safe_distance_list

    @property
    def jerk_list(self) -> Dict[int, float]:
        return self._jerk_list

    @property
    def obstacle_type(self) -> ObstacleType:
        return self._obstacle_type

    @property
    def lanelet_assignment(self) -> Dict[int, Set[int]]:
        return self._lanelet_assignment

    def rear_position(self, time_step: int) -> float:
        """
        Calculates rear position of vehicle based on longitudinal curvilinear state

        :param time_step: time step to consider
        :returns rear position [m]
        """
        return self._states_lon[time_step].s - self.shape.length/2

    def front_position(self, time_step: int) -> float:
        """
        Calculates front position of vehicle based on longitudinal curvilinear state

        :param time_step: time step to consider
        :returns front position [m]
        """
        return self._states_lon[time_step].s + self.shape.length/2

    def append_state_lon(self, state: StateLongitudinal, time_step: int):
        """
        Appends a state to the longitudinal curvilinear state list

        :param state: state to append
        :param time_step: time step of new data
        """
        self._states_lon[time_step] = state

    def append_state_lat(self, state: StateLateral, time_step: int):
        """
        Appends a state to the lateral curvilinear state list

        :param state: state to append
        :param time_step: time step of new data
        """
        self._states_lat[time_step] = state

    def append_state_cr(self, state: State, time_step: int):
        """
        Appends a state to the CommonRoad state list

        :param state: state to append
        :param time_step: time step of new data
        """
        self._states_cr[time_step] = state

    def append_lane_category(self, lane_category: LaneCategory, time_step: int):
        """
        Appends a lane number to the lane number list (index corresponds to time step)

        :param lane_category: value to append
        :param time_step: time step of new data
        """
        self._lane_category_list[time_step] = lane_category

    def append_maneuver(self, maneuver: Maneuver, time_step: int):
        """
        Sets the maneuver type at a specific time step

        :param maneuver: maneuver of vehicle
        :param time_step: time step of new data
        """
        self._maneuver_list[time_step] = maneuver

    def append_safe_distance(self, distance: float, time_step: int):
        """
        Sets safe distance at a specific time step

        :param distance: safe distance of vehicle
        :param time_step: time step of new data
        """
        self._safe_distance_list[time_step] = distance

    def append_lanelet_assignment(self, lanelets: Set[int], time_step: int):
        """
        Sets lanelets at a specific time step

        :param lanelets: set of lanelet IDs at a specific time step
        :param time_step: time step of new data
        """
        self._lanelet_assignment[time_step] = lanelets

    def append_jerk(self, jerk: float, time_step: int):
        """
        Sets jerk at a specific time step

        :param jerk: jerk of vehicle
        :param time_step: time step of new data
        """
        self._jerk_list[time_step] = jerk
