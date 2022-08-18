from typing import List, Set, Dict, Union, Tuple
import numpy as np

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.trajectory import State
from commonroad_dc.geometry.geometry import CurvilinearCoordinateSystem

from common.vehicle import StateLongitudinal, StateLateral


class Lane:
    """
    Lane representation build from several lanelets
    """
    def __init__(self, merged_lanelet: Lanelet, contained_lanelets: List[int], road_network_param: Dict):
        """
        :param merged_lanelet: lanelet element of lane
        :param contained_lanelets: lanelets lane consists of
        :param road_network_param: dictionary with parameters for the road network
        """
        self._lanelet = merged_lanelet
        self._contained_lanelets = set(contained_lanelets)
        self._clcs = Lane.create_curvilinear_coordinate_system_from_reference(merged_lanelet.center_vertices,
                                                                              road_network_param)
        self._orientation = self._compute_orientation_from_polyline(merged_lanelet.center_vertices)
        self._curvature = self._compute_curvature_from_polyline(merged_lanelet.center_vertices)
        self._path_length = self._compute_path_length_from_polyline(merged_lanelet.center_vertices)
        self._width = self._compute_witdh_from_lanalet_boundary(merged_lanelet.left_vertices,
                                                                merged_lanelet.right_vertices)

    @property
    def lanelet(self) -> Lanelet:
        return self._lanelet

    @property
    def contained_lanelets(self) -> Set[int]:
        return self._contained_lanelets

    @property
    def clcs(self) -> CurvilinearCoordinateSystem:
        return self._clcs

    @property
    def orientation(self) -> np.ndarray:
        return self._orientation

    @property
    def curvature(self) -> np.ndarray:
        return self._curvature

    @property
    def path_length(self) -> np.ndarray:
        return self._path_length

    def width(self, s_position: float) -> float:
        """
        Calculates width of lane given a longitudinal position along lane

        :param s_position: longitudinal position
        :returns width of lane at a given position
        """
        return np.interp(s_position, self._path_length, self._width)

    @staticmethod
    def _compute_orientation_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes orientation along a polyline

        :param polyline: polyline for which orientation should be calculated
        :return: orientation along polyline
        """
        assert isinstance(polyline, np.ndarray) and len(polyline) > 1 and polyline.ndim == 2 and len(
            polyline[0, :]) == 2, '<Math>: not a valid polyline. polyline = {}'.format(polyline)
        if len(polyline) < 2:
            raise ValueError('Cannot create orientation from polyline of length < 2')

        orientation = [0]
        for i in range(1, len(polyline)):
            pt1 = polyline[i - 1]
            pt2 = polyline[i]
            tmp = pt2 - pt1
            orientation.append(np.arctan2(tmp[1], tmp[0]))

        return np.array(orientation)

    @staticmethod
    def _compute_curvature_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes curvature along a polyline

        :param polyline: polyline for which curvature should be calculated
        :return: curvature along  polyline
        """
        assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
            polyline[:, 0]) > 2, 'Polyline malformed for curvature computation p={}'.format(polyline)

        x_d = np.gradient(polyline[:, 0])
        x_dd = np.gradient(x_d)
        y_d = np.gradient(polyline[:, 1])
        y_dd = np.gradient(y_d)

        return (x_d * y_dd - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3. / 2.))

    @staticmethod
    def _compute_path_length_from_polyline(polyline: np.ndarray) -> np.ndarray:
        """
        Computes the path length of a polyline

        :param polyline: polyline for which path length should be calculated
        :return: path length along polyline
        """
        assert isinstance(polyline, np.ndarray) and polyline.ndim == 2 and len(
            polyline[:, 0]) > 2, 'Polyline malformed for pathlenth computation p={}'.format(polyline)

        distance = np.zeros((len(polyline),))
        for i in range(1, len(polyline)):
            distance[i] = distance[i - 1] + np.linalg.norm(polyline[i] - polyline[i - 1])

        return np.array(distance)

    @staticmethod
    def _compute_witdh_from_lanalet_boundary(left_polyline: np.ndarray, right_polyline: np.ndarray) -> np.ndarray:
        """
        Computes the width of a lanelet

        :param left_polyline: left boundary of lanelet
        :param right_polyline: right boundary of lanelet
        :return: width along lanelet
        """
        width_along_lanelet = np.zeros((len(left_polyline),))
        for i in range(len(left_polyline)):
            width_along_lanelet[i] = np.linalg.norm(left_polyline[i] - right_polyline[i])
        return width_along_lanelet

    @staticmethod
    def create_curvilinear_coordinate_system_from_reference(ref_path: np.array, road_network_param: Dict) \
            -> CurvilinearCoordinateSystem:
        """
        Generates curvilinear coordinate system for a reference path

        :param ref_path: reference path (polyline)
        :param road_network_param: dictionary containing parameters of the road network
        :returns curvilinear coordinate system for reference path
        """
        return CurvilinearCoordinateSystem(ref_path,
                                           # default_projection_domain_limit=50.,
                                           # eps=1.,
                                           resample=True,
                                           num_chaikins_corner_cutting=road_network_param['num_chaikins_corner_cutting'],
                                           max_polyline_resampling_step=road_network_param['polyline_resampling_step'])

    def create_curvilinear_state(self, state: State, prev_state: Union[State, None], dt: float) \
            -> Tuple[StateLongitudinal, StateLateral]:
        """
        Computes initial state of ego vehicle

        :param state: CommonRoad state
        :param prev_state: previous CommonRoad state
        :param dt: time step size
        :return: lateral and longitudinal state of vehicle
        """
        s, d = self._clcs.convert_to_curvilinear_coords(state.position[0], state.position[1])
        theta_cl = np.interp(s, self._path_length, self._orientation)
        if prev_state is not None:
            x_lon = StateLongitudinal(s, state.velocity, (state.velocity - prev_state.velocity) / dt)
        else:
            x_lon = StateLongitudinal(s, state.velocity, 0)

        x_lat = StateLateral(d, theta_cl - state.orientation)

        return x_lon, x_lat


def get_lanes(scenario: Scenario, ego_lanelet_id: int, road_network_param: Dict) -> \
        Tuple[Lane, Lane, Lane]:
    """
    Generates ego vehicle and adjacent lanes with corresponding curvilinear coordinate system

    :param scenario: CommonRoad scenario
    :param ego_lanelet_id: lanelet ID which the current state of the ego vehicle occupies
    :param road_network_param: dictionary with parameters for the road network
    :returns lane and corresponding curvilinear coordinate system for ego, left adjacent and right
    adjacent lanelet, respectively
    """
    ego_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)
    start_lanelet = ego_lanelet
    if ego_lanelet.predecessor is not None and len(ego_lanelet.predecessor) > 0:
        start_lanelet = scenario.lanelet_network.find_lanelet_by_id(ego_lanelet.predecessor[0])

    if ego_lanelet.successor is not None and len(ego_lanelet.successor) > 0:
        merged_lanelets, merge_jobs = \
            Lanelet.all_lanelets_by_merging_successors_from_lanelet(start_lanelet, scenario.lanelet_network, 1000)
        ego_lane = Lane(merged_lanelets[0], merge_jobs[0], road_network_param)
    elif ego_lanelet.predecessor and len(ego_lanelet.predecessor) > 0:
        merged_lanelets, merge_jobs = \
            Lanelet.all_lanelets_by_merging_successors_from_lanelet(start_lanelet, scenario.lanelet_network, 1000)
        ego_lane = Lane(merged_lanelets[0], merge_jobs[0], road_network_param)
    else:
        ego_lane = Lane(ego_lanelet, [ego_lanelet_id], road_network_param)

    if start_lanelet.adj_left_same_direction is True:
        merged_lanelets, merge_jobs = \
            Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                scenario.lanelet_network.find_lanelet_by_id(start_lanelet.adj_left), scenario.lanelet_network,
                1000)
        left_lane = Lane(merged_lanelets[0], merge_jobs[0], road_network_param)
    else:
        left_lane = None
    if start_lanelet.adj_right_same_direction is True:
        merged_lanelets, merge_jobs = \
            Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                scenario.lanelet_network.find_lanelet_by_id(start_lanelet.adj_right), scenario.lanelet_network,
                1000)
        right_lane = Lane(merged_lanelets[0], merge_jobs[0], road_network_param)
    else:
        right_lane = None

    return ego_lane, left_lane, right_lane
