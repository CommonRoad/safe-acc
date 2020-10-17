from typing import List, Tuple
import numpy as np
import math


def emg_stopping_distance(s, v: float, a: float, dt: float, t_react: float, a_min: float, a_max: float,
                          j_max: float, v_min: float, v_max: float, a_corr: float,
                          emergency_profile: List[float]) -> float:
    """
   Calculates stopping distance of a vehicle which applies predefined emergency jerk profile
    and considering reaction time

   :param s: current longitudinal front position of vehicle
   :param v: current velocity of vehicle
   :param a: current acceleration of vehicle
   :param dt: time step size
   :param t_react: reaction time of vehicle
   :param a_min: minimum acceleration of vehicle
   :param a_max: maximum acceleration of vehicle
   :param j_max: maximum jerk of vehicle
   :param v_max: maximum velocity of vehicle
   :param v_min: minimum velocity of vehicle
   :param a_corr: maximum deviation of vehicle from real acceleration
   :param emergency_profile: jerk emergency profile
   :returns: stopping distance
   """
    # application of reaction time (maximum jerk of vehicle):
    a = min(a + a_corr, a_max)
    if v == v_max:
        a = 0
    steps_reaction_time = round(t_react / dt)
    for i in range(steps_reaction_time):
        s, v, a = vehicle_dynamics_jerk(s, v, a, j_max, v_min, v_max, a_min, a_max, dt)

    # application of the emergency profile:
    index = 0
    while v > 0:
        a = min(a + a_corr, a_max)
        if v == v_max:
            a = 0
        s, v, a = vehicle_dynamics_jerk(s, v, a, emergency_profile[index], v_min, v_max, a_min, a_max, dt)
        index += 1

    return s


def safe_distance_profile_based(s_follow: float, v_follow: float, a_follow: float, s_lead: float, v_lead: float,
                                dt: float, t_react: float, a_min_follow: float, a_max_follow: float,
                                j_max_follow: float, v_min_follow: float, v_max_follow: float, a_min_lead: float,
                                v_min_lead: float, v_max_lead, a_corr_follow: float, const_dist_offset: float,
                                emergency_profile: List[float], emg_idx: int) -> float:
    """
    Safe distance between two vehicles, where the following vehicle executes a predefined jerk profile
    and the leading vehicle applies full brake. The safe distance considers jerk limitations
    and the reaction time of the following vehicle.

    :param s_follow: current longitudinal position at following vehicle's front
    :param v_follow: current velocity of following vehicle
    :param a_follow: current acceleration of following vehicle
    :param s_lead: current longitudinal position at leading vehicle front
    :param v_lead: current velocity of leading vehicle
    :param dt: time step size
    :param t_react: reaction time of vehicle
    :param a_min_follow: minimum acceleration of following vehicle
    :param a_max_follow: maximum acceleration of following vehicle
    :param j_max_follow: maximum jerk of following vehicle
    :param v_min_follow: minimum velocity of following vehicle
    :param v_max_follow: maximum velocity of following vehicle
    :param a_min_lead: minimum acceleration of leading vehicle
    :param v_min_lead: minimum velocity of leading vehicle
    :param v_max_lead: maximum velocity of leading vehicle
    :param a_corr_follow: acceleration correction term for following vehicle
    :param const_dist_offset: desired distance at standstill
    :param emergency_profile: jerk emergency profile
    :param emg_idx: execution index of emergency maneuver
    :returns safe distance
    """
    v_lead_profile = [v_lead]
    s_lead_profile = [s_lead]
    v_follow_profile = [v_follow]
    s_follow_profile = [s_follow]
    a_follow_profile_tmp = [min(max(a_follow + a_corr_follow, a_min_follow), a_max_follow)]

    # braking of leading vehicle with a_min:
    while v_lead_profile[-1] > 0:
        s_lead_new, v_lead_new = vehicle_dynamics_acc(s_lead_profile[-1], v_lead_profile[-1], a_min_lead, v_min_lead,
                                                      v_max_lead, dt)
        s_lead_profile.append(s_lead_new)
        v_lead_profile.append(v_lead_new)

    # following vehicle motion during reaction time (maximum jerk of following vehicle):
    steps_reaction_time = round(t_react / dt)
    for i in range(steps_reaction_time):
        s_follow_new, v_follow_new, a_follow_new = vehicle_dynamics_jerk(s_follow_profile[-1], v_follow_profile[-1],
                                                                         a_follow_profile_tmp[-1], j_max_follow,
                                                                         v_min_follow, v_max_follow, a_min_follow,
                                                                         a_max_follow, dt)
        s_follow_profile.append(s_follow_new)
        v_follow_profile.append(v_follow_new)
        a_follow_profile_tmp.append(a_follow_new)

    index = emg_idx
    while v_follow_profile[-1] > 0:
        s_follow_new, v_follow_new, a_follow_new = vehicle_dynamics_jerk(s_follow_profile[-1], v_follow_profile[-1],
                                                                         a_follow_profile_tmp[-1],
                                                                         emergency_profile[index], v_min_follow,
                                                                         v_max_follow, a_min_follow, a_max_follow, dt)
        s_follow_profile.append(s_follow_new)
        v_follow_profile.append(v_follow_new)
        a_follow_profile_tmp.append(a_follow_new)
        index += 1

    # safe distance based on the position profile:
    len_follow = len(s_follow_profile)
    len_lead = len(s_lead_profile)
    if len_follow < len_lead:
        s_follow_profile += [s_follow_profile[-1]] * (len_lead - len_follow)
    else:
        s_lead_profile += [s_lead_profile[-1]] * (len_follow - len_lead)

    diff_max = np.min(np.array(s_lead_profile) - np.array(s_follow_profile))
    d_safe = s_lead - s_follow - diff_max
    d_safe += const_dist_offset

    return d_safe


def ics(s_follow: float, v_follow: float, a_min_follow: float, s_lead: float, v_lead: float, v_min_lead: float,
        v_max_lead: float, v_min_follow: float, v_max_follow: float, dt: float) -> bool:
    """
    Evaluation if following vehicle is in an Inevitable collision state (ICS) by assuming following vehicle immediately
    applies full braking and leading vehicle drives with constant velocity

    :param s_follow: current longitudinal position at following vehicle's front
    :param v_follow: current velocity of following vehicle
    :param a_min_follow: minimum acceleration of following vehicle
    :param s_lead: current longitudinal position at leading vehicle front
    :param v_lead: current velocity of leading vehicle
    :param v_min_lead: minimum velocity of leading vehicle
    :param v_max_lead: maximum velocity of leading vehicle
    :param v_min_follow: minimum velocity of following vehicle
    :param v_max_follow: maximum velocity of following vehicle
    :param dt: time step size
    :return: boolean indicating if following vehicle is in an ICS
    """
    v_lead_profile = [v_lead]
    s_lead_profile = [s_lead]
    v_follow_profile = [v_follow]
    s_follow_profile = [s_follow]

    while v_follow_profile[-1] > 0 and not s_follow_profile[-1] >= s_lead_profile[-1]:
        # constant driving of leading vehicle:
        s_lead_new, v_lead_new = vehicle_dynamics_acc(s_lead_profile[-1], v_lead_profile[-1], 0, v_min_lead,
                                                      v_max_lead, dt)
        s_lead_profile.append(s_lead_new)
        v_lead_profile.append(v_lead_new)

        # application of full braking by following vehicle:
        s_follow_new, v_follow_new = vehicle_dynamics_acc(s_follow_profile[-1], v_follow_profile[-1], a_min_follow,
                                                          v_min_follow, v_max_follow, dt)
        s_follow_profile.append(s_follow_new)
        v_follow_profile.append(v_follow_new)

    if s_follow_profile[-1] >= s_lead_profile[-1]:
        return True
    else:
        return False


def vehicle_dynamics_jerk(s_0: float, v_0: float, a_0: float, j_input: float, v_min: float, v_max: float, a_min: float,
                          a_max: float, dt: float) -> Tuple[float, float, float]:
    """
    Applying vehicle dynamics for one times step with jerk as input

    :param s_0: current longitudinal position at vehicle's front
    :param v_0: current velocity of vehicle
    :param a_0: current acceleration of vehicle
    :param j_input: jerk input for vehicle
    :param v_min: minimum velocity of vehicle
    :param v_max: maximum velocity of vehicle
    :param a_min: minimum acceleration of vehicle
    :param a_max: maximum acceleration of vehicle
    :param dt: time step size
    :return: new position, velocity, acceleration
    """
    a_new = a_0 + j_input * dt
    if a_new > a_max:
        t_a = abs((a_max - a_0) / j_input)
        a_new = a_max
    elif a_new < a_min:
        t_a = abs((a_0 - a_min) / j_input)
        a_new = a_min
    else:
        t_a = dt

    v_new = v_0 + a_0 * dt + 0.5 * j_input * t_a**2
    if v_new > v_max and j_input != 0.0:    # apply solution for quadratic equation
        discriminant = a_0 ** 2 - 4 * 0.5 * j_input * (v_0 - v_max)
        if discriminant >= 0:
            t_1 = (a_0 + math.sqrt(discriminant)) / (2 * 0.5 * j_input)
            t_2 = (a_0 - math.sqrt(discriminant)) / (2 * 0.5 * j_input)
            t = min(abs(t_1), abs(t_2))
        else:
            t = 0
        v_new = v_max
        s_new = s_0 + v_0 * t + 0.5 * a_0 * t ** 2 + (1 / 6) * j_input * t ** 3 + (dt - t) * v_max
    elif v_new > v_max and j_input == 0.0:
        t = abs((v_max - v_0) / a_0)
        v_new = v_max
        s_new = s_0 + v_0 * t + 0.5 * a_0 * t ** 2 + (1 / 6) * j_input * t ** 3 + (dt - t) * v_max
    elif v_new < v_min and j_input != 0.0:
        discriminant = a_0 ** 2 - 4 * 0.5 * j_input * (v_0 - v_min)
        if discriminant >= 0:
            t_1 = (a_0 + math.sqrt(discriminant)) / (2 * 0.5 * j_input)
            t_2 = (a_0 - math.sqrt(discriminant)) / (2 * 0.5 * j_input)
            t = min(abs(t_1), abs(t_2))
        else:
            t = 0
        v_new = v_min
        s_new = s_0 + v_0 * t + 0.5 * a_0 * t ** 2 + (1 / 6) * j_input * t_a ** 3 + (dt - t) * v_min
    elif v_new < v_min and j_input == 0.0:
        t = abs((v_0 - v_min) / a_0)
        v_new = v_min
        s_new = s_0 + v_0 * t + 0.5 * a_0 * t ** 2 + (1 / 6) * j_input * t ** 3 + (dt - t) * v_min
    else:
        t_v = dt
        s_new = s_0 + v_0 * t_v + 0.5 * a_0 * t_a ** 2 + (1 / 6) * j_input * t_a ** 3

    if v_new == v_max or v_new == v_min:
        a_new = 0

    return s_new, v_new, a_new


def vehicle_dynamics_acc(s_0: float, v_0: float, a_input: float, v_min: float, v_max: float,
                         dt: float) -> Tuple[float, float]:
    """
    Applying vehicle dynamics for one times step with acceleration as input

    :param s_0: current longitudinal position at vehicle's front
    :param v_0: current velocity of vehicle
    :param a_input: acceleration input for vehicle
    :param v_min: minimum velocity of vehicle
    :param v_max: maximum velocity of vehicle
    :param dt: time step size
    :return: new position and velocity
    """
    v_new = v_0 + a_input * dt
    if v_new > v_max:
        if a_input == 0.0:
            raise ValueError  # v_0 is already larger than v_max, there is somewhere else a bug
        t_v = (v_max - v_0) / a_input
        v_new = v_max
    elif v_new < v_min:
        if a_input == 0.0:
            raise ValueError  # v_0 is already smaller than v_min, there is somewhere else a bug
        t_v = (v_0 - v_min) / a_input
        v_new = v_min
    else:
        t_v = dt

    s_new = s_0 + v_0 * t_v + 0.5 * a_input * t_v ** 2

    return s_new, v_new
