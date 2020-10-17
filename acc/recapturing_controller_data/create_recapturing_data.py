from common.configuration import *
from acc.recapturing_control import RecapturingControl
from common.util_motion import safe_distance_profile_based, vehicle_dynamics_jerk, vehicle_dynamics_acc
import pickle


def create_recapturing_data():
    """
    Precalculates solutions of recapturing controller for different initial states.
    During runtime a LUT is used to check whether a solution exists for the initial state.
    """
    # Initialization of variables for simulation
    print("Initialization")
    config = load_yaml("../../config.yaml")
    simulation_param = config.get("simulation_param")
    acc_param = create_acc_param(config.get("acc_param"), config.get("ego_vehicle_param").get("j_min"))
    other_vehicles_param = config.get("other_vehicles_param")
    ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), simulation_param, acc_param)
    with open("recapturing_controllers" + ".pkl", 'rb') as file:
        recapturing_controllers = pickle.load(file)

    recapturing_control = RecapturingControl(simulation_param, acc_param.get("cutin"), ego_vehicle_param,
                                             other_vehicles_param, acc_param.get("emergency"), acc_param.get("common"),
                                             {}, {}, recapturing_controllers)

    # Initialization of vehicle variables
    vehicles_recapturing_data_nominal = {}
    vehicles_recapturing_data_acc_bounded = {}
    v_min_ego = ego_vehicle_param.get("v_min")
    v_max_ego = ego_vehicle_param.get("v_max")
    v_min_cutin = other_vehicles_param.get("v_min")
    v_max_cutin = other_vehicles_param.get("v_max")
    a_min_ego = ego_vehicle_param.get("a_min")
    a_max_ego = ego_vehicle_param.get("a_max")
    a_min_cutin = other_vehicles_param.get("a_min")
    j_max_ego = ego_vehicle_param.get("j_max")
    dt = simulation_param.get("dt")
    a_cutin_min = acc_param.get("cutin").get("a_min_cut_in")
    t_react = ego_vehicle_param.get("t_react")
    a_corr = ego_vehicle_param.get("a_corr")
    const_dist_offset = acc_param.get("common").get("const_dist_offset")
    emergency_profile = acc_param.get("emergency").get("emergency_profile")
    emg_idx = 0
    t_clear_min = acc_param.get("cutin").get("t_clear_min")
    t_clear_max = acc_param.get("cutin").get("t_clear_max")
    t_clear_step = acc_param.get("cutin").get("t_clear_step")
    v_ego_step = acc_param.get("cutin").get("v_ego_step")
    v_cutin_step = acc_param.get("cutin").get("v_cutin_step")
    a_ego_step = acc_param.get("cutin").get("a_ego_step")
    delta_s_step = acc_param.get("cutin").get("delta_s_step")
    fov = ego_vehicle_param.get("fov")
    s_safe_step = acc_param.get("cutin").get("s_safe_step")

    # run QP evaluation for nominal recapturing controller
    print("Start evaluation")
    for v_ego in range(math.floor(v_min_ego), math.ceil(v_max_ego) + v_ego_step, v_ego_step):
        if v_ego > v_max_ego:
            v_ego = v_max_ego
        if v_ego < v_min_ego:
            v_ego = v_min_ego
        for v_cutin in range(math.floor(ego_vehicle_param.get("v_min")),
                             math.ceil(ego_vehicle_param.get("v_max")) + v_cutin_step, v_cutin_step):
            if v_cutin > v_max_cutin:
                v_cutin = v_max_cutin
            if v_cutin < v_min_cutin:
                v_cutin = v_min_cutin
            for delta_s in range(0, math.ceil(ego_vehicle_param.get("fov")) + delta_s_step, delta_s_step):
                if delta_s > fov:
                    delta_s = fov
                for a_ego in range(math.floor(ego_vehicle_param.get("a_min")),
                                   math.ceil(ego_vehicle_param.get("a_max")) + a_ego_step, a_ego_step):
                    if a_ego > a_max_ego:
                        a_ego = a_max_ego
                    if a_ego < a_min_ego:
                        a_ego = a_min_ego
                    for safe_distance in range(s_safe_step, math.ceil(ego_vehicle_param.get("fov")) + s_safe_step,
                                               s_safe_step):
                        for num_steps in range(round(t_clear_min/dt), round(t_clear_max/dt) + round(t_clear_step/dt),
                                               round(t_clear_step/dt)):
                            if vehicles_recapturing_data_nominal.get((delta_s, v_ego, v_cutin, a_ego)) is not None:
                                continue
                            print("v_ego: " + str(v_ego) + " | v_cutin: " + str(v_cutin) + " | delta_s: " + str(delta_s)
                                  + " | a_ego: " + str(a_ego) + " | num_steps: " + str(num_steps) + " | safe_distance: "
                                  + str(safe_distance))
                            try:
                                jerk = recapturing_control.calculate_input_nominal(a_ego, v_ego, v_cutin, 0,
                                                                                   delta_s, safe_distance, num_steps)
                            except:
                                print("No solution found")
                                continue
                            if jerk is None:
                                print("No solution found")
                                continue
                            print("Solution found")
                            s_ego, v_ego_tmp, a = 0, v_ego, a_ego
                            s_cutin, v_cutin_tmp = delta_s, v_cutin
                            for j in jerk:
                                s_ego, v, a = vehicle_dynamics_jerk(s_ego, v_ego_tmp, a, j, v_min_ego, v_max_ego,
                                                                    a_min_ego, a_max_ego, dt)

                                s_cutin, v_cutin_tmp = vehicle_dynamics_acc(s_cutin, v_cutin_tmp, a_cutin_min,
                                                                            v_min_cutin, v_max_cutin, dt)
                            safe_distance_real = \
                                safe_distance_profile_based(s_ego, v_ego_tmp, a, s_cutin, v_cutin_tmp, dt, t_react,
                                                            a_min_ego, a_max_ego, j_max_ego, v_min_ego, v_max_ego,
                                                            a_min_cutin, v_min_cutin, v_max_cutin, a_corr,
                                                            const_dist_offset, emergency_profile, emg_idx)
                            if s_ego > safe_distance_real:
                                vehicles_recapturing_data_nominal[(delta_s, v_ego, v_cutin, a_ego)] = \
                                    (num_steps, safe_distance_real)
                                break
                        else:
                            continue

    # run QP evaluation for nominal recapturing controller
    for v_ego in range(math.floor(v_min_ego), math.ceil(v_max_ego) + v_ego_step, v_ego_step):
        if v_ego > v_max_ego:
            v_ego = v_max_ego
        if v_ego < v_min_ego:
            v_ego = v_min_ego
        for v_cutin in range(math.floor(ego_vehicle_param.get("v_min")),
                             math.ceil(ego_vehicle_param.get("v_max")) + v_cutin_step, v_cutin_step):
            if v_cutin > v_max_cutin:
                v_cutin = v_max_cutin
            if v_cutin < v_min_cutin:
                v_cutin = v_min_cutin
            for delta_s in range(0, math.ceil(ego_vehicle_param.get("fov")) + delta_s_step, delta_s_step):
                if delta_s > fov:
                    delta_s = fov
                for a_ego in range(math.floor(ego_vehicle_param.get("a_min")),
                                   math.ceil(ego_vehicle_param.get("a_max")) + a_ego_step, a_ego_step):
                    if a_ego > a_max_ego:
                        a_ego = a_max_ego
                    if a_ego < a_min_ego:
                        a_ego = a_min_ego
                    for safe_distance in range(s_safe_step, math.ceil(ego_vehicle_param.get("fov")) + s_safe_step,
                                               s_safe_step):
                        for num_steps in range(round(t_clear_min / dt),
                                               round(t_clear_max / dt) + round(t_clear_step / dt),
                                               round(t_clear_step / dt)):
                            if vehicles_recapturing_data_acc_bounded.get((delta_s, v_ego, v_cutin, a_ego)) is not None:
                                continue
                            print("v_ego: " + str(v_ego) + " | v_cutin: " + str(v_cutin) + " | delta_s: " + str(delta_s)
                                  + " | a_ego: " + str(a_ego) + " | num_steps: " + str(num_steps) + " | safe_distance: "
                                  + str(safe_distance))
                            try:
                                jerk = recapturing_control.calculate_input_bounded(a_ego, v_ego, v_cutin, 0,
                                                                                   delta_s, safe_distance, num_steps)
                            except:
                                print("No solution found")
                                continue
                            if jerk is None:
                                print("No solution found")
                                continue
                            print("Solution found")
                            s_ego, v_ego_tmp, a = 0, v_ego, a_ego
                            s_cutin, v_cutin_tmp = delta_s, v_cutin
                            for j in jerk:
                                s_ego, v, a = vehicle_dynamics_jerk(s_ego, v_ego_tmp, a, j, v_min_ego, v_max_ego,
                                                                    a_min_ego, a_max_ego, dt)

                                s_cutin, v_cutin_tmp = vehicle_dynamics_acc(s_cutin, v_cutin_tmp, a_cutin_min,
                                                                            v_min_cutin, v_max_cutin, dt)
                            safe_distance_real = \
                                safe_distance_profile_based(s_ego, v_ego_tmp, a, s_cutin, v_cutin_tmp, dt, t_react,
                                                            a_min_ego, a_max_ego, j_max_ego, v_min_ego, v_max_ego,
                                                            a_min_cutin, v_min_cutin, v_max_cutin, a_corr,
                                                            const_dist_offset, emergency_profile, emg_idx)
                            if s_ego > safe_distance_real:
                                vehicles_recapturing_data_acc_bounded[(delta_s, v_ego, v_cutin, a_ego)] = \
                                    (num_steps, safe_distance_real)
                                break
                        else:
                            continue

    # Store clearance time and safe distance for recapturing controllers
    print("Storing solution")
    with open("recapturing_data_nominal" + ".pkl", "wb") as outfile:
        pickle.dump(vehicles_recapturing_data_nominal, outfile)
    with open("recapturing_data_acc_bounded" + ".pkl", "wb") as outfile:
        pickle.dump(vehicles_recapturing_data_acc_bounded, outfile)
    print("Finished")


if __name__ == "__main__":
    create_recapturing_data()
