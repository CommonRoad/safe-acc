from common.configuration import *
from acc.recapturing_control import RecapturingControl
import pickle
from common.quadratic_program import QP


def create_recapturing_controllers():
    """
    Creates recapturing controller objects for different time horizons
    """
    # Initialization of variables for simulation
    print("Initialization")
    config = load_yaml("config.yaml")
    simulation_param = config.get("simulation_param")
    acc_param = create_acc_param(config.get("acc_param"), config.get("ego_vehicle_param").get("j_min"))
    other_vehicles_param = config.get("other_vehicles_param")
    ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), simulation_param, acc_param)
    cutin_config_param = acc_param.get("cutin")

    recapturing_control = RecapturingControl(simulation_param, cutin_config_param, ego_vehicle_param,
                                             other_vehicles_param, acc_param.get("emergency"), acc_param.get("common"),
                                             {}, [{}, {}], [{}, {}])

    # Initialization of vehicle variables
    dt = simulation_param.get("dt")
    num_states = 3
    t_clear_max = acc_param.get("cutin").get("t_clear_max")
    a_d, b_d, q, r = recapturing_control.motion_equations(dt, cutin_config_param.get("cost_s"),
                                                          cutin_config_param.get("cost_v"),
                                                          cutin_config_param.get("cost_a"),
                                                          cutin_config_param.get("cost_j"))

    # Create controllers
    print("Start controller generation")
    qp_nominal_dict = {}
    qp_acc_bounded_dict = {}
    for num_steps in range(1, round(t_clear_max / dt) + 1):
        print("Generated controller with time horizon of " + str(num_steps) + " time steps")
        qp_nominal = QP(num_states, a_d, b_d, q, r, num_steps, cutin_config_param.get("solver"))
        qp_nominal.create_constraint_matrices(ego_vehicle_param.get("j_min"), ego_vehicle_param.get("j_max"),
                                              recapturing_control.state_constraint_format())
        qp_acc_bounded = QP(num_states, a_d, b_d, q, r, num_steps, cutin_config_param.get("solver"))
        qp_acc_bounded.create_constraint_matrices(None, None, recapturing_control.state_constraint_format())
        qp_nominal_dict[num_steps] = qp_nominal
        qp_acc_bounded_dict[num_steps] = qp_acc_bounded

    recapturing_controllers = [qp_nominal_dict, qp_acc_bounded_dict]

    # Store recapturing controllers
    print("Store controllers")
    with open("recapturing_controllers" + ".pkl", "wb") as outfile:
        pickle.dump(recapturing_controllers, outfile)
    print("finished")


if __name__ == "__main__":
    create_recapturing_controllers()
