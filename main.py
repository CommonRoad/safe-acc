import pickle
import os


from common.simulation import Simulation
from common.configuration import load_yaml, create_acc_param, create_ego_vehicle_param
from output.visualization import create_scenario_video, plot_figures, create_profile_videos, get_date_and_time


def main():
    """
    Main function for the execution of the safe ACC
    """
    # Initialization of variables for simulation
    config = load_yaml("config.yaml")
    simulation_param = config.get("simulation_param")
    acc_param = create_acc_param(config.get("acc_param"), config.get("ego_vehicle_param").get("j_min"))
    nominal_acc = acc_param.get("nominal_acc") # Added
    other_vehicles_param = config.get("other_vehicles_param")
    ego_vehicle_param = create_ego_vehicle_param(config.get("ego_vehicle_param"), simulation_param, acc_param)
    input_constr_param = config.get("input_constr_param")
    visualization_param = config.get("visualization")
    road_network_param = config.get("road_network_param")

    # Import controller LUTs
    with open("./acc/recapturing_controller_data/recapturing_data_nominal" + ".pkl", 'rb') as file:
        recapturing_data_nominal = pickle.load(file)
    with open("./acc/recapturing_controller_data/recapturing_data_acc_bounded" + ".pkl", 'rb') as file:
        recapturing_data_acc_bounded = pickle.load(file)
    with open("./acc/recapturing_controller_data/recapturing_controllers" + ".pkl", 'rb') as file:
        recapturing_controllers = pickle.load(file)

    # Simulate scenario until end of time horizon or goal region is reached
    sim = Simulation(simulation_param, road_network_param, acc_param, ego_vehicle_param, other_vehicles_param,
                     input_constr_param, recapturing_data_nominal, recapturing_data_acc_bounded,
                     recapturing_controllers)

    sim.simulate(nominal_acc)
    #print("nominal_acc:", nominal_acc.get("distance"))
    sim.get_plots()
    #print("sim._ego_vehicle[0]._safe_distance_list:", sim._ego_vehicle[1]._safe_distance_list)

    # take out leading vehicle A as dynamical obstacle
    #ego_obstacle = sim.scenario.dynamic_obstacles[-len(sim._planning_problem)]
    #ego_obstacle = sim.scenario.dynamic_obstacles[0]
    ego_obstacle = None
    # Generate output (video/plots)
    if simulation_param.get("commonroad_video_creation") is True or simulation_param.get("store_profiles") is True \
            or simulation_param.get("profile_video_creation") is True:
        date_time = get_date_and_time()
    else:
        date_time = ""
    if simulation_param.get("commonroad_video_creation") is True \
            or simulation_param.get("profile_video_creation") is True:

        path = "./" + simulation_param.get("video_output_folder") + "/" \
               + simulation_param.get("commonroad_benchmark_id") + "_" + date_time
        if not os.path.exists("./" + simulation_param.get("video_output_folder")):
            os.mkdir("./" + simulation_param.get("video_output_folder"))
        if not os.path.exists(path):
            os.mkdir(path) # create the path.
    else:
        path = "./"
    if simulation_param.get("commonroad_video_creation") is True:
        print("Create commonroad video")
        create_scenario_video(out_path=path, scenario=sim.scenario,
                              ego_obstacle=ego_obstacle, visualization_param=visualization_param.get("video"))
    if simulation_param.get("profile_video_creation") is True:
        print("Create profile videos")
        create_profile_videos(path, sim.ego_vehicle, sim.obstacles,
                              ego_vehicle_param, other_vehicles_param, simulation_param)
    if simulation_param.get("plotting_profiles") is True:
        if simulation_param.get("store_profiles"):
            path = "./" + simulation_param.get("figure_output_folder") + "/" \
                   + simulation_param.get("commonroad_benchmark_id") + "_" + date_time
            if not os.path.exists("./" + simulation_param.get("figure_output_folder")):
                os.mkdir("./" + simulation_param.get("figure_output_folder"))
            if not os.path.exists(path):
                os.mkdir(path)
        plot_figures(sim.ego_vehicle, sim.obstacles, sim.safe_acc.emergency_maneuver_activity,
                     ego_vehicle_param, simulation_param, visualization_param.get("plot"), sim.comp_time,
                     sim.safe_acc.strategy_list, sim.safe_acc.num_vehicles, path)


if __name__ == "__main__":
    main()
