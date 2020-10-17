## Provably-Correct and Comfortable Adaptive Cruise Control

##### Getting Started
This introduction will give you an overview how to install, parametrize, 
and execute the provably-correct and comfortable Adaptive Cruise Control (safe ACC) by using CommonRoad scenarios 
and the corresponding CommonRoad tools. 

We recommend Ubuntu 18.04 as operating system and to usage the Anaconda Python distribution. 
For the execution of the ACC simulation you need at least Python 3.6 
and the following packages:
* *matplotlib* >= 2.5.0
* *numpy* >= 3.1.0
* *ruamel.yaml* >= 1.3.1
* *qpsolvers* >= 1.0.7
* *commonroad-io* >= 2020.2
* *commonroad-vehicle-models* >= 1.0.0

You can install the required Python packages with the provided 
*requirements.txt* file (*pip install -r requirements.txt*). 

Additionally, you need the following software:
* *[CommonRoad Drivability Checker](https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker)* 
which has to be installed according to linked installation instructions inside the repository.
* Software for the transformation from a Cartesian into a Curvilinear coordinate system and back. 
Unfortunately, we cannot provide you code for this at the moment. 
We added `TODO` to the code parts which require a conversion.

##### Running the safe ACC using CommonRoad
The main files for the execution of the safe ACC are *config.yaml* and *main.py*:

**1. config.yaml:** This file allows to adapt all parameters used by 
the ACC system, e.g. ACC vehicle parameters, simulation parameters, other vehicle parameters, 
or parameters of the provably-correct and comfortable ACC. 
The different parameters are described within the file.

**2. main.py:**  This file starts the simulation, e.g., by executing it within the command line (*python main.py*).

The folder *./scenarios* contains CommonRoad highway scenarios. 

To generate the data structures for the recapturing controller offline execute the files 
*create_recapturing_controllers.py* and *create_recapturing_data.py*. 
The repository contains already default recapturing controllers which can be used.

**If you use our code for research, please consider citing our paper:**
```
@article{althoff2020,
	author = "Matthias Althoff, Sebastian Maierhofer, and Christian Pek",
	title = "Provably-Correct and Comfortable Adaptive Cruise Control",
	journal = "IEEE Transactions on Intelligent Vehicles",
	year = "2020",
	abstract = "Adaptive cruise control is one of the most common comfort features of road vehicles. Despite its large 
                   market penetration, current systems are not safe in all driving conditions and require supervision by 
                   human drivers. While several previous works have proposed solutions for safe adaptive cruise control, 
                   none of these works considers comfort, especially in the event of cut-ins. We provide a novel solution 
                   that simultaneously meets our specifications and provides comfort in all driving conditions including 
                   cut-ins. This is achieved by an exchangeable nominal controller ensuring comfort combined with a 
                   provably correct fail-safe controller that gradually engages an emergency  maneuver—this ensures 
                   comfort, since most threats are already cleared before emergency braking is fully activated. As a 
                   consequence, one can easily exchange the nominal controller without having to re-certify the overall 
                   system  safety. We also provide the first user study for a provably correct adaptive cruise controller. 
                   It shows that even though our approach never causes an accident, passengers rate the performance as 
                   good as a state-of-the-art solution  that does not ensure safety.",
	doi = "10.1109/TIV.2020.2991953",
}
```