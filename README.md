Submission for PJDSC 2025 - Knight to D4

AGOS: Aid & Goods Optimization System

For the application to work, the dashboard.py and solver2.py need to be located in the same working folder.
To run the dashboard, in terminal, type:
  streamlit run dashboard.py 

Dashboard Walkthrough

WARNING: USING THE CURRENT MAP AND NODES PROVIDED, IT TAKES A SIGNIFICANT AMOUNT OF TIME TO RUN THE OUTPUT (>4 HOURS)
Therefore it is advisable to use the sample map built into the code to test an output

--- Side Panel ---

On the side panel, 3 settings will be presented to the user

## Setting 1: Map

The user may use a sample map that was used during testing, which provides 2 shelter nodes, 1 depot node, and 2 connector nodes. From here, the user may immedately move on to the next settings should they choose to use this map. 

Otherwise, the user may upload two csv files or manually input the details of the graph:
 - "nodes" csv file with columns: 
  
   id,type,lat,lon,demand      where "id" is a unique integer for every node;
                                     "type" is either depot or shelter;
                                     "lat" and "lon" are floats and coordinates of the node;
                                     "demand" is the demand for a node (important, nodes with the "type" node should have a value of 0 in the                                                  demand column)
   
 - "edges" csv file with columns:

   u,v,weight,max_capacity,time,damage     where "u" and "v" are connected nodes and correspond to the id's in the nodes csv;
                                                 "weight" is the distance in km between u and v;
                                                 "max_capacity" is the maximum number of supported vehicles on that road(edge);
                                                 "time" is the time it takes in min to travel between u and v;
                                                 "damage" is an integer from 0-10 to indicate road damage

## Setting 2: Vehicles

The user manually assigns the number of vehicles, the capacity of each vehicle and the id of the node where each vehicle starts

## Setting 3: Heuristics

The user selects the type of heuristic based on the paper from 

    Anuar, W. K., Lee, L. S., Seow, H.-V., & Pickl, S. (2022). A Multi-Depot Dynamic Vehicle Routing Problem with Stochastic Road Capacity: An       MDP Model and Dynamic Policy for Post-Decision State Rollout Algorithm in Reinforcement Learning.

The user can set the number of simulations that the selected heuristic will run to check the most optimal solution amongst the number of simulations.

The user can also set the number of "look-ahead" that the algorithm runs per step for the algorithm to "look ahead" n number of steps before committing to a solution. 

The user is then able to run the simulation to output a solution


-- Map --

In the middle of the dasbaord, a map of the nodes and the edges connecting them is presented. The user will see a bar labeled steps which, when dragged, will display the most optimal output computed by the algorthm at that step. A hard limit of 1000 steps has been set to eventually end the program.

