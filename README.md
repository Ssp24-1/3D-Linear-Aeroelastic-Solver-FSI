This repository contains a 1 way coupled 3D CFD(Vortex Lattice Method) and 1D Beam Model. 

# ============================================ #

-> To run the FSI program, download all the python files and store in a single folder. 
-> Once downloaded, run the 1-way FSI.py folder. 
-> Changes to the wing geometry, flow parameters, FEM and VLM parameters can be done in the Constants_VLM_FEM.py file. 

# ============================================ #

Wing Geometries for common, experiemntally studied wings are already included in the Constants_VLM_FEM file:
      1. Tang-Dowell
      2. Goland

For detailed explainations on procedure, methodologies and coupling method, a comprehensive report is also added. 

Future Updates to make code more comprehensive and accurate will be added as completed. 
Current lineup of future updates to the code are:

Features in progress to be added as future updates: 

1. Change 1D Beam Model to 2D Flat Plate Model.
2. Improve Coupling Strategy (Mapping) from Nearest Neighbour
      a. Update Nearest Neighbour to RBF
      b. Sub-iterations
3. Improve VLM code to include Viscous Effects, effect of sweep etc. 
4. Pseudo 2-way FSI with Single Iteration only from VLM -> FEM -> VLM.
5. Complete 2-way FSI model setup 
                  .
                  .
                  .
   
