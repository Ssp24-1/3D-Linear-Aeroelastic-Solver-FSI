import FEM_Beam_Struct_Solver as FEM
import VLM_Pure_CFD as VLM
import Constants_VLM_FEM as CNST

import matplotlib.pyplot as plt
import numpy as np
import math


# ================= Solving VLM (CFD) ===================== #
'''
Solve For Gamma (Strength) 
'''
gamma = []
gamma = np.linalg.solve(VLM.A_matrix, VLM.RHS_vector)

# Convert 1D gamma to 2D array [M, N]
gamma_2D = []
for i in range(0, VLM.M):
    row = []
    for j in range(0, VLM.N):
        row.append(gamma[VLM.N*i + j])
    gamma_2D.append(row)

gamma_2D = np.array(gamma_2D)

'''
Secondary Computations - Lift Force and Lift Coefficient
'''

del_y_ij = VLM.span/VLM.N
total_lift = 0

# Create 1D array for net lift force of each panel (same indexing as gamma)
panel_lift_forces = np.zeros(VLM.M * VLM.N)

for i in range(0, VLM.M):  # Loop through vortex rings
    for j in range(0, VLM.N):
        vr_index = i * VLM.N + j  
        
        if i == 0:
            # First chordwise row: lift = ρ * U * Γ * Δy
            panel_lift = VLM.density * VLM.U * gamma_2D[i][j] * del_y_ij
            total_lift = total_lift + panel_lift
        else:
            # Other rows: lift = ρ * U * (Γ_current - Γ_upstream) * Δy
            panel_lift = VLM.density * VLM.U * (gamma_2D[i][j] - gamma_2D[i-1][j]) * del_y_ij
            total_lift = total_lift + panel_lift
        
        panel_lift_forces[vr_index] = panel_lift

print(f"Total Lift(N): {total_lift:.6f} N")

Cl = total_lift / (0.5 * VLM.density * VLM.chord * VLM.span * VLM.U**2)
print(f"Lift Coefficient: {Cl:.6f}")


col_x_plot = VLM.col_points[:, :, 0].flatten()
col_y_plot = VLM.col_points[:, :, 1].flatten()
col_z_plot = VLM.col_points[:, :, 2].flatten()

col_points_flatten = np.column_stack([col_x_plot, col_y_plot, col_z_plot])

# ================= End of Solving VLM (CFD) ===================== #



# ============ Mapping Aero. Colloc. Points to Structure Nodes ============= #
'''
Calculate the coordinates of Elastic Axis of the Wing (VLM).
It is then discretized based on the FEM model (Nodes).

Once the Elastic Axis is discretized, Nearest Neighbour approach is used to assign the individual panel lift forces, 
to the corresponding node on the beam.
'''

# End Points of the Wing: 

[x_LE_root, y_LE_root, z_LE_root] = VLM.geo_points[0][0]
[x_LE_tip, y_LE_tip, z_LE_tip] = VLM.geo_points[0][VLM.N]
[x_TE_root, y_TE_root, z_TE_root] = VLM.geo_points[VLM.M][0]
[x_TE_tip, y_TE_tip, z_TE_tip] = VLM.geo_points[VLM.M][VLM.N]


x_EA_root = x_LE_root + (x_TE_root - x_LE_root)*0.25
y_EA_root = y_LE_root + (y_TE_root - y_LE_root)*0.25            # EA placed at 0.25^chord
z_EA_root = z_LE_root + (z_TE_root - z_LE_root)*0.25

x_EA_tip = x_LE_tip + (x_TE_tip - x_LE_tip)*0.25
y_EA_tip = y_LE_tip + (y_TE_tip - y_LE_tip)*0.25                # EA placed at 0.25^chord
z_EA_tip = z_LE_tip + (z_TE_tip - z_LE_tip)*0.25

x_EA_nodes = np.linspace(x_EA_root, x_EA_tip, FEM.n_node)
y_EA_nodes = np.linspace(y_EA_root, y_EA_tip, FEM.n_node)       # Discretized EA with FEM Elements
z_EA_nodes = np.linspace(z_EA_root, z_EA_tip, FEM.n_node)

EA_nodes = np.column_stack([x_EA_nodes, y_EA_nodes, z_EA_nodes])

print(EA_nodes)

# Nearest Neighbours Approach: 
'''
Distance measured to find the nearest node for each colloc. point. 
'''
col_index = np.zeros(np.size(col_points_flatten, axis = 0),)

for i in range(np.size(col_points_flatten, axis = 0)):
    temp = 10e11

    for j in range(np.size(EA_nodes, axis = 0)):
        dist_t = ((col_points_flatten[i][0] - EA_nodes[j][0])**2 + (col_points_flatten[i][1] - EA_nodes[j][1])**2 + (col_points_flatten[i][2] - EA_nodes[j][2])**2)**0.5

        if dist_t<temp:
            temp = dist_t
            index = j
    
    col_index[i] = index



# ================= Solving FEM Beam Model ===================== #

# ============== Load Vector [F] =============== #
'''
3 Types of Loads can be assigned in this case, since we have 3 independent DOFs per node:

  1. Out of Plane Vector Force
  2. Torque
  3. Bending Moment Distribution

'''
force = np.zeros((FEM.n_node, ))
torque = np.zeros((FEM.n_node, ))
moment = np.zeros((FEM.n_node, ))

for i in range(VLM.M * VLM.N):
    force[int(col_index[i])] += panel_lift_forces[i]        # Assign Panel Lift Force to corresponding FEM Node based on Mapping

# sum = 0
# for i in range(FEM.n_node):
#     sum += force[i]

load_vector = np.zeros((FEM.n_dof, ))

load_vector[0::3] = force
load_vector[1::3] = torque
load_vector[2::3] = moment

load_vector_red = np.delete(load_vector, FEM.dof_clamped, axis=0)       # Reduced load vector by removing the known DOF entries


# ============== Solve For Displacements =============== #
'''
    [K].{u} = [F]
    {u} = [K]^{-1} [F]
'''

u = np.linalg.solve(FEM.K, load_vector_red)                 # Solve for all DOF values

full_u = np.zeros((FEM.n_dof,))                             # Re-assemble the full DOF vector (Add the removed DOFS back)
full_u[3:] = u


# ============== Plot Results =============== #
labels = ['Displacement (m)', 'Twist (θ)', 'Bend-Twist (β)']
colors = ['red', 'black', 'green']

fig, ax = plt.subplots(3, 1, sharex=True, num=1)
for i in range(3):

    ax[i].plot(FEM.y_nodes, full_u[i::3], color = colors[i])
    ax[i].set_xlabel('Spanwise Position')
    ax[i].set_ylabel(labels[i])

plt.show()


# # =========================== (Plotting) ========================= #

# Create wireframe plot of geometric points
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract coordinates for bound vortex ring wireframe plotting
X_vr = VLM.vr_points[:, :, 0]  # X coordinates of bound vortex rings (M+1 x N+1)
Y_vr = VLM.vr_points[:, :, 1]  # Y coordinates 
Z_vr = VLM.vr_points[:, :, 2]  # Z coordinates

# Create wireframe plot for bound vortex rings (wing)
wireframe_vr = ax.plot_wireframe(X_vr, Y_vr, Z_vr, color='blue', alpha=0.8, linewidth=1.5, label='Bound Vortex Rings')

ax.plot(EA_nodes[:, 0], EA_nodes[:, 1], EA_nodes[:, 2], color='black', linewidth=2, label='Elastic Axis')


# # Extract coordinates for wake vortex ring wireframe plotting
# X_wake = VLM.wake_points[:, :, 0]  # X coordinates of wake vortex rings (M_w+1 x N+1)
# Y_wake = VLM.wake_points[:, :, 1]  # Y coordinates 
# Z_wake = VLM.wake_points[:, :, 2]  # Z coordinates

# # Create wireframe plot for wake vortex rings
# wireframe_wake = ax.plot_wireframe(X_wake, Y_wake, Z_wake, color='red', alpha=0.6, linewidth=1.5, label='Wake Vortex Rings')

# # Plot collocation points for reference
# col_x_plot = VLM.col_points[:, :, 0].flatten()
# col_y_plot = VLM.col_points[:, :, 1].flatten()
# col_z_plot = VLM.col_points[:, :, 2].flatten()
# ax.scatter(col_x_plot, col_y_plot, col_z_plot, color='green', s=30, marker='x', label='Collocation Points')

# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
# ax.set_title(f'Vortex Ring System (AOA = {CNST.AOA*np.pi/180}°)\nBound Rings: M={CNST.M} x N={CNST.N}, Wake: M_w={CNST.M_w} x N={CNST.N}')

# Set viewing angle for better visualization of wing and wake
ax.view_init(elev=90, azim=0)

# Add legend
ax.legend()

# Add grid
ax.grid(True, alpha=0.3)

# Show the plot
plt.show()