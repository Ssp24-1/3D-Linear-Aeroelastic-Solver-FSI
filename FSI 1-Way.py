import FEM_Beam_Struct_Solver as FEM
import VLM_Pure_CFD as VLM
import Constants_VLM_FEM as CNST

import matplotlib.pyplot as plt
import numpy as np


# ================= Solving VLM (CFD) ===================== #
'''
Calculate total and panel-wise lift forces from VLM

'''
gamma_2D, panel_lift_forces = VLM.Solve_Gamma()

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

EA_nodes = np.column_stack([x_EA_nodes, y_EA_nodes, z_EA_nodes])    # Coordinates of nodes of elastic axis


# Nearest Neighbours Approach: #
'''
Distance measured to find the nearest node for each colloc. point. 
'''
col_x_plot = VLM.col_points[:, :, 0].flatten()
col_y_plot = VLM.col_points[:, :, 1].flatten()
col_z_plot = VLM.col_points[:, :, 2].flatten()

col_points_flatten = np.column_stack([col_x_plot, col_y_plot, col_z_plot])

col_index = np.zeros(np.size(col_points_flatten, axis = 0),)

for i in range(np.size(col_points_flatten, axis = 0)):
    temp = 10e11

    for j in range(np.size(EA_nodes, axis = 0)):
        dist_t = ((col_points_flatten[i][0] - EA_nodes[j][0])**2 + (col_points_flatten[i][1] - EA_nodes[j][1])**2 + (col_points_flatten[i][2] - EA_nodes[j][2])**2)**0.5

        if dist_t<temp:
            temp = dist_t
            index = j
    
    col_index[i] = index                # List of collocation point indices which correspond to respective node in structural beam model



# ================= Solving FEM Beam Model ===================== #
'''
Calculate all displacements of each DOF in structure beam model

'''
U_disp = FEM.Solve_Displacements(col_index, panel_lift_forces, VLM.M, VLM.N)


# ================== VLM Plotting ==================== #
# Create wireframe plot of geometric points
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract coordinates for bound vortex ring wireframe plotting
X_vr = VLM.vr_points[:, :, 0]  # X coordinates of bound vortex rings (M+1 x N+1)
Y_vr = VLM.vr_points[:, :, 1]  # Y coordinates 
Z_vr = VLM.vr_points[:, :, 2]  # Z coordinates

# Create wireframe plot for bound vortex rings (wing)
wireframe_vr = ax.plot_wireframe(X_vr, Y_vr, Z_vr, color='blue', linewidth=1.5, label='Bound Vortex Rings')

ax.plot(EA_nodes[:, 0], EA_nodes[:, 1], EA_nodes[:, 2], color='black', linewidth=2, label='Elastic Axis')


# # Extract coordinates for wake vortex ring wireframe plotting
# X_wake = VLM.wake_points[:, :, 0]  # X coordinates of wake vortex rings (M_w+1 x N+1)
# Y_wake = VLM.wake_points[:, :, 1]  # Y coordinates 
# Z_wake = VLM.wake_points[:, :, 2]  # Z coordinates

# # Create wireframe plot for wake vortex rings
# wireframe_wake = ax.plot_wireframe(X_wake, Y_wake, Z_wake, color='red', linewidth=1.5, label='Wake Vortex Rings')

# Plot collocation points for reference
col_x_plot = VLM.col_points[:, :, 0].flatten()
col_y_plot = VLM.col_points[:, :, 1].flatten()
col_z_plot = VLM.col_points[:, :, 2].flatten()
ax.scatter(col_x_plot, col_y_plot, col_z_plot, color='green', s=30, marker='x', label='Collocation Points')

# Plot panel lift forces as arrows
# Normalize lift forces for arrow length scaling
max_lift = np.max(np.abs(panel_lift_forces))
if max_lift > 0:  # Avoid division by zero
    arrow_scale = 0.005  # Adjust this scale factor as needed for visualization
    arrow_lengths = arrow_scale * panel_lift_forces / max_lift
    
    # Arrow directions (pointing upward in Z direction for lift forces)
    u = np.zeros_like(arrow_lengths)
    v = np.zeros_like(arrow_lengths)
    w = arrow_lengths
    
    # Plot arrows at each collocation point
    ax.quiver(col_x_plot, col_y_plot, col_z_plot, u, v, w, length=1, normalize=False, color='red', arrow_length_ratio=0.15, label='Panel Lift Forces')


# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title(f'Vortex Rings (AOA = {VLM.AOA*180/np.pi}°)\nBound Rings: M={VLM.M} x N={VLM.N}, Wake: M_w={VLM.M_w} x N={VLM.N}')

# Set viewing angle for better visualization of wing and wake
ax.view_init(elev=90, azim=0)

# Add legend
ax.legend()

# Add grid
ax.grid()
plt.show()

# ============== Plot Results =============== #
labels = ['Displacement (m)', 'Twist (θ)', 'Bend-Twist (β)']
colors = ['red', 'black', 'green']

fig, ax = plt.subplots(3, 1, sharex=True, num=1)
for i in range(3):

    ax[i].plot(FEM.y_nodes, U_disp[i::3], color = colors[i])
    ax[i].set_xlabel('Spanwise Position')
    ax[i].set_ylabel(labels[i])

plt.show()

