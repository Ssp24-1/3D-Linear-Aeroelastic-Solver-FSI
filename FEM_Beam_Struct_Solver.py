import numpy as np
import matplotlib.pyplot as plt
import math

import Constants_VLM_FEM as CNST


## ============== Global Stiffness Matrix ================ #
'''
Global Stiffness Matrix is built for each element, for node (i) and node (i+1)

For this matrix, the DOFs are arranged in the order:

# ===================== #
v_i : Out of Plane Bending - (Node i)
theta_i : Torsion - (Node i)
beta_i : Bend-Twist - Coupling (Node i)
# = = = = = = = = = = = #
v_i+1 : Out of Plane Bending - (Node i+1)
theta_i+1 : Torsion - (Node i+1)
beta_i+1 : Bend-Twist Coupling - (Node i+1)

'''

def Global_Stiffness_Matrix(EI, GJ, KBT, elem_length, dof_clamped):

    '''
    To build the global stiffness matrix, each elements stiffness matrix should be built first,
    and then added to the stiffness matrix is a diagonal format, with overlapping nodes'
    contributions summed up.
    '''
    # =========== Element Stiffness Matrix =========== #

    K_elem = np.zeros((6, 6))

    T1 = 12*EI/elem_length**3
    T2 = 6*EI/elem_length**2
    T3 = 4*EI/elem_length
    T4 = 2*EI/elem_length

    T5 = GJ/elem_length
    T6 = KBT/elem_length

    K_elem[0, 0] = K_elem[3, 3] = T1
    K_elem[0, 2] = K_elem[2, 0] = K_elem[0, 5] = K_elem[5, 0] = T2
    K_elem[2, 2] = K_elem[5, 5] = T3
    K_elem[2, 5] = K_elem[5, 2] = T4

    K_elem[0, 3] = K_elem[3, 0] = -T1
    K_elem[2, 3] = K_elem[3, 2] = K_elem[3, 5] = K_elem[5, 3] = -T2


    K_elem[1, 1] = K_elem[4, 4] = T5
    K_elem[1, 2] = K_elem[2, 1] = K_elem[4, 5] = K_elem[5, 4] = T6

    K_elem[1, 4] = K_elem[4, 1] = -T5
    K_elem[1, 5] = K_elem[5, 1] = K_elem[2, 4] = K_elem[4, 2] = -T6


    # ====== Assembling Global Stiffness Matrix ======= #

    '''
    Since each element has 2 nodes, and apart from the end nodes, the nodes are shared with adjacant elements,
    the contributions of the shared nodes should be summed up.

    Hence element matrix will be added in a diagonal format every 3 indices. (3DOF)

    '''

    K_Full = np.zeros((n_dof, n_dof))

    for i in range(n_elem):
        K_Full[3*i:3*(i + 2), 3*i:3*(i + 2)] += K_elem
        

    K_Red = np.delete(K_Full, dof_clamped, axis=0)  # remove rows of known DOF (1st node)
    K_Red = np.delete(K_Red, dof_clamped, axis=1)  # remove columns of known DOF (1st node)

    '''
    Having the reduced K with the cantilever beam approach also prevernts singularity in the matrix if all DOF's were kept.
    '''

    return K_Red


def Solve_Displacements(col_index, panel_lift_forces, M, N):

    force = np.zeros((n_node, ))
    torque = np.zeros((n_node, ))
    moment = np.zeros((n_node, ))

    for i in range(M * N):
        force[int(col_index[i])] += panel_lift_forces[i]        # Assign Panel Lift Force to corresponding FEM Node based on Mapping

    # sum = 0
    # for i in range(FEM.n_node):
    #     sum += force[i]

    load_vector = np.zeros((n_dof, ))

    load_vector[0::3] = force
    load_vector[1::3] = torque
    load_vector[2::3] = moment

    load_vector_red = np.delete(load_vector, dof_clamped, axis=0)       # Reduced load vector by removing the known DOF entries


    # ============== Solve For Displacements =============== #
    '''
        [K].{u} = [F]
        {u} = [K]⁻¹ [F]
    '''

    u = np.linalg.solve(K, load_vector_red)                 # Solve for all DOF values

    U_disp = np.zeros((n_dof,))                             
    U_disp[3:] = u

    return U_disp


# ============== Main =============== #

'''
Assumptions taken here:

1. Uniform Beam
2. Cantilever Beam to match nominal wing - Wing clamped at 1st node.

'''
m, Ip, x_cg_ea, EI, GJ, KBT, chord, span = CNST.TW_Wing_Constants()

n_elem, n_node, n_dof = CNST.FEM_Prop()

# ========= Node and Element Geometric Coordinates =========== #
y_nodes = np.linspace(0, span, n_node)
elem_length = np.diff(y_nodes)


# ============== Boundary Condition ================== #
'''
"Cantilever Beam": hence 1st node is fully constrained, which means the 1st 3 DOFs are fixed.
Enter index of DOFs that are clamped.

'''
dof_clamped = [0, 1, 2]

# ============== Global Stiffness Matrix [K] =============== #

K = Global_Stiffness_Matrix(EI, GJ, KBT, elem_length[0], dof_clamped)        # Final Full Global Stiffness Matrix


