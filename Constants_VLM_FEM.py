import math
import numpy as np


# Flow Parameters
def Flow_parameters():

    U = 20             # Freestream Velocity
    U_vec = [U, 0, 0]  # Horizontal freestream (wing rotates instead of flow)
    density = 1.225
    AOA_deg = 3.0
    AOA = np.deg2rad(AOA_deg) # Angle of Attack (5 degrees in radians)
    sweep_deg = 15.0 # Sweep angle of the wing 
    sweep = np.deg2rad(sweep_deg) # Angle of Attack (5 degrees in radians)

    return U, U_vec, density, AOA, sweep_deg, sweep


# VLM Constants
def VLM_parameters():
    M = 10         # Number of chordwise panels
    N = 16         # Number of spanwise panels
    M_w = 100      # Number of chordwise wake panels

    return M, N, M_w


# FEM Constants
def TW_Wing_Constants():        # Tang Dowell Wing

  m = 0.2351  # Mass per unit Length [kg/m]
  Ip = 0.2056e-4  # Polar Moment of Inertia (Multiplied by Density per unit length) = I * (rho)
  x_cg_ea = -0.000508  # Distance between CG and EA
  EI = 0.4186  # Bending Stiffness, [Nm**2]
  GJ = 0.9539  # Torsional Stiffness, [Nm**2]
  KBT = 0.0*(EI*GJ)**0.5  # Bend Twist Coupling Coefficient (KBT**2 < EI*GJ), [Nm**2]
  chord = 0.0508 # Chord length of the wing (m)
  span = 0.4508  # Span length of the wing (m)

  return m, Ip, x_cg_ea, EI, GJ, KBT, chord, span

def Gol_Wing_Constants():     # Goland Wing

  m = 35.71  # Mass per unit Length [kg/m]
  Ip = 8.64  # Polar Moment of Inertia (Multiplied by Density per unit length) = I * (rho)
  x_cg_ea = -0.18288  # Distance between CG and EA
  EI = 9.77e6  # Bending Stiffness, [Nm**2]
  GJ = 0.99e6  # Torsional Stiffness, [Nm**2]
  KBT = -0.1*(EI*GJ)**0.5  # Bend Twist Coupling Coefficient (KBT**2 < EI*GJ), [Nm**2]
  chord = 1.8288 # Chord length of the wing (m)
  span = 6.096  # Span length of the wing (m)

  return m, Ip, x_cg_ea, EI, GJ, KBT, chord, span

def FEM_Prop():

  n_elem = 50 # Number of elements
  n_node = n_elem+1 # Number of Nodes (Number of elements + 1)
  n_dof = n_node*3  # Number of DOFs, here it is 3 dof per node (Out of plane Bending; Torsion; Bend-twist)

  return n_elem, n_node, n_dof