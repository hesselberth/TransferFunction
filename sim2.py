#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 19:38:59 2026

@author: Marcel Hesselberth
"""

# Backward Euler integrator for simple circuits with linear components,
# switched and constant voltage diodes.

import numpy as np

# 1. Component Parameters
L, C, R_bat = 100e-6, 470e-6, 0.05
V_in, V_bat, V_d = 12.0, 3.7, 0.7
f_sw, duty = 100e3, 0.4
T_sw = 1/f_sw
h = 1e-8  # Timestep (100ns)
total_time = 0.002

# 2. Matrices
A = np.array([[0, -1/L], [1/C, -1/(R_bat*C)]])
A_dcm = np.array([[0, 0], [0, -1/(R_bat*C)]]) # Forced dIL/dt = 0

B_on = np.array([[1/L, 0, 0], [0, 1/(R_bat*C), 0]])
B_off = np.array([[0, 0, -1/L], [0, 1/(R_bat*C), 0]])
B_dcm = np.array([[0, 0, 0], [0, 1/(R_bat*C), 0]])

# Pre-calculate Identity and Matrix Inverses for Speed
I = np.eye(2)
M_inv = np.linalg.inv(I - h * A)
M_inv_dcm = np.linalg.inv(I - h * A_dcm)

# 3. Solver Setup
u = np.array([V_in, V_bat, V_d])
x = np.array([0.0, V_bat]) # Initial [IL, VC]
t = 0.0
results = []

# 4. Simulation Loop
while t < total_time:
    pwm_on = (t % T_sw) < (duty * T_sw)
    
    if pwm_on:
        # Standard Switch-ON Step
        x = M_inv @ (x + h * B_on @ u)
    else:
        # Try Switch-OFF Step
        x_next = M_inv @ (x + h * B_off @ u)
        
        if x_next[0] < 0: # Diode blocks: Inductor current hit zero
            # Precise Split: Find fraction of step until current was 0
            # dt0 = h * (IL_prev / (IL_prev - IL_next))
            dt0 = h * (x[0] / (x[0] - x_next[0]))
            
            # Step 1: Sub-step to the zero-crossing (Conducting)
            M_sub = np.linalg.inv(I - dt0 * A)
            x = M_sub @ (x + dt0 * B_off @ u)
            x[0] = 0.0 # Force clean zero
            
            # Step 2: Sub-step remaining time in DCM (Blocked)
            dt_rem = h - dt0
            M_rem = np.linalg.inv(I - dt_rem * A_dcm)
            x = M_rem @ (x + dt_rem * B_dcm @ u)
        else:
            x = x_next
            
    results.append([t, x[0], x[1]])
    t += h

# results contains [time, IL, VC]
print(results)