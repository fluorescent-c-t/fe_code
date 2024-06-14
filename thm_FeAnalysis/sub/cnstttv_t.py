#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import sys
import numpy as np
from sub import shapefunction as sf
from sub import global_variable as gl

def intrpl_gp(T):
    T_gp = np.zeros((gl.num_element, gl.num_gp))
    for i in range(gl.num_element):
        T_ele = np.zeros((gl.num_node_quad1, 1))
        T_ele[0] = T[gl.conn[i][0] - 1]
        T_ele[1] = T[gl.conn[i][1] - 1]
        T_ele[2] = T[gl.conn[i][2] - 1]
        T_ele[3] = T[gl.conn[i][3] - 1]
        for j in range(gl.num_gp):
            xi  = gl.normcoord_gp[j][0]
            eta = gl.normcoord_gp[j][1]
            N2  = sf.mtrx_1dof(xi, eta)
            
            T_gp[i] = np.dot(N2, T_ele)
    
    return T_gp

def matrcs_ele(detJ, B2, i, T_gp):
    if gl.act_latentheat == 0:#Latent heat (de)activation
        if gl.thrm_modeltype[i] == 1:
            print('')
            print("Notice: The latent heat mode should be activated if the thermo-model type: 1")
            print("Notice: Activate the latent heat mode: go to setting file")
            print("Notice: The program was terminated")
            sys.exit()
            
    elif gl.act_latentheat == 1:
        L = gl.Lw
        if gl.thrm_modeltype[i] == 0:
            print('')
            print("Notice: This mode is not compatible with the current thermo-model type: 0")
            print("Notice: Change the thermo-model type from 0 to 1")
            print("Notice: The program was terminated")
            sys.exit()
    
    rho_s = gl.cmmn_matparam[i][1]#Read material parameters
    if gl.thrm_modeltype[i] == 0:
        kTxx = gl.thrm_matparam[i][0]
        kTzz = gl.thrm_matparam[i][1]
        Cv   = gl.thrm_matparam[i][1]#[kJ/m3/K]
        
    elif gl.thrm_modeltype[i] == 1:
        kTxx_s = gl.thrm_matparam[i][0]
        kTzz_s = gl.thrm_matparam[i][1]
        cs     = gl.thrm_matparam[i][1]#[kJ/kg/K]
        
    K3e = np.zeros((gl.dof_t * gl.num_node_quad1, gl.dof_t * gl.num_node_quad1))
    ce  = np.zeros((gl.dof_t * gl.num_node_quad1, gl.dof_t * gl.num_node_quad1))
    for j in range(gl.num_gp):
        if gl.thrm_modeltype[i] == 1:#Calulate the state dependent thermal properties
            T = T_gp[i][j]
            if gl.T_unfr <= T:#if unfrozen
                rhow = gl.rhow_unfr
                kTw  = gl.kTw_unfr
                cw   = gl.cw_unfr
                
            elif gl.T_fr < T < gl.T_unfr:#if in frozen process
                rhow = gl.rhow_fr + (gl.rhow_unfr - gl.rhow_fr) / (gl.T_unfr - gl.T_fr) * (T - gl.T_fr)
                kTw  = gl.kTw_fr  + (gl.kTw_unfr  - gl.kTw_fr ) / (gl.T_unfr - gl.T_fr) * (T - gl.T_fr)
                cw   = gl.cw_fr   + (gl.cw_unfr   - gl.cw_fr  ) / (gl.T_unfr - gl.T_fr) * (T - gl.T_fr) + L / (gl.T_unfr - gl.T_fr)
            
            else:#if frozen
                rhow = gl.rhow_unfr
                kTw  = gl.kTw_unfr
                cw   = gl.cw_unfr
                
            e = gl.voidratio_gp[i][j]#Read state variables
            n = e / (1. + e)
            Gs = rho_s / rhow
            c_sat   = (Gs * cs + e * cw) / (Gs + e)
            rho_sat = rhow * (e + Gs) / (1. + e)
            
            Cv   = c_sat * rho_sat
            kTxx = kTxx_s ** (1. - n) * kTw ** n
            kTzz = kTzz_s ** (1. - n) * kTw ** n
        
        else:
            pass
        
        dim = 2 #dimension of analysis
        kT       = np.zeros((dim, dim))
        kT[0][0] = kTxx
        kT[1][1] = kTzz
        K3e += gl.weight_gi[j] * B2[i][j].T @ kT @ B2[i][j] * detJ[i][j]
        
        xi  = gl.normcoord_gp[j][0]
        eta = gl.normcoord_gp[j][1]
        N2  = sf.mtrx_1dof(xi, eta)
        N2T = np.reshape(N2, [4, 1])
        
        ce += Cv * gl.weight_gi[j] * N2T * N2 * detJ[i][j]
    
    return K3e, ce