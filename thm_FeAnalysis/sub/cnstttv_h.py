#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import os
import datetime
import sys
import numpy as np

from sub import global_variable as gl
from sub import shapefunction as sf


myu = lambda T: 2.414 * 1e-5 * 10 ** (247.8 / (T + 273.15 - 140.))#Viscosity of water

def matrcs_ele(detJ, B2, Bv, i, T_gp):
    gamma_w = gl.g * gl.rhow_unfr / 1000. #[kN/m3]
    if gl.hydr_modeltype[i] == 0:
        kxx = gl.hydr_matparam[i][0]
        kzz = gl.hydr_matparam[i][1]
        
    elif gl.hydr_modeltype[i] in {1,2}:
        e0      = gl.cmmn_matparam[i][0]
        kxx0    = gl.hydr_matparam[i][0]
        kzz0    = gl.hydr_matparam[i][1]
        lmd_kxx = gl.hydr_matparam[i][2] / 2.303
        lmd_kzz = gl.hydr_matparam[i][3] / 2.303
        
    K2e = np.zeros((gl.dof_h * gl.num_node_quad1, gl.dof_h * gl.num_node_quad1))
    HMe = np.zeros((gl.dof_m * gl.num_node_quad1, gl.dof_h * gl.num_node_quad1))
    for j in range(gl.num_gp):
        if gl.hydr_modeltype[i] in {1,2}:
            e = gl.voidratio_gp[i][j]
            kxx = kxx0 * np.exp((e - e0) / lmd_kxx)
            kzz = kzz0 * np.exp((e - e0) / lmd_kzz)
            if gl.hydr_modeltype[i] == 2:
                T    = T_gp[i][j]
                Tref = gl.mech_anchrval[i][2]
                
                kxx_ref = kxx
                kzz_ref = kzz
                
                kxx  = kxx_ref * myu(Tref) / myu(T)
                kzz  = kzz_ref * myu(Tref) / myu(T)
            
        dim = 2 #dimension of analysis
        h       = np.zeros((dim, dim))#constitutive matrix
        h[0][0] = kxx / gamma_w
        h[1][1] = kzz / gamma_w
        K2e += gl.weight_gi[j] * B2[i][j].T @ h @ B2[i][j] * detJ[i][j]
        
        xi  = gl.normcoord_gp[j][0]
        eta = gl.normcoord_gp[j][1]
        N2  = sf.mtrx_1dof(xi, eta)
        BvT = np.reshape(Bv[i][j], (8, 1))
        HMe += - gl.weight_gi[j] * BvT @ N2 * detJ[i][j]
    
    return K2e, HMe
