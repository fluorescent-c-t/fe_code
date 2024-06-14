#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import numpy as np
from sub import global_variable as gl

n1      = lambda xi, eta: (1. - xi) * (1. - eta) / 4.
n2      = lambda xi, eta: (1. + xi) * (1. - eta) / 4.
n3      = lambda xi, eta: (1. + xi) * (1. + eta) / 4.
n4      = lambda xi, eta: (1. - xi) * (1. + eta) / 4.

dn1dxi  = lambda eta: - (1. - eta) / 4.
dn2dxi  = lambda eta:   (1. - eta) / 4.
dn3dxi  = lambda eta:   (1. + eta) / 4.
dn4dxi  = lambda eta: - (1. + eta) / 4.
dn1deta = lambda xi : - (1. -  xi) / 4.
dn2deta = lambda xi : - (1. +  xi) / 4.
dn3deta = lambda xi :   (1. +  xi) / 4.
dn4deta = lambda xi :   (1. -  xi) / 4.

dim                   = 2 #dimension of analysis, used here only;so, hardcoded here
dim3                  = 3 #full expression for spatial dimension, 3d
num_sf                = 4
num_strain_vctr_cmpnt = 4

def mtrcs_deriv_gp():
    dndxi_dndeta = np.zeros((gl.num_gp, dim, num_sf))
    
    J            = np.zeros((gl.num_element, gl.num_gp,                   dim,           dim))
    detJ         = np.zeros((gl.num_element, gl.num_gp                                      ))
    dndx_dndz    = np.zeros((gl.num_element, gl.num_gp,                   dim,        num_sf))
    B1           = np.zeros((gl.num_element, gl.num_gp, num_strain_vctr_cmpnt, dim3 * num_sf))
    B2           = np.zeros((gl.num_element, gl.num_gp,                   dim,        num_sf))
    Bv           = np.zeros((gl.num_element, gl.num_gp, dim * num_sf                        ))
    
    for i, coord in enumerate(gl.normcoord_gp):
        xi  = coord[0]
        eta = coord[1]
        
        dndxi_dndeta[i][0, :] = dn1dxi(eta), dn2dxi(eta), dn3dxi(eta), dn4dxi(eta)
        dndxi_dndeta[i][1, :] = dn1deta(xi), dn2deta(xi), dn3deta(xi), dn4deta(xi)
    
    for i in range(gl.num_element):
        for j in range(gl.num_gp):
            J[i][j]    = np.dot(dndxi_dndeta[j], gl.element_coord[i])
            detJ[i][j] = np.linalg.det(J[i][j])
            
    for i in range(gl.num_element):
        for j in range(gl.num_gp):
            Jinv = np.linalg.inv(J[i][j])
            
            dndx_dndz[i][j] = np.dot(Jinv, dndxi_dndeta[j])
            dndx = dndx_dndz[i][j][0]
            dndz = dndx_dndz[i][j][1]
            
            B1[i][j][0][::3]  = dndx
            B1[i][j][2][2::3] = dndz
            B1[i][j][3][::3]  = dndz
            B1[i][j][3][2::3] = dndx
            
            B2[i][j][0] = dndx
            B2[i][j][1] = dndz
            
            Bv[i][j][::2]  = dndx
            Bv[i][j][1::2] = dndz
            
    return detJ, B1, B2, Bv

def mtrx_2dof(xi, eta):
    N1 = np.zeros((1, dim, num_sf))
    N1[::2]  = n1(xi, eta), n2(xi, eta), n3(xi, eta), n4(xi, eta)
    N1[1::2] = n1(xi, eta), n2(xi, eta), n3(xi, eta), n4(xi, eta)
    
    return N1

def mtrx_1dof(xi, eta):
    N2 = np.zeros((1, num_sf))
    N2[:] = n1(xi, eta), n2(xi, eta), n3(xi, eta), n4(xi, eta)
    
    return N2