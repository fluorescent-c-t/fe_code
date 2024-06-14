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

def mtrcs_deriv_gp():
    dndxi_dndeta = np.zeros((gl.num_gp, gl.dim, gl.num_sf))
    
    J            = np.zeros((gl.num_element, gl.num_gp,            gl.dof_m,                   gl.dof_m))
    detJ         = np.zeros((gl.num_element, gl.num_gp))
    dndx_dndz    = np.zeros((gl.num_element, gl.num_gp,            gl.dof_m,                  gl.num_sf))
    B1           = np.zeros((gl.num_element, gl.num_gp,     gl.num_strn_cmp, (gl.dof_m + 1) * gl.num_sf))
    B2           = np.zeros((gl.num_element, gl.num_gp,            gl.dof_m,                  gl.num_sf))
    Bv           = np.zeros((gl.num_element, gl.num_gp, gl.dof_m * gl.num_sf))
    
    Bf           = np.zeros((gl.num_element, gl.num_gp, gl.num_dfmgrd_cmp, (gl.dof_m + 1) * gl.num_sf))
    Bvm          = np.zeros((gl.num_element, gl.num_gp,                 1, (gl.dof_m + 1) * gl.num_sf))
    
    for i, coord in enumerate(gl.normcoord_gp):
        xi  = coord[0]
        eta = coord[1]
        
        dndxi_dndeta[i][0, :] = dn1dxi(eta), dn2dxi(eta), dn3dxi(eta), dn4dxi(eta)
        dndxi_dndeta[i][1, :] = dn1deta(xi), dn2deta(xi), dn3deta(xi), dn4deta(xi)
    
    gl.element_coord = np.array([[gl.global_coord[gl.conn[i][j] - 1] for j in range(gl.num_node_quad1)] for i in range(gl.num_element)])
    
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
            
            Bf[i][j][0][::3]  = dndx
            Bf[i][j][2][2::3] = dndz
            Bf[i][j][3][::3]  = dndz
            Bf[i][j][4][2::3] = dndx
            
            Bvm[i][j][0][::3]  = dndx
            Bvm[i][j][0][2::3] = dndz
            
    return detJ, B1, B2, Bv, Bf, Bvm

def mtrx_2dof(xi, eta):
    N1 = np.zeros((1, gl.dim, gl.num_sf))
    N1[::2]  = n1(xi, eta), n2(xi, eta), n3(xi, eta), n4(xi, eta)
    N1[1::2] = n1(xi, eta), n2(xi, eta), n3(xi, eta), n4(xi, eta)
    
    return N1

def mtrx_1dof(xi, eta):
    N2 = np.zeros((1, gl.num_sf))
    N2[:] = n1(xi, eta), n2(xi, eta), n3(xi, eta), n4(xi, eta)
    
    return N2