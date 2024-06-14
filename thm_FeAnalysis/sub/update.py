#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import numpy as np

from sub import global_variable as gl

def stress_invariant(stress_vctr):
    sigxx, sigyy, sigzz, tauzx = stress_vctr.T[0]

    term_normal = (sigxx - sigyy) ** 2. + (sigyy - sigzz) ** 2. + (sigzz - sigxx) ** 2.
    term_shear  = 6. * tauzx ** 2.

    p = (sigxx + sigyy + sigzz) / 3.
    q = np.sqrt(term_normal + term_shear) / 1.41421356
    
    return [p, q]

def strain_invariant(strain_vctr):
    epsxx, epsyy, epszz, gammazx = strain_vctr.T[0]

    term_normal = epsxx ** 2. + epszz ** 2. + (epszz - epsxx) ** 2. 
    term_shear  = 1.5 * gammazx ** 2.

    epsv = epsxx + epsyy + epszz
    epsd = (1.414213 / 3.0) * np.sqrt(term_normal + term_shear)

    return [epsv, epsd]

def strain_increment_gp(dd, B1):
    dε_gp = np.zeros((gl.num_element, gl.num_gp, gl.num_strn_cmp, 1))
    for i in range(gl.num_element):
        dde = np.zeros(((gl.dof_m + 1) * gl.num_node_quad1, 1))
        for j in range(gl.num_node_quad1):
            dde[(gl.dof_m + 1) * j    ] = dd[gl.dof_m * (gl.conn[i][j] - 1)    ]
            dde[(gl.dof_m + 1) * j + 2] = dd[gl.dof_m * (gl.conn[i][j] - 1) + 1]
        
        for j in range(gl.num_gp):
            dε_gp[i][j] = - B1[i][j] @ dde
        
    return dε_gp

def variables_gp(De_gp, D_gp, dε_gp, dεvp):
    for i in range(gl.num_element):
        for j in range(gl.num_gp):
            dsig = D_gp[i][j] @ (dε_gp[i][j] - dεvp[i][j])
            
            gl.strain_gp[i][j]     += dε_gp[i][j]
            gl.stress_gp[i][j]     += dsig
            gl.strain_inv_gp[i][j]  = strain_invariant(gl.strain_gp[i][j])
            gl.stress_inv_gp[i][j]  = stress_invariant(gl.stress_gp[i][j])
            
            e0 = gl.init_voidratio_gp[i][j]
            εv = gl.strain_inv_gp[i][j][0]
            de = εv * (1. + e0)
            
            gl.voidratio_gp[i][j] = e0 - de

def extrapolate_to_node():
    num_inv = 2
    stress_node = np.zeros((gl.num_node, gl.num_strs_cmp, 1))
    strain_node = np.zeros((gl.num_node, gl.num_strn_cmp, 1))
    stress_inv_node = np.zeros((gl.num_node, num_inv))
    strain_inv_node = np.zeros((gl.num_node, num_inv))
    
    for i in range(gl.num_element):
        fg1_strs, fg2_strs, fg3_strs, fg4_strs = gl.stress_gp[i]
        fn1_strs =   1.866 * fg1_strs - 0.500 * fg2_strs - 0.500 * fg3_strs + 0.134 * fg4_strs
        fn2_strs = - 0.500 * fg1_strs + 1.866 * fg2_strs + 0.134 * fg3_strs - 0.500 * fg4_strs
        fn3_strs = - 0.500 * fg1_strs + 0.134 * fg2_strs + 1.866 * fg3_strs - 0.500 * fg4_strs
        fn4_strs =   0.134 * fg1_strs - 0.500 * fg2_strs - 0.500 * fg3_strs + 1.866 * fg4_strs

        extrpltd_strs = [fn1_strs, fn2_strs, fn3_strs, fn4_strs]
        fg1_strn, fg2_strn, fg3_strn, fg4_strn = gl.strain_gp[i]
        fn1_strn =   1.866 * fg1_strn - 0.500 * fg2_strn - 0.500 * fg3_strn + 0.134 * fg4_strn
        fn2_strn = - 0.500 * fg1_strn + 1.866 * fg2_strn + 0.134 * fg3_strn - 0.500 * fg4_strn
        fn3_strn = - 0.500 * fg1_strn + 0.134 * fg2_strn + 1.866 * fg3_strn - 0.500 * fg4_strn
        fn4_strn =   0.134 * fg1_strn - 0.500 * fg2_strn - 0.500 * fg3_strn + 1.866 * fg4_strn
        
        extrpltd_strn = [fn1_strn, fn2_strn, fn3_strn, fn4_strn]
        
        fg1_strs_inv, fg2_strs_inv, fg3_strs_inv, fg4_strs_inv = gl.stress_inv_gp[i]
        fn1_strs_inv =   1.866 * fg1_strs_inv - 0.500 * fg2_strs_inv - 0.500 * fg3_strs_inv + 0.134 * fg4_strs_inv
        fn2_strs_inv = - 0.500 * fg1_strs_inv + 1.866 * fg2_strs_inv + 0.134 * fg3_strs_inv - 0.500 * fg4_strs_inv
        fn3_strs_inv = - 0.500 * fg1_strs_inv + 0.134 * fg2_strs_inv + 1.866 * fg3_strs_inv - 0.500 * fg4_strs_inv
        fn4_strs_inv =   0.134 * fg1_strs_inv - 0.500 * fg2_strs_inv - 0.500 * fg3_strs_inv + 1.866 * fg4_strs_inv
        
        extrpltd_strs_inv = [fn1_strs_inv, fn2_strs_inv, fn3_strs_inv, fn4_strs_inv]
        
        fg1_strn_inv, fg2_strn_inv, fg3_strn_inv, fg4_strn_inv = gl.strain_inv_gp[i]
        fn1_strn_inv =   1.866 * fg1_strn_inv - 0.500 * fg2_strn_inv - 0.500 * fg3_strn_inv + 0.134 * fg4_strn_inv
        fn2_strn_inv = - 0.500 * fg1_strn_inv + 1.866 * fg2_strn_inv + 0.134 * fg3_strn_inv - 0.500 * fg4_strn_inv
        fn3_strn_inv = - 0.500 * fg1_strn_inv + 0.134 * fg2_strn_inv + 1.866 * fg3_strn_inv - 0.500 * fg4_strn_inv
        fn4_strn_inv =   0.134 * fg1_strn_inv - 0.500 * fg2_strn_inv - 0.500 * fg3_strn_inv + 1.866 * fg4_strn_inv
        
        extrpltd_strn_inv = [fn1_strn_inv, fn2_strn_inv, fn3_strn_inv, fn4_strn_inv]
        
        for j in range(gl.num_node_quad1):
            stress_node[gl.conn[i][j] - 1] += extrpltd_strs[j]
            strain_node[gl.conn[i][j] - 1] += extrpltd_strn[j]
            stress_inv_node[gl.conn[i][j] - 1] += extrpltd_strs_inv[j]
            strain_inv_node[gl.conn[i][j] - 1] += extrpltd_strn_inv[j]
        
    for i in range(gl.num_node):
        I = i + 1
        
        overlapdegree = np.count_nonzero(np.array(gl.conn) < I + 1) - np.count_nonzero(np.array(gl.conn) < I)
        stress_node[i] = stress_node[i] / overlapdegree
        strain_node[i] = strain_node[i] / overlapdegree
        stress_inv_node[i] = stress_inv_node[i] / overlapdegree
        strain_inv_node[i] = strain_inv_node[i] / overlapdegree
    return stress_node, strain_node, stress_inv_node, strain_inv_node