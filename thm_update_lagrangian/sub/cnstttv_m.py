# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import numpy as np
import pandas as pd
from sub import global_variable as gl
from sub import shapefunction as sf

def save_as_excel(matrix_data, filename):
    df = pd.DataFrame(matrix_data)
    excel_path = "./%s.xlsx" % filename
    df.to_excel(excel_path)

def deriv_pq(i, j):
    sigxx, sigyy, sigzz, tauzx = gl.stress_gp[i][j].T[0]
    dpdsig       = np.array([
                             [1./3.],
                             [1./3.],
                             [1./3.],
                                [0.]
                            ])
    dqdsig_tms_q = np.array([
                             [        sigxx - 0.5 * sigyy - 0.5 * sigzz],
                             [- 0.5 * sigxx +       sigyy - 0.5 * sigzz],
                             [- 0.5 * sigxx - 0.5 * sigyy +       sigzz],
                                                            [3. * tauzx]
                            ])#dqdsig_tms_q: dqdσ * q
    return dpdsig, dqdsig_tms_q
    
def stffnss(i, E, poi, p, kpp, pN):
    if gl.mech_modeltype[i] == 0:
        K  =  E / 3. / (1. - 2. * poi)
        
    elif gl.mech_modeltype[i] in {1, 2, 3, 4}:
        if p <= pN:
            E   = gl.init_E
            poi = gl.init_poi
            K   = E / 3. / (1. - 2. * poi)
        else:
            K = p / kpp
            
    G = 3. * (1. - 2. * poi) * K / 2. / (1. + poi)
    return K, G

def matrcs_ele(detJ, B1, Bvm, Bf, i, T_gp, dt, De_gp, D_gp, dεvp_gp, σA_gp, σB_gp, σC_gp):
    def plstc_cmpl_mcc(i, j, De_gp, lmd, kpp, M, p, dpdsig, dqdsig_tms_q, pN, N):
        if gl.mech_modeltype[i] == 1:#needs yield judgement
            if p <= pN:
                cm = 0.
            else:
                eta   = q / p
                piso  = p * ((M ** 2. + eta ** 2.) / M ** 2.)
                εv    = gl.strain_inv_gp[i][j][0]
                
                ρ  = (N + lmd * np.log(piso / pN)) - εv
                
                if ρ <= gl.yld_tol:
                    cf       = (lmd - kpp) * (M ** 2. - eta ** 2.) / p / (M ** 2. + eta ** 2.)
                    dfdsig   = cf * (dpdsig + 2. / p / (M ** 2. - eta ** 2.) * dqdsig_tms_q)
                    a        = gl.mech_matparam[i][8]
                    dfdh     = - 1.
                    dfdε     = np.array([1., 1., 1., 0.])
                    dfdρdρdh = - a * ρ
                    nmr      = (dfdsig.T @ De_gp[i][j] @ dfdsig).item()
                    dnm      = - (dfdh + dfdρdρdh) * (dfdε @ dfdsig).item() + (dfdsig.T @ De_gp[i][j] @ dfdsig).item()
                    
                    cm       =  nmr / dnm
                else:
                    cm = 0.
        else:
            cm = 0.

        return cm
    
    def isotach_state(lmd, kpp, alph, beta, cT, p, T, Tref, dpdsig, dqdsig_tms_q, dt, pN, N):
        dεvp  = np.zeros((gl.num_strn_cmp, 1))
        if gl.mech_modeltype[i] in {2, 3, 4}:#calculate isotach state
            if p <= pN:
                pass
            else:
                eta  = q / p
                piso = p * ((M ** 2. + eta ** 2.) / M ** 2.)
                εv   = gl.strain_inv_gp[i][j][0]
                
                if gl.mech_modeltype[i] == 2:#volumetric hardening
                    dεvpdt = dεvpdt_ref * np.exp((N + lmd * np.log(piso / pN) - εv) / alph)
                    
                else:
                    if gl.mech_modeltype[i] == 4:
                        alph += lmd * cT * (T - Tref)
                    
                    dεvpdt = dεvpdt_ref * np.exp((N + lmd * np.log(piso / pN) + beta * (T - Tref) - εv) / alph)
                
                dεvp   = dt * dεvpdt * ((M ** 2. - eta ** 2.) * dpdsig + 2. / p * dqdsig_tms_q) / M ** 2.
                
        return dεvp
    
    E          = gl.mech_matparam[i][0]
    poi        = gl.mech_matparam[i][1]
    lmd        = gl.mech_matparam[i][2] / 2.303
    kpp        = gl.mech_matparam[i][3] / 2.303
    M          = gl.mech_matparam[i][4]
    alph       = gl.mech_matparam[i][5] / 2.303
    beta       = gl.mech_matparam[i][6]
    cT         = gl.mech_matparam[i][7]
    
    pN         = gl.mech_anchrval[i][0]
    N          = gl.mech_anchrval[i][1]
    dεvpdt_ref = gl.mech_anchrval[i][2]
    Tref       = gl.mech_anchrval[i][3]
    
    K1e  = np.zeros(((gl.dof_m + 1) * gl.num_node_quad1, (gl.dof_m + 1) * gl.num_node_quad1))
    rx_e = np.zeros(((gl.dof_m + 1) * gl.num_node_quad1, 1))
    
    for j in range(gl.num_gp):
        T     = T_gp[i][j]
        p, q  = gl.stress_inv_gp[i][j][0], gl.stress_inv_gp[i][j][1]

        K, G = stffnss(i, E, poi, p, kpp, pN)#Stiffness
        De_gp[i][j][0][0]   = De_gp[i][j][1][1]    = De_gp[i][j][2][2]  = K + 4. / 3. * G
        De_gp[i][j][0][1:3] = De_gp[i][j][1][:3:2] = De_gp[i][j][2][:2] = K - 2. / 3. * G
        De_gp[i][j][3][3]   = G
        
        dpdsig, dqdsig_tms_q = deriv_pq(i, j)
        
        cm         = plstc_cmpl_mcc(i, j, De_gp, lmd, kpp, M, p, dpdsig, dqdsig_tms_q, pN, N)
        D_gp[i][j] = (1. - cm) * De_gp[i][j]
        
        #geometric nonlinearity
        u = pwp_gp(i, j)
        append_strs_tnsr(i, j, σA_gp, σB_gp, σC_gp, u)
        
        K1e       += gl.weight_gi[j] * B1[i][j].T @  D_gp[i][j]   @  B1[i][j] * detJ[i][j]
        K1e       -= gl.weight_gi[j] * B1[i][j].T @ σA_gp[i][j]   @  B1[i][j] * detJ[i][j]
        K1e       += gl.weight_gi[j] * Bf[i][j].T @ σB_gp[i][j]   @  Bf[i][j] * detJ[i][j]
        K1e       += gl.weight_gi[j] * B1[i][j].T @ σC_gp[i][j]   @ Bvm[i][j] * detJ[i][j]
        K1e       -= gl.weight_gi[j] *          u *    Bf[i][j].T @  Bf[i][j] * detJ[i][j]
        
        dεvp_gp[i][j] = isotach_state(lmd, kpp, alph, beta, cT, p, T, Tref, dpdsig, dqdsig_tms_q, dt, pN, N)#Stress ralaxation term
        rx_e         += - gl.weight_gi[j] * B1[i][j].T @ De_gp[i][j] @ dεvp_gp[i][j] * detJ[i][j]
        
    return K1e, rx_e

def pwp_gp(i, j):
    ue   = np.zeros((gl.dof_h * gl.num_node_quad1, 1))#pore water pressure, element
    for j in range(gl.num_node_quad1):
        ue[gl.dof_h * j] = gl.u[gl.dof_h * (gl.conn[i][j] - 1)]
    N2 = sf.mtrx_1dof(gl.normcoord_gp[j][0], gl.normcoord_gp[j][1])

    return (N2 @ ue)[0][0]

def append_strs_tnsr(i, j, σA_gp, σB_gp, σC_gp, u):
    sigxx, sigyy, sigzz, tauzx = gl.stress_gp[i][j].T[0]
    
    σA_gp[i][j][0][0], σA_gp[i][j][1][1], σA_gp[i][j][2][2]                       = 2.0 *  sigxx, 2.0 * sigyy, 2.0 * sigzz
    σA_gp[i][j][3][3]                                                             = 0.5 * (sigxx + sigzz)
    σA_gp[i][j][0][3] = σA_gp[i][j][2][3] = σA_gp[i][j][3][0] = σA_gp[i][j][3][1] = tauzx
    
    σB_gp[i][j][0][0], σB_gp[i][j][1][1], σB_gp[i][j][2][2]                       = sigxx, sigyy, sigzz
    σB_gp[i][j][3][4]                                                             = sigxx
    σB_gp[i][j][4][3]                                                             = sigzz
    σB_gp[i][j][0][3] = σB_gp[i][j][2][4] = σB_gp[i][j][3][2] = σB_gp[i][j][4][0] = tauzx
    
    σC_gp[i][j][0][0], σC_gp[i][j][1][0], σC_gp[i][j][2][0] = sigxx + u, sigyy + u, sigzz + u
    σC_gp[i][j][3][0]                                       = tauzx

def judge_loading(dε_gp, De_gp, D_gp):
    def loading_criteria(unldng_flg):
        dpdsig, dqdsig_tms_q = deriv_pq(i, j)
        cf       = (lmd - kpp) * (M ** 2. - eta ** 2.) / p / (M ** 2. + eta ** 2.)
        dfdsig   =  cf * (dpdsig + 2. / p / (M ** 2. - eta ** 2.) * dqdsig_tms_q)
        
        if dfdsig.T @ De_gp[i][j] @ dε_gp[i][j] < 0:
            unldng_flg += 1
            D_gp[i][j] = De_gp[i][j]
        else:
            pass
    
    unldng_flg = 0#Loading: +0, unloading: +1
    for i in range(gl.num_element):
        if gl.mech_modeltype[i] == 1:#MCC
            lmd        = gl.mech_matparam[i][2] / 2.303
            kpp        = gl.mech_matparam[i][3] / 2.303
            M          = gl.mech_matparam[i][4]
            
            pN         = gl.mech_anchrval[i][0]
            N          = gl.mech_anchrval[i][1]
            
            for j in range(gl.num_gp):
                p, q  = gl.stress_inv_gp[i][j][0], gl.stress_inv_gp[i][j][1]
                if p <= pN:
                    pass
                else:
                    eta   = q / p
                    piso  = p * ((M ** 2. + eta ** 2.) / M ** 2.)
                    
                    εv = gl.strain_inv_gp[i][j][0]
                    
                    ρ  = εv - (N + lmd * np.log(piso / pN))
                    
                    if ρ <= gl.yld_tol:#If yielding, then needs the loading judgement
                        loading_criteria(unldng_flg)
                        
                    else:
                        pass
                    
    return unldng_flg

def reconstruct(B1, Bf, Bvm, detJ, D_gp, σA_gp, σB_gp, σC_gp):
    K1  = np.zeros((gl.dof_m * gl.num_node, gl.dof_m * gl.num_node))
    for i in range(gl.num_element):
        K1e   = np.zeros(((gl.dof_m + 1) * gl.num_node_quad1, (gl.dof_m + 1) * gl.num_node_quad1))
        for j in range(gl.num_gp):
            u = pwp_gp(i, j)
            
            K1e       += gl.weight_gi[j] * B1[i][j].T @  D_gp[i][j]   @  B1[i][j] * detJ[i][j]
            K1e       -= gl.weight_gi[j] * B1[i][j].T @ σA_gp[i][j]   @  B1[i][j] * detJ[i][j]
            K1e       += gl.weight_gi[j] * Bf[i][j].T @ σB_gp[i][j]   @  Bf[i][j] * detJ[i][j]
            K1e       += gl.weight_gi[j] * B1[i][j].T @ σC_gp[i][j]   @ Bvm[i][j] * detJ[i][j]
            K1e       -= gl.weight_gi[j] *          u *    Bf[i][j].T @  Bf[i][j] * detJ[i][j]
        
        for j in range(gl.num_node_quad1):
            for k in range(gl.num_node_quad1):
                K1[gl.dof_m * (gl.conn[i][j] - 1)    ][gl.dof_m * (gl.conn[i][k] - 1)    ] += K1e[(gl.dof_m + 1) * j    ][(gl.dof_m + 1) * k    ]
                K1[gl.dof_m * (gl.conn[i][j] - 1)    ][gl.dof_m * (gl.conn[i][k] - 1) + 1] += K1e[(gl.dof_m + 1) * j    ][(gl.dof_m + 1) * k + 2]
                K1[gl.dof_m * (gl.conn[i][j] - 1) + 1][gl.dof_m * (gl.conn[i][k] - 1)    ] += K1e[(gl.dof_m + 1) * j + 2][(gl.dof_m + 1) * k    ]
                K1[gl.dof_m * (gl.conn[i][j] - 1) + 1][gl.dof_m * (gl.conn[i][k] - 1) + 1] += K1e[(gl.dof_m + 1) * j + 2][(gl.dof_m + 1) * k + 2]
    return K1
    