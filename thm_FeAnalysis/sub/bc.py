#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import numpy as np
import sys
from sub import global_variable as gl

def timedep(timestep):
    t_crrnt = gl.bc_timedep[timestep][0]
    t_nxt   = gl.bc_timedep[timestep + 1][0]
    
    dt  = t_nxt - t_crrnt
    
    if dt == 0.:
        print("")
        print("")
        print("Notice : You have consecutive same time values")
        print("Notice : Go to ./5_BC_TimeDep.txt and correct them")
        print("Message: The program was terminated")
        sys.exit()
    
    df  = np.zeros((gl.dof_m * gl.num_node, 1))
    Q2  = np.zeros((gl.dof_h * gl.num_node, 1))
    dQ2 = np.zeros((gl.dof_h * gl.num_node, 1))
    Q3  = np.zeros((gl.dof_t * gl.num_node, 1))
    dQ3 = np.zeros((gl.dof_t * gl.num_node, 1))
    b1  = np.zeros((gl.dof_m * gl.num_node, 1))
    b2  = np.zeros((gl.dof_h * gl.num_node, 1))
    
    for bc_crrnt, bc_nxt in zip(gl.bc_timedep[timestep][1:], gl.bc_timedep[timestep + 1][1:]):
        #bc_set has the structure like: [1, Fxx, 0.5] in the order of node no., type of b.c. and it value
        i           = bc_crrnt[0] - 1 #same as bc_nxt[0] - 1, node number
        type_bc     = bc_crrnt[1]     #same as bc_nxt[1]
        type_bc_nxt = bc_nxt[1]  
        val_crrnt   = bc_crrnt[2]
        val_nxt     = bc_nxt[2]
        
        if i+1  > gl.num_node:
            print('')
            print('Notice: the specified node no. in File: 5_BC_TimeDep.txt, exceeds the max. no of node')
            print("Notice: The program was terminated.")
            sys.exit()
            
        if type_bc_nxt == type_bc:
            if type_bc in {'Fxx', 'Fzz'}:
                df[2 * i + (type_bc == 'Fzz')] += val_nxt - val_crrnt
            
            elif type_bc in {'b1'}:
                for i in range(gl.num_element):
                    rho_sat = gl.cmmn_matparam[i][2]
                    
                
            
            elif type_bc in {'Q2'}:
                Q2[i]  = val_crrnt
                dQ2[i] = val_nxt - val_crrnt
                
            elif type_bc in {'Q3'}:
                Q3[i]  = val_crrnt
                dQ3[i] = val_nxt - val_crrnt
            
            #elif ActDeact_selfweight_porefluid == 1:
            
            elif type_bc in {'T'}:
                gl.T[i] = val_crrnt
            
        else:
            pass
    
    return dt, df, Q2, dQ2, Q3, dQ3, b1, b2

def timedep_assign(timestep, THM_mtrx, THM_vctr):
    for bc_crrnt, bc_nxt in zip(gl.bc_timedep[timestep][1:], gl.bc_timedep[timestep + 1][1:]):
        #bc_set has the structure like: [1, Fxx, 0.5] in the order of node no., type of b.c. and it value
        i           = bc_crrnt[0] - 1 #same as bc_nxt[0] - 1, node number
        type_bc     = bc_crrnt[1] 
        type_bc_nxt = bc_nxt[1] #same as bc_nxt[1]
        val_crrnt   = bc_crrnt[2]
        val_nxt     = bc_nxt[2]
        
        if type_bc_nxt == type_bc:
            if type_bc in {'dxx', 'dzz'}:
                THM_vctr -= (val_nxt - val_crrnt) * np.reshape(THM_mtrx[:, gl.dof_m * i + (type_bc == 'dzz')], (gl.dof_thm * gl.num_node, 1))
                THM_mtrx[:, gl.dof_m * i + (type_bc == 'dzz')] = 0.
                THM_mtrx[gl.dof_m * i + (type_bc == 'dzz'), :] = 0.
                THM_mtrx[gl.dof_m * i + (type_bc == 'dzz')][gl.dof_m * i + (type_bc == 'dzz')] = 1.
                THM_vctr[gl.dof_m * i + (type_bc == 'dzz')] = (val_nxt - val_crrnt)
            
        else:
            pass
        
def timeindep_assign(THM_mtrx, THM_vctr):
    for bc_sets in gl.bc_timeindep:
        i = bc_sets[0] - 1
        if i+1  > gl.num_node:
            print('')
            print('Notice: the specified node no. in File: 6_BC_TimeInDep.txt, exceeds the max. no of node')
            print("Notice: The program was terminated.")
            sys.exit()
            
        for bc_set in bc_sets[1:]:
            type_bc = bc_set[0]
            val_bc  = bc_set[1]
            
            if type_bc in {'ddxx', 'ddzz'}:
                THM_vctr -= val_bc * np.reshape(THM_mtrx[:, gl.dof_m * i + (type_bc == 'ddzz')], (gl.dof_thm * gl.num_node, 1))
                THM_mtrx[:, gl.dof_m * i + (type_bc == 'ddzz')] = 0.
                THM_mtrx[gl.dof_m * i + (type_bc == 'ddzz'), :] = 0.
                THM_mtrx[gl.dof_m * i + (type_bc == 'ddzz')][gl.dof_m * i + (type_bc == 'ddzz')] = 1.
                THM_vctr[gl.dof_m * i + (type_bc == 'ddzz')] = val_bc
                
            elif type_bc in {'du'}:
                THM_vctr -= val_bc * np.reshape(THM_mtrx[:, gl.dof_m * gl.num_node + gl.dof_h * i], (gl.dof_thm * gl.num_node, 1))
                THM_mtrx[:, gl.dof_m * gl.num_node + gl.dof_h * i] = 0.
                THM_mtrx[gl.dof_m * gl.num_node + gl.dof_h * i, :] = 0.
                THM_mtrx[gl.dof_m * gl.num_node + gl.dof_h * i][gl.dof_m * gl.num_node + gl.dof_h * i] = 1.
                THM_vctr[gl.dof_m * gl.num_node + gl.dof_h * i] = val_bc
            
            elif type_bc in {'dT'}:
                THM_vctr -= val_bc * np.reshape(THM_mtrx[:, gl.dof_hm * gl.num_node + gl.dof_t * i], (gl.dof_thm * gl.num_node, 1))
                THM_mtrx[:, gl.dof_hm * gl.num_node + gl.dof_t * i] = 0.
                THM_mtrx[gl.dof_hm * gl.num_node + gl.dof_t * i, :] = 0.
                THM_mtrx[gl.dof_hm * gl.num_node + gl.dof_t * i][gl.dof_hm * gl.num_node + gl.dof_t * i] = 1.
                THM_vctr[gl.dof_hm * gl.num_node + gl.dof_t * i] = val_bc
                
            else:
                print('')
                print("Notice: Specify the proper type of time-indep. B.C.: either ddxx, ddzz, du or dT.")
                print("Notice: The program was terminated.")
                sys.exit()