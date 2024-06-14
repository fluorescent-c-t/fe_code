#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import os
dirname = os.path.dirname(__file__)
os.chdir(dirname)

import sys
import numpy as np
import importlib
import time

start = time.time()

from sub import readfile
from sub import global_variable as gl
from sub import shapefunction as sf
from sub import cnstttv_m as cm
from sub import cnstttv_h as ch
from sub import cnstttv_t as ct
from sub import bc
from sub import solver as slv
from sub import update as updt
from sub import writefile

importlib.reload(readfile)
importlib.reload(gl)
importlib.reload(sf)
importlib.reload(cm)
importlib.reload(ch)
importlib.reload(bc)
importlib.reload(slv)
importlib.reload(updt)
importlib.reload(writefile)

print('')
print('Message: This code is written based on Python 3.9')
print('Message: This code employs the explicit solver w/o convergence iterations')
print('')
print('')

readfile.setting()
readfile.constant()
readfile.node()
readfile.element()
readfile.bc()
readfile.state_variable()

writefile.construct()

t = 0.
num_timestep = len(gl.bc_timedep) - 1
for timestep in range(num_timestep):
    sys.stdout.write('\033[2K\033[GMessage: Current timestep is {} out of {}  '.format(timestep, num_timestep - 1))
    sys.stdout.flush()
    
    #Read geometry
    detJ, B1, B2, Bv, Bf, Bvm = sf.mtrcs_deriv_gp()
    
    #Intialize
    dt, df, Q2, dQ2, Q3, dQ3, b1, b2 = bc.timedep(timestep)#Read b.c.: contrcution of vectors
    
    THM_mtrx = np.zeros((gl.dof_thm * gl.num_node, gl.dof_thm * gl.num_node))#Construct THM matrix and vector
    THM_vctr = np.zeros((gl.dof_thm * gl.num_node, 1))
    
    K1  = np.zeros((gl.dof_m * gl.num_node, gl.dof_m * gl.num_node))#Construct matrices
    rx  = np.zeros((gl.dof_m * gl.num_node, 1))
    HM  = np.zeros((gl.dof_m * gl.num_node, gl.dof_h * gl.num_node))
    K2  = np.zeros((gl.dof_h * gl.num_node, gl.dof_h * gl.num_node))
    K3  = np.zeros((gl.dof_t * gl.num_node, gl.dof_t * gl.num_node))
    c   = np.zeros((gl.dof_t * gl.num_node, gl.dof_t * gl.num_node))
    
    De_gp   = np.zeros((gl.num_element, gl.num_gp, gl.num_strn_cmp, gl.num_strn_cmp))
    D_gp    = np.zeros((gl.num_element, gl.num_gp, gl.num_strn_cmp, gl.num_strn_cmp))
    dεvp_gp = np.zeros((gl.num_element, gl.num_gp, gl.num_strn_cmp,               1))
    
    σA_gp   = np.zeros((gl.num_element, gl.num_gp,   gl.num_strn_cmp,   gl.num_strn_cmp))
    σB_gp   = np.zeros((gl.num_element, gl.num_gp, gl.num_dfmgrd_cmp, gl.num_dfmgrd_cmp))
    σC_gp   = np.zeros((gl.num_element, gl.num_gp,   gl.num_strn_cmp,                 1))
    
    #Create the matrices and vectors according to the current soil states
    T_gp = ct.intrpl_gp(gl.T)
    for i in range(gl.num_element): 
        K1e, rx_e = cm.matrcs_ele(detJ, B1, Bvm, Bf, i, T_gp, dt, De_gp, D_gp, dεvp_gp, σA_gp, σB_gp, σC_gp)#Form element matrices
        K2e, HMe  = ch.matrcs_ele(detJ, B2, Bv, i, T_gp)
        K3e, ce   = ct.matrcs_ele(detJ, B2, i, T_gp)
        for j in range(gl.num_node_quad1):
            for k in range(gl.num_node_quad1):#Globalize them
                K1[gl.dof_m * (gl.conn[i][j] - 1)    ][gl.dof_m * (gl.conn[i][k] - 1)    ] += K1e[(gl.dof_m + 1) * j    ][(gl.dof_m + 1) * k    ]
                K1[gl.dof_m * (gl.conn[i][j] - 1)    ][gl.dof_m * (gl.conn[i][k] - 1) + 1] += K1e[(gl.dof_m + 1) * j    ][(gl.dof_m + 1) * k + 2]
                K1[gl.dof_m * (gl.conn[i][j] - 1) + 1][gl.dof_m * (gl.conn[i][k] - 1)    ] += K1e[(gl.dof_m + 1) * j + 2][(gl.dof_m + 1) * k    ]
                K1[gl.dof_m * (gl.conn[i][j] - 1) + 1][gl.dof_m * (gl.conn[i][k] - 1) + 1] += K1e[(gl.dof_m + 1) * j + 2][(gl.dof_m + 1) * k + 2]
                K2[gl.dof_h * (gl.conn[i][j] - 1)    ][gl.dof_h * (gl.conn[i][k] - 1)    ] += K2e[gl.dof_h       * j    ][gl.dof_h       * k    ]
                HM[gl.dof_m * (gl.conn[i][j] - 1)    ][gl.dof_h * (gl.conn[i][k] - 1)    ] += HMe[gl.dof_m       * j    ][gl.dof_h       * k    ]
                HM[gl.dof_m * (gl.conn[i][j] - 1) + 1][gl.dof_h * (gl.conn[i][k] - 1)    ] += HMe[gl.dof_m       * j + 1][gl.dof_h       * k    ]
                K3[gl.dof_t * (gl.conn[i][j] - 1)    ][gl.dof_t * (gl.conn[i][k] - 1)    ] += K3e[gl.dof_t       * j    ][gl.dof_t       * k    ]
                c [gl.dof_t * (gl.conn[i][j] - 1)    ][gl.dof_t * (gl.conn[i][k] - 1)    ] +=  ce[gl.dof_t       * j    ][gl.dof_t       * k    ]
            
            rx[gl.dof_m * (gl.conn[i][j] - 1)    ] += rx_e[(gl.dof_m + 1) * j    ]
            rx[gl.dof_m * (gl.conn[i][j] - 1) + 1] += rx_e[(gl.dof_m + 1) * j + 2]
    
    THM_mtrx[:gl.dof_m * gl.num_node                       , :gl.dof_m * gl.num_node                       ] = K1
    THM_mtrx[:gl.dof_m * gl.num_node                       , gl.dof_m * gl.num_node:gl.dof_hm * gl.num_node] = HM
    THM_mtrx[gl.dof_m * gl.num_node:gl.dof_hm * gl.num_node, :gl.dof_m * gl.num_node                       ] = HM.T
    THM_mtrx[gl.dof_m * gl.num_node:gl.dof_hm * gl.num_node, gl.dof_m * gl.num_node:gl.dof_hm * gl.num_node] =   - gl.theta * dt * K2
    THM_mtrx[gl.dof_hm * gl.num_node:                      , gl.dof_hm * gl.num_node:                      ] = c + gl.theta * dt * K3

    THM_vctr[:gl.dof_m * gl.num_node                       ] =   df + rx
    THM_vctr[gl.dof_m * gl.num_node:gl.dof_hm * gl.num_node] =   dt * (K2 @ gl.u - gl.theta * dQ2 + Q2 + b2)
    THM_vctr[gl.dof_hm * gl.num_node:                      ] = - dt * (K3 @ gl.T - gl.theta * dQ3 - Q3     )
    
    #Solve
    bc.timedep_assign(timestep, THM_mtrx, THM_vctr)
    bc.timeindep_assign(THM_mtrx, THM_vctr)
    solution = slv.explicit_sparse(THM_mtrx, THM_vctr)
    
    #Check loading state
    dd         = solution[:gl.dof_m * gl.num_node]
    dε_gp_trl  = updt.strain_increment_gp(dd, B1)
    unldng_flg = cm.judge_loading(dε_gp_trl, De_gp, D_gp)
    
    if unldng_flg > 0:
        K1 = cm.reconstruct(B1, detJ, D_gp)
        THM_mtrx[:gl.dof_m * gl.num_node, :gl.dof_m * gl.num_node] = K1
        solution = slv.explicit_sparse(THM_mtrx, THM_vctr)
        dd       = solution[:gl.dof_m * gl.num_node]
        dε_gp    = updt.strain_increment_gp(dd, B1)
        
    else:
        dε_gp = dε_gp_trl
        
    #Update
    updt.variables_gp(De_gp, D_gp, dε_gp, dεvp_gp, dd, Bf, B1, σA_gp)
    updt.coordinate(solution[:gl.dof_m * gl.num_node])#Update-lagrangian method
    
    t += dt
    gl.d += solution[:gl.dof_m * gl.num_node]
    gl.u += solution[gl.dof_m * gl.num_node:gl.dof_hm * gl.num_node]
    gl.T += solution[gl.dof_hm * gl.num_node:]
    if timestep % 1 == 0:
        writefile.append(t)

end = time.time()
print("")
print("")
print("")
print("Message: Total D.O.F.  :", gl.dof_thm * gl.num_node)
print("Message: Process time  :", round((end - start) / 60., 2), "[min]")
