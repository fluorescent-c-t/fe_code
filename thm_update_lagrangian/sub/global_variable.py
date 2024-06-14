#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

max_iteration               = None  #Max. number of N.R. iteration: not used
rsd_tol                     = None  #Tolerance for N.R. interation: not used
theta                       = None  #Time-marching factor
yld_tol                     = None  #Tolerance for yileding 
init_E                      = None  #Kick-off parameters for Cam-clay elasticity
init_poi                    = None
act_hydrst_prsr             = None  #Activation flag for hydro static pressure
act_latentheat              = None  #Activation flag for latent heat: water to ice

sett_output                 = None  #Setting about outputting files, not used

g                           = None  #Gravitational accelaration
rhow_unfr                   = None  #Density of unfrozen water
rhow_fr                     = None  #Density of frozen water
Lw                          = None  #Thermal propeties of water
kTw_unfr                    = None  #unfr: unfrozen
kTw_fr                      = None  #fr  : frozen
cw_unfr                     = None
cw_fr                       = None
T_unfr                      = None
T_fr                        = None

rho_ghst                    = None  #Parameters for ghost element
E_ghst                      = None
poi_ghst                    = None
k_ghst                      = None
kT_ghst                     = None
Cv_ghst                     = None

dof_m                       = 2
dof_h                       = 1
dof_t                       = 1
dof_hm                      = dof_m + dof_h
dof_thm                     = dof_m + dof_h + dof_t

num_element                 = None  #Variables for F.E. algorithm
num_node                    = None
global_coord                = None
element_coord               = None 
conn                        = None
num_node_quad1              = 4
num_gp                      = 4
normcoord_gp                = [
    [- 0.57735, - 0.57735], 
    [  0.57735, - 0.57735], 
    [  0.57735,   0.57735], 
    [- 0.57735,   0.57735]
    ]
weight_gi                   = [1., 1., 1., 1.] #gi refers to gauss intergration
num_strn_cmp                = 4
num_strs_cmp                = 4
num_sf                      = 4
num_dfmgrd_cmp              = 5

dim                         = 2

ele_act_flag                = None  #Element activation flag
cmmn_matparam               = None  #Material properties of element
mech_modeltype              = None
mech_matparam               = None
mech_anchrval               = None
hydr_modeltype              = None
hydr_matparam               = None
thrm_modeltype              = None
thrm_matparam               = None

bc_timedep                  = None  #Boundary condition sets
bc_timeindep                = None

stress_gp                   = None  #State variables
strain_gp                   = None
stress_inv_gp               = None
strain_inv_gp               = None
voidratio_gp                = None
init_voidratio_gp           = None

d                           = None
u                           = None
T                           = None