#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""
import numpy as np
import sys
from sub import global_variable as gl
from sub import update as updt

def read_file(file_path, skip_empty_lines=True):
        with open(file_path, 'r') as file:
            data = file.read().splitlines()
            if skip_empty_lines:
                data = [line for line in data if line]
            return [list(line.split()) for line in data]

def setting():
     setting = read_file('./1_Setting.txt') #相対パスはmain.pyがあるディレクトリを参照する
     
     gl.rsd_tol            = float(setting[0][0].replace(',', ''))
     gl.max_iteration      = int(  setting[1][0].replace(',', ''))
     gl.theta              = float(setting[2][0].replace(',', ''))
     gl.yld_tol            = float(setting[3][0].replace(',', ''))
     gl.init_E             = float(setting[4][0].replace(',', ''))
     gl.init_poi           = float(setting[5][0].replace(',', ''))
     gl.act_hydrst_prsr    = int(  setting[6][0].replace(',', ''))
     gl.act_latentheat     = int(  setting[7][0].replace(',', ''))
     gl.pN                 = float(setting[8][0].replace(',', ''))

     gl.sett_output        = setting[20:40]

def constant():
     constant = read_file('./2_Constant.txt')

     gl.g          = float(constant[0][0].replace(',', ''))
     gl.rhow_unfr  = float(constant[1][0].replace(',', ''))
     gl.rhow_fr    = float(constant[2][0].replace(',', ''))
     gl.Lw         = float(constant[3][0].replace(',', ''))
     gl.kTw_unfr   = float(constant[4][0].replace(',', ''))
     gl.kTw_fr     = float(constant[5][0].replace(',', ''))
     gl.cw_unfr    = float(constant[6][0].replace(',', ''))
     gl.cw_fr      = float(constant[7][0].replace(',', ''))
     gl.T_unfr     = float(constant[8][0].replace(',', ''))
     gl.T_fr       = float(constant[9][0].replace(',', ''))

     gl.rho_ghst   = float(constant[20][0].replace(',', ''))
     gl.E_ghst     = float(constant[21][0].replace(',', ''))
     gl.poi_ghst   = float(constant[22][0].replace(',', ''))
     gl.k_ghst     = float(constant[23][0].replace(',', ''))
     gl.kT_ghst    = float(constant[24][0].replace(',', ''))
     gl.Cv_ghst    = float(constant[25][0].replace(',', ''))
     
def node():
    node               = read_file('./3_Nodes.txt')

    gl.num_node        = len(node)
    gl.global_coord    = [[float(coord[1]), float(coord[2])] for coord in node]
    
    if gl.num_node == len(list(map(list, set(map(tuple, gl.global_coord))))):
        pass
    else:
        print("")
        print("Notice : You have overlappeing nodes")
        print("Notice : Go to ./3_Nodes.txt and correct them")
        print("Message: The program was terminated")
        sys.exit()

def element():
    element                = read_file('./4_Elements.txt')

    gl.num_element         = len(element)
    gl.conn                = [[int(  item[1].replace(',', '')),
                               int(  item[2].replace(',', '')),
                               int(  item[3].replace(',', '')),
                               int(  item[4].replace(',', ''))] for item in element]
    gl.ele_act_flag        = [ int(  item[5].replace(',', ''))  for item in element]
    gl.cmmn_matparam       = [[float(item[i].replace(',', ''))  for i in range(6, 16)] for item in element]
    gl.mech_modeltype      = [ int(  item[16].replace(',', '')) for item in element]
    gl.mech_matparam       = [[float(item[i].replace(',', ''))  for i in range(17, 57)] for item in element]
    gl.mech_anchrval       = [[float(item[i].replace(',', ''))  for i in range(57, 67)] for item in element]
    gl.hydr_modeltype      = [ int(  item[67].replace(',', '')) for item in element]
    gl.hydr_matparam       = [[float(item[i].replace(',', ''))  for i in range(68, 88)] for item in element]
    gl.thrm_modeltype      = [ int(  item[88].replace(',', '')) for item in element]
    gl.thrm_matparam       = [[float(item[i].replace(',', ''))  for i in range(89, 99)] for item in element]
    
    if np.amax(gl.conn) > gl.num_node:
        print("Notice :  You have numbers that exceed the max. node No.")
        print("Notice :  Go to 4_Elements.txt and correct the error")
        print("Message:  The program was terminated")
        sys.exit()
    else:
        pass

def bc():
    gl.bc_timedep = read_file('./5_BC_TimeDep.txt')#本行ではテキストデータをそのまま格納し、以下で変数の型指定
    for line in gl.bc_timedep:
        line[0]     = float(line[0])
        num_bc      = int((len(line) - 1) / 3)
        line[1:]    = [[int(line[1 + 3 * j].replace(',', '')), str(line[1 + 3 * j + 1]), float(line[1 + 3 * j + 2].replace(',', ''))] for j in range(num_bc)]
    
    gl.bc_timeindep = read_file('./6_BC_TimeInDep.txt')#本行ではテキストデータをそのまま格納し、以下で変数の型指定
    for line in gl.bc_timeindep:
        line[0]     = int(line[0])
        num_bc      = int((len(line) - 1) / 2)
        line[1:]    = [[str(line[1 + 2 * j]), float(line[1 + 2 * j + 1].replace(',', ''))] for j in range(num_bc)]

def state_variable():
    initial_stress_gp    = read_file('./7_InitialStressGauss.txt')
    initial_temperature  = read_file('./8_InitialTempNode.txt')
    initial_voidratio_gp = read_file('./9_InitialVoidRatioGauss.txt')

    mmax         = len(initial_stress_gp[0])
    interval     = int((len(initial_stress_gp[0]) - 1) / gl.num_gp)

    gl.stress_gp            = np.array([[(np.array(list(map(float, line[i:i+4])))).reshape(-1, 1) for i in range(1, mmax, interval)] for line in initial_stress_gp])
    gl.strain_gp            = np.zeros((gl.num_element, gl.num_gp, gl.num_strn_cmp, 1))
    gl.strain_inv_gp        = np.array([[updt.strain_invariant(strain_vctr) for strain_vctr in line] for line in gl.strain_gp])
    gl.stress_inv_gp        = np.array([[updt.stress_invariant(stress_vctr) for stress_vctr in line] for line in gl.stress_gp])
    gl.voidratio_gp         = np.array([[float(line[i + 1]) for i in range(gl.num_gp)] for line in initial_voidratio_gp])
    gl.init_voidratio_gp    = np.array([[float(line[i + 1]) for i in range(gl.num_gp)] for line in initial_voidratio_gp])
    
    gl.d                    = np.zeros((gl.dof_m * gl.num_node, 1)) 
    gl.u                    = np.zeros((gl.dof_h * gl.num_node, 1))
    gl.T                    = np.array([[float(line[1])] for line in initial_temperature])
