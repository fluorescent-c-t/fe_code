#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import numpy as np
import datetime

from sub import global_variable as gl
from sub import update as updt

def construct():
    global name
    dt_now = datetime.datetime.now()
    name = dt_now.strftime('%Y%m%d%H%M%S')
    
    column_index_node_dof1 = '\t'.join([item for i in range(gl.num_node) for item in ['{}'.format(i+1)]])
    column_index_node_dof2 = '\t'.join([item for i in range(gl.num_node) for item in ['{}'.format(i+1), '{}'.format(i+1)]])
    column_index_node_dof4 = '\t'.join([item for i in range(gl.num_node) for item in ['{}'.format(i+1), '{}'.format(i+1), '{}'.format(i+1), '{}'.format(i+1)]])
    
    column_index_gp_dof1 = '\t'.join([item for i in range(gl.num_node) for j in range(gl.num_gp) for item in ['{}-G{}'.format(i+1, j+1)]])
    column_index_gp_dof2 = '\t'.join([item for i in range(gl.num_node) for j in range(gl.num_gp) for item in ['{}-G{}'.format(i+1, j+1), '{}-G{}'.format(i+1, j+1)]])
    column_index_gp_dof4 = '\t'.join([item for i in range(gl.num_node) for j in range(gl.num_gp) for item in ['{}-G{}'.format(i+1, j+1), '{}-G{}'.format(i+1, j+1), '{}-G{}'.format(i+1, j+1), '{}-G{}'.format(i+1, j+1)]])
    
    with open('./Output/1_DISPLACEMENT_NODES_%s.txt' %name, 'x') as f1:
        column_node_disp = '\t'.join(['dxx', 'dzz'] * gl.num_node)
        f1.write('Node No.:' + '\t' + column_index_node_dof2 + '\n')
        f1.write('time' + '\t' + column_node_disp + '\n')
        
    with open('./Output/2_STRAIN_NODES_%s.txt' %name, 'x') as f2:
        column_node_strain = '\t'.join(['epsxx', 'espyy', 'epszz', 'gammazx'] * gl.num_node)
        f2.write('Node No.:' + '\t' + column_index_node_dof4 + '\n')
        f2.write('time' + '\t' + column_node_strain + '\n')
    
    with open('./Output/3_STRESS_NODES_%s.txt' %name, 'x') as f3:
        column_node_stress = '\t'.join(['sigxx', 'sigyy', 'sigzz', 'tauzx'] * gl.num_node)
        f3.write('Node No.:' + '\t' + column_index_node_dof4 + '\n')
        f3.write('time' + '\t' + column_node_stress + '\n')
        
    with open('./Output/4_INVARIANTS_NODES_%s.txt' %name, 'x') as f4:
        column_node_inv = '\t'.join(['p', 'q', 'epsv', 'epsd'] * gl.num_node)
        f4.write('Node No.:' + '\t' + column_index_node_dof4 + '\n')
        f4.write('time' + '\t' + column_node_inv + '\n')
    
    with open('./Output/5_PWP_NODES_%s.txt' %name, 'x') as f5:
        columns_node_pwp = '\t'.join(['u'] * gl.num_node)
        f5.write('Node No.:' + '\t' + column_index_node_dof1 + '\n')
        f5.write('time' + '\t' + columns_node_pwp + '\n')
        
    with open('./Output/6_TEMPERATURE_NODES_%s.txt' %name, 'x') as f6:
        columns_node_temp = '\t'.join(['T'] * gl.num_node)
        f6.write('Node No.:' + '\t' + column_index_node_dof1 + '\n')
        f6.write('time' + '\t' + columns_node_temp + '\n')
    
    #with open('./Output/7_STRESS_GP_%s.txt' %name, 'x') as f7:
        #column_gp_stress = '\t'.join((['sigxx', 'sigyy', 'sigzz', 'tauzx'] * gl.num_gp) * gl.num_element)
        #f6.write('Node No.:' + '\t' + column_index_gp_dof4 + '\n')
        #f6.write('time' + '\t' + column_gp_stress + '\n')
    
    #with open('./Output/8_VOIDRATIO_GP_%s.txt' %name, 'x') as f8:
        #column_gp_stress = '\t'.join(['eG1','eG2','eG3','eG4'] * gl.num_element)
        #f7.write('Node No.:' + '\t' + column_index_gp_dof4 + '\n')
        #f7.write('time' + '\t' + column_gp_stress + '\n')
    
        
def append(time):
    stress_node, strain_node, stress_inv_node, strain_inv_node = updt.extrapolate_to_node()
    inv_node = np.hstack((stress_inv_node, strain_inv_node))
    
    with open('./Output/1_DISPLACEMENT_NODES_%s.txt' %name, 'a') as f1:
        f1.write('{}'.format(time) + '\t' + "\t".join(map(str, gl.d.flatten())) + '\n')
    
    with open('./Output/2_STRAIN_NODES_%s.txt' %name, 'a') as f2:
        f2.write('{}'.format(time) + '\t' + "\t".join(map(str, strain_node.flatten())) + '\n')
    
    with open('./Output/3_STRESS_NODES_%s.txt' %name, 'a') as f3:
        f3.write('{}'.format(time) + '\t' + "\t".join(map(str, stress_node.flatten())) + '\n')
    
    with open('./Output/4_INVARIANTS_NODES_%s.txt' %name, 'a') as f4:
        f4.write('{}'.format(time) + '\t' + "\t".join(map(str, inv_node.flatten())) + '\n')
        
    with open('./Output/5_PWP_NODES_%s.txt' %name, 'a') as f5:
        f5.write('{}'.format(time) + '\t' + "\t".join(map(str, gl.u.flatten()))  + '\n')
    
    with open('./Output/6_TEMPERATURE_NODES_%s.txt' %name, 'a') as f6:
        f6.write('{}'.format(time) + '\t' + "\t".join(map(str, gl.T.flatten()))  + '\n')
        
    #with open('./Output/7_STRESS_GP_%s.txt' %name, 'a') as f7:
        #f6.write('{}'.format(time) + '\t' + "\t".join(map(str, gl.stress_gp.flatten()))  + '\n')
        
    #with open('./Output/8_VOIDRATIO_GP_%s.txt' %name, 'a') as f8:
        #f7.write('{}'.format(time) + '\t' + "\t".join(map(str, gl.voidratio_gp.flatten()))  + '\n')
    
    