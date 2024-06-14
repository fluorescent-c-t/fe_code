#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:54:42 2024

@author: kochitaishi
"""

import os
import sys
import numpy as np
import scipy as sp

from sub import global_variable as gl

def explicit_normal(THM_mtrx, THM_vctr):
    solution = np.linalg.solve(THM_mtrx, THM_vctr)
    return solution

def explicit_sparse(THM_mtrx, THM_vctr):
    THM_mtrx = sp.sparse.csr_matrix(THM_mtrx)
    solution = sp.sparse.linalg.spsolve(THM_mtrx, THM_vctr)
    return solution.reshape((gl.dof_thm * gl.num_node, 1))