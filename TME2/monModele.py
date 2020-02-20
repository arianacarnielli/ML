# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:47:26 2020

@author: arian
"""
import numpy as np

def histogramme (geo_mat, xmin, xmax, ymin, ymax, steps):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(ymin, ymax, steps)
    case_x = 0
    case_y = 0
    hist = np.zeros((steps, steps))
    for yi, xi in geo_mat:
        for i in range(1, x.shape[0]):
            if x[i - 1] <= xi <= x[i]: 
                case_x = i
        for j in range(1, y.shape[0]):
            if y[j - 1] <= yi <= y[j]:
                case_y = j
        hist[case_y][case_x] += 1
    return hist