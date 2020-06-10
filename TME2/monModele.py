# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:47:26 2020

@author: arian
"""
import numpy as np
import scipy.stats as st

def histogramme_simple (geo_mat, xmin, xmax, ymin, ymax, steps):
    x = np.linspace(xmin, xmax, steps + 1)
    y = np.linspace(ymin, ymax, steps + 1)
    case_x = 0
    case_y = 0
    hist = np.zeros((steps, steps))
    for yi, xi in geo_mat:
        for i in range(1, x.shape[0]):
            if x[i - 1] <= xi < x[i]: 
                case_x = i - 1
        for j in range(1, y.shape[0]):
            if y[j - 1] <= yi < y[j]:
                case_y = j - 1
        hist[case_y][case_x] += 1
    return hist

def histogramme_opt (geo_mat, xmin, xmax, ymin, ymax, steps):
    dx = (xmax - xmin) / steps
    dy = (ymax - ymin) / steps
    hist = np.zeros((steps, steps))
    for yi, xi in geo_mat:    
        i = min(int((xi - xmin) / dx), steps - 1)
        j = min(int((yi - ymin) / dy), steps - 1)
        hist[j, i] += 1
    return hist

def histogramme (geo_mat, xmin, xmax, ymin, ymax, steps):
    hist = histogramme_opt(geo_mat, xmin, xmax, ymin, ymax, steps)
    dx = (xmax - xmin) / steps
    dy = (ymax - ymin) / steps
    N = geo_mat.shape[0]
    return hist/(N * dx * dy)


def noyau (geo_mat, xmin, xmax, ymin, ymax, steps, hx, hy, phi):
    hist = np.zeros((steps, steps))
    dx = (xmax - xmin) / steps
    dy = (ymax - ymin) / steps
    N = geo_mat.shape[0]
    for i in range(steps):
        for j in range(steps):
            x = xmin + i * dx + dx / 2
            y = ymin + j * dy + dy / 2
            hist[j, i] = (1 / N) * (1 / (hx * hy)) * phi((np.array([[y, x]]) - geo_mat) / np.array([[hy, hx]])).sum()
    return hist

def uniform (tab):
    return np.prod(abs(tab) <= 1 / 2, 1)

def normal (tab):
    rv = st.multivariate_normal(mean = np.zeros(tab.shape[1]))
    return rv.pdf(tab)
    
    
    
    








    