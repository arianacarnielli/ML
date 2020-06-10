# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:34:29 2020

@author: arian
"""

import numpy as np

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)
    
class OvO(object):
    def __init__(self, classifier, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.classifiers = dict()
        self.params = kwargs
        for i in range(self.nb_classes):
            for j in range(i + 1, self.nb_classes):
                self.classifiers[(i, j)] = classifier(**kwargs)
        self.params['classifier'] = classifier
        self.params['nb_classes'] = nb_classes
                
    def get_params(self, deep = False):
        return self.params
    
    def set_params(self, **params):
        for key, val in params.items():
            self.params[key] = val
        for cl in self.classifiers.values():
            cl.set_params(**params)
        return self
                
    def fit(self, datax, datay):
        self.classes = np.unique(datay)
        assert self.classes.size == self.nb_classes
        for i in range(self.nb_classes):
            for j in range(i + 1, self.nb_classes):
                valplus = i
                valminus = j
                train_x = datax[np.logical_or(datay == valplus, datay == valminus), :]    
                train_y = datay[np.logical_or(datay == valplus, datay == valminus)]
                train_y = np.where(train_y == valplus, 1, -1)
                self.classifiers[(i, j)].fit(train_x, train_y)
                
    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1,-1)
        score = np.zeros((datax.shape[0], self.nb_classes), dtype = int)
        for i in range(self.nb_classes):
            for j in range(i + 1, self.nb_classes):
                x = (self.classifiers[(i, j)].predict(datax) >= 0)
                score[:, i] += x
                score[:, j] += np.logical_not(x)
        return np.array([self.classes[k] for k in np.argmax(score, 1)])   
    
    def score(self, datax, datay):
        datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
        y_pred = self.predict(datax).reshape(-1, 1)
        return (y_pred == datay).mean()
    
    
class OvA(object):
    def __init__(self, classifier, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.classifiers = np.empty(self.nb_classes, dtype = object)
        self.params = kwargs
        for i in range(self.nb_classes):
            self.classifiers[i] = classifier(**kwargs)
        self.params['classifier'] = classifier
        self.params['nb_classes'] = nb_classes
            
    def get_params(self, deep = False):
        return self.params
    
    def set_params(self, **params):
        for key, val in params.items():
            self.params[key] = val
        for cl in self.classifiers:
            cl.set_params(**params)
        return self
                
    def fit(self, datax, datay):
        self.classes = np.unique(datay)
        assert self.classes.size == self.nb_classes
        for i in range(self.nb_classes):
            valplus = i
            train_x = datax 
            train_y = np.where(datay == valplus, -1, 1)
            self.classifiers[i].fit(train_x, train_y)
                
    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1,-1)
        score = np.zeros((datax.shape[0], self.nb_classes))
        for i in range(self.nb_classes):
            x = self.classifiers[i].predict(datax)
            score[:, i] = x
        return np.array([self.classes[k] for k in np.argmin(score, 1)])   
    
    def score(self, datax, datay):
        datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
        y_pred = self.predict(datax).reshape(-1, 1)
        return (y_pred == datay).mean()