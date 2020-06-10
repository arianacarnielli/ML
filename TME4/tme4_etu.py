# -*- coding: utf-8 -*-
import numpy as np

class Egg(object):
    def __init__(self, classifier, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.classifiers = dict()
        for i in range(self.nb_classes):
            for j in range(i + 1, self.nb_classes):
                self.classifiers[(i, j)] = classifier(**kwargs)
                
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
                score[:, i] += x[:, 0]
                score[:, j] += np.logical_not(x)[:, 0]
        return np.array([self.classes[k] for k in np.argmax(score, 1)])   
    
    def accuracy(self, datax, datay):
        datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
        y_pred = self.predict(datax).reshape(-1, 1)
        return (y_pred == datay).mean()
    
    
class Ova(object):
    def __init__(self, classifier, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.classifiers = np.empty(self.nb_classes, dtype = object)
        for i in range(self.nb_classes):
            self.classifiers[i] = classifier(**kwargs)
                
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
            score[:, i] = x[:, 0]
        return np.array([self.classes[k] for k in np.argmin(score, 1)])   
    
    def accuracy(self, datax, datay):
        datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
        y_pred = self.predict(datax).reshape(-1, 1)
        return (y_pred == datay).mean()