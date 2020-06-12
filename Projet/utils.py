# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:42:43 2020

@author: arian
"""

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

from sklearn.linear_model import Lasso

class Image:
    """
    """
    
    def __init__(self, filename = None, img_data = None, hsv = True):
        """
        """
        self.hsv = hsv
        if img_data is not None:
            self.img = img_data.copy()
        else:
            self._from_file(filename)
        
    def _from_file(self, filename):
        """
        """
        self.img = plt.imread(filename)
        
        # Si l'image a été lue comme des entiers, on la convertit en floats
        # dans [0, 1]
        if np.issubdtype(self.img.dtype, np.integer):
            info = np.iinfo(self.img.dtype)
            self.img = self.img/info.max
        
        # Conversion en hsv si nécessaire
        if self.hsv:
            self.img = clr.rgb_to_hsv(self.img)
            
        # On la convertit à des entrées dans [-1, 1]
        self.img = 2*self.img - 1

    def show(self, color = None, ax = None):
        """
        color : tuple avec 3 entrées dans [0, 1] représentant la couleur
        utilisée, dans le mode hsv si self.hsv est True, en mode rgb sinon.
        
        """
        new_im = self.img.copy()
        
        # Conversion à des entrées dans [0, 1]
        new_im = (new_im + 1)/2
        
        if color is None:
            color = (0, 0, 0)
        
        for i in range(3):
            canal = new_im[:, :, i]
            canal[canal < -30] = color[i]
        
        if self.hsv:
            new_im = clr.hsv_to_rgb(new_im)
        
        if ax is None:
            plt.imshow(new_im)
        else:
            ax.imshow(new_im)
    
    def get_patch(self, i, j, h):
        """
        (i, j) : coordonnées du point au centre
        """
        imin = i - (h//2)
        jmin = j - (h//2)
        return self.img[imin:imin+h, jmin:jmin+h, :]
               
    def add_noise(self, p):
        """
        """
        noise = np.random.random(self.img.shape[:2]) <= p
        for i in range(self.img.shape[2]):
            self.img[:, :, i] = np.where(noise, -100, self.img[:, :, i])
        
        
    def add_noise_rect(self, p, i, j, height, width):
        """
        (i, j) : coordonnées du point au centre
        """
        i0 = i - (height//2)
        j0 = j - (width//2)
        noise = np.random.random((height, width)) <= p
        for k in range(self.img.shape[2]):
            self.img[i0:i0+height, j0:j0+width, k] = \
                np.where(noise, -100, self.img[i0:i0+height, j0:j0+width, k])
        
    
    def delete_rect(self, i, j, height, width):
        """
        (i, j) : coordonnées du point au centre
        """
        i0 = i - (height//2)
        j0 = j - (width//2)
        self.img[i0:(i0+height), j0:(j0+width), :] = -100
        
    def get_noisy_and_atoms(self, h, step):
        """
        """
        noisy = {}
        atoms = {}
        for i in range(h//2, self.img.shape[0] - (h//2), step):
            for j in range(h//2, self.img.shape[1] - (h//2), step):
                p = self.get_patch(i, j, h)
                if np.all(p > -100):
                    atoms[(i, j)] = p
                else:
                    noisy[(i, j)] = p
        return noisy, atoms
    
    def get_boundary_patches(self, h):
        boundary = {}
        
        noisy = set(zip(*np.where(self.img[:, :, 0] == -100)))
        for i, j in noisy:
            if (i-1 >= 0 and (i-1, j) not in noisy) or \
               (j+1 < self.img.shape[1] and (i, j+1) not in noisy) or \
               (i+1 < self.img.shape[0] and (i+1, j) not in noisy) or \
               (j-1 >= 0 and (i, j-1) not in noisy):
                boundary[(i, j)] = self.get_patch(i, j, h)
        return boundary

def patch_to_vect(patch):
    """
    """
    return patch.reshape(-1)

def vect_to_patch(vect):
    """
    """
    h = int(np.sqrt(vect.size//3))
    return vect.reshape(h, h, 3)

def learn_w(noisy_patch, atoms, alpha = 0.01, predict = True):
    """
    """
    Y = patch_to_vect(noisy_patch)
    mask = Y > -100
    list_keys = list(atoms.keys())
    X = np.empty((Y.size, len(list_keys)))
    for i, k in enumerate(list_keys):
        X[:, i] = patch_to_vect(atoms[k])
        
    lasso = Lasso(alpha = alpha, fit_intercept = True, max_iter = 5000)
    lasso.fit(X[mask, :], Y[mask])
    
    w0 = lasso.intercept_
    w = {k:lasso.coef_[i] for i, k in enumerate(list_keys)}
    
    if not predict:
        return w0, w
    
    new_patch = Y.copy()
    new_patch[np.logical_not(mask)] = lasso.predict(X[np.logical_not(mask), :])
    
    new_patch = vect_to_patch(new_patch)
    new_patch[new_patch >= 1] = 1
    new_patch[new_patch <= -1] = -1
    
    return w0, w, new_patch

def fill_image(img, h, step, score, alpha = 0.01, perc_best = 0.1,\
               verbose = False):
    """
    """
    boundary = img.get_boundary_patches(h)
    list_pos = []
    
    while len(boundary) > 0:
        if verbose:
            print(len(boundary), end=' ')
        
        # Calcul d'un patch assez proche du meilleur
        scores = {k: score(p) for k, p in boundary.items()}
        max_score = max(scores.values())
        tol_score = max_score * (1 - perc_best)
        l = [k for k, s in scores.items() if s >= tol_score]
        best_pos = np.random.choice(len(l))
        best_pos = l[best_pos]
        best_patch = boundary[best_pos]
        
        list_pos.append(best_pos)
        
        _, atoms = img.get_noisy_and_atoms(h, step)
        
        _, _, new_p = learn_w(best_patch, atoms, alpha = alpha)
        best_patch[:, :, :] = new_p
        
        boundary = img.get_boundary_patches(h)
    return list_pos

def simple_score(patch):
    """
    """
    return np.sum(patch!=-100)

def std_score(patch, alpha = 0.5):
    filled = np.sum(patch!=-100)/patch.size
    std = 0
    index = patch[:, :, 0] != -100
    for i in range(patch.shape[2]):
        std += patch[:, :, i][index].std()
    std /= patch.shape[2]
    
    return filled + alpha * std

if __name__=="__main__":
    plt.close("all")
    
    img0 = Image("Images_Test/Bliss_Windows_XP.png", hsv = False)
    plt.figure()
    img0.show()
    
    img = Image(img_data = img0.img, hsv = img0.hsv)
    
    #img.add_noise(0.5)
    #img.add_noise_rect(0.75, 125, 75, 30, 50)
    img.delete_rect(125, 75, 30, 50)
    plt.figure()
    img.show()
    
    order = fill_image(img, 9, 9, std_score, alpha = 0.001, perc_best = 0, verbose = True)
    
    plt.figure()
    img.show()