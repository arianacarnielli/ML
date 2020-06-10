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

    def show(self, color = None):
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
        
        plt.imshow(new_im)
    
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

def patch_to_vect(patch):
    """
    """
    return patch.reshape(-1)

def vect_to_patch(vect):
    """
    """
    h = int(np.sqrt(vect.size//3))
    return vect.reshape(h, h, 3)

def learn_w(noisy_patch, atoms, alpha = 0.01):
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
    w = lasso.coef_
    
    new_patch = Y.copy()
    new_patch[np.logical_not(mask)] = lasso.predict(X[np.logical_not(mask), :])
    
    new_patch = vect_to_patch(new_patch)
    
    return w0, {k:w[i] for i, k in enumerate(list_keys)}, new_patch

if __name__=="__main__":
    plt.close("all")
    
    img0 = Image("Images_Test/Bliss_Windows_XP.png", hsv = False)
    plt.figure()
    img0.show()
    
    img = Image(img_data = img0.img, hsv = img0.hsv)
    
    #img.add_noise(0.5)
    #img.add_noise_rect(0.75, 125, 75, 30, 50)
    img.delete_rect(150, 75, 30, 50)
    plt.figure()
    img.show()
    
    noisy, atoms = img.get_noisy_and_atoms(14, 14)
    
    first_noisy = list(noisy.keys())[0]
    
    p0 = img0.get_patch(*first_noisy, 14)
    ip0 = Image(img_data = p0, hsv = img.hsv)
    plt.figure()
    ip0.show()
    
    p1 = noisy[first_noisy]
    ip1 = Image(img_data = p1, hsv = img.hsv)
    plt.figure()
    ip1.show()
    
    w0, w, p2 = learn_w(p1, atoms)
    ip2 = Image(img_data = p2, hsv = img.hsv)
    plt.figure()
    ip2.show()
    
    p1[:, :, :] = p2
    plt.figure()
    img.show()