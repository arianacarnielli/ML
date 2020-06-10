from arftools import gen_arti, plot_data, plot_frontiere, make_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
    w = w.reshape(-1, 1)    
    return ((datax.dot(w) - datay)**2).mean()

def mse_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return (datax.T.dot(datax.dot(w) - datay)) * (2 / datay.size)
    
def hinge(datax, datay, w):
    """ retourn la moyenne de l'erreur hinge """
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return np.maximum(0, -datay * datax.dot(w)).mean()

def hinge_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur hinge """
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return (datax.T.dot(((datay * datax.dot(w)) <= 0) * -datay)) / datay.size

def tikhonov(datax, datay, w, alpha=1, lamb=1):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return np.maximum(0, alpha - datay * datax.dot(w)).mean() + lamb * (w**2).sum()

def tikhonov_g(datax, datay, w, alpha=1, lamb=1):
    datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)    
    w = w.reshape(-1, 1)
    return (datax.T.dot(((datay * datax.dot(w)) <= alpha) * -datay)) / datay.size + 2 * lamb * w
    
    
class Lineaire(object):
    def __init__(self,loss=hinge, loss_g=hinge_g, max_iter=1000, eps=0.01, hist=False, bias=True):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter, eps
        self.loss, self.loss_g = loss, loss_g
        self.hist = hist
        self.bias = bias

    def fit(self, data_x, data_y, testx=None, testy=None, batch=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = data_y.reshape(-1,1)
        N = len(datay)
        datax = data_x.reshape(N,-1)
        if self.bias:
            datax = np.hstack((datax, np.ones((N, 1))))
        D = datax.shape[1]
        self.w = np.random.random((D,1))
        if self.hist:
            self.w_hist = np.empty((self.max_iter, D))
        if testx is not None:
            erreurs = np.empty((self.max_iter, 2))

        for i in range(self.max_iter):             
            if batch is not None:
                list_batch = np.random.choice(N, batch, False)
                grad = self.loss_g(datax[list_batch, :], datay[list_batch, :], self.w)
            else:
                grad = self.loss_g(datax, datay, self.w)
            self.w = self.w - self.eps * grad     
            if self.hist:
                self.w_hist[i, :] = self.w[:, 0]
            if testx is not None:
                erreurs[i, 0] = self.score(data_x, data_y)
                erreurs[i, 1] = self.score(testx, testy)      
        if testx is not None:
            return erreurs
        
    def predict(self, datax):
        if len(datax.shape) == 1:
            datax = datax.reshape(1,-1)
        if self.bias:
            datax = np.hstack((datax, np.ones((datax.shape[0], 1))))
        return datax.dot(self.w)

    def score(self, datax, datay):
        if self.bias:
            datax = np.hstack((datax, np.ones((datax.shape[0], 1))))
        return self.loss(datax, datay, self.w)
    
    def accuracy(self, datax, datay):
        datax, datay = datax.reshape(len(datay), -1), datay.reshape(-1, 1)  
        return ((self.predict(datax) * datay) >= 0).mean()

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()

def plot_error(datax, datay, f, step=10):
    grid, x1list, x2list = make_grid(xmin=-4, xmax=4, ymin=-4, ymax=4, step=step)
    plt.contourf(x1list, x2list, np.array([f(datax, datay, w) for w in grid]).reshape(x1list.shape), 25)
    plt.colorbar()
    plt.show()
    
def projection_polynomiale(datax): 
    res = np.ones((datax.shape[0], 6))
    res[:, 0:2] = datax
    res[:, 3] = datax[:, 0]**2
    res[:, 4] = datax[:, 0] * datax[:, 1]
    res[:, 5] = datax[:, 1]**2
    return res

def projection_gaussienne(datax, x1min, x1max, x2min, x2max, n1, n2):
    x1 = np.linspace(x1min, x1max, n1)
    x2 = np.linspace(x2min, x2max, n2)
    sigma1 = x1[1] - x1[0]
    sigma2 = x2[1] - x2[0]    
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid_x1 = grid_x1.reshape(-1)
    grid_x2 = grid_x2.reshape(-1)
    res = np.empty((datax.shape[0], grid_x1.size))

    for d in range(res.shape[1]):
       res[:, d] = np.exp(- (datax[:,0] - grid_x1[d])**2/ (2 * sigma1**2)) * np.exp(- (datax[:,1] - grid_x2[d])**2/ (2 * sigma2**2)) 
    return res
    

if __name__=="__main__":
    plt.close("all")
    
    """ Tracer des isocourbes de l'erreur """
    plt.ion()
    
    trainx, trainy = gen_arti(nbex = 1000, data_type = 1, epsilon = 1)
    testx, testy = gen_arti(nbex =1000, data_type = 1, epsilon = 1)
    
    #Pour tester si le biais fonctionne
    #testx += 1
    #trainx += 1
    
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.05, hist=True)
    perceptron.fit(trainx,trainy)
    
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    color = np.empty((perceptron.w_hist.shape[0], 3))
    color[:, 0] = 1
    color[:, 1] = np.linspace(0, 1, color.shape[0])
    color[:, 2] = np.linspace(0, 1, color.shape[0])
    plt.scatter(perceptron.w_hist[:, 0], perceptron.w_hist[:, 1], c = color)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx, perceptron.predict, 200)
    plot_data(trainx,trainy)

 