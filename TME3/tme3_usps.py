from arftools import gen_arti, plot_data, plot_frontiere, make_grid
from tme3_etu import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

    

if __name__=="__main__":
    plt.close("all")                

#premier point
# =============================================================================
#     train_datax, train_datay = load_usps("USPS_train.txt")    
#     test_datax, test_datay = load_usps("USPS_test.txt")
# 
#     valplus = 9
#     valminus = 6
# 
#     train_x = train_datax[np.logical_or(train_datay == valplus, train_datay == valminus), :]    
#     train_y = train_datay[np.logical_or(train_datay == valplus, train_datay == valminus)]
#     train_y = np.where(train_y == valplus, 1, -1)
#     
#     test_x = test_datax[np.logical_or(test_datay == valplus, test_datay == valminus), :]    
#     test_y = test_datay[np.logical_or(test_datay == valplus, test_datay == valminus)]
#     test_y = np.where(test_y == valplus, 1, -1)
#     
#     perceptron = Lineaire(hinge, hinge_g, max_iter=5000, eps=0.05)
#     perceptron.fit(train_x, train_y)
#     print("Erreur : train %f, test %f"% (perceptron.score(train_x,train_y),perceptron.score(test_x,test_y)))
#     show_usps(perceptron.w[:-1])
# =============================================================================
    
#2eme point
# =============================================================================
#     train_datax, train_datay = load_usps("USPS_train.txt")    
#     test_datax, test_datay = load_usps("USPS_test.txt")
# 
#     valplus = 6
# 
#     train_x = train_datax 
#     train_y = np.where(train_datay == valplus, -1, 1)
#     
#     test_x = test_datax
#     test_y = np.where(test_datay == valplus, -1, 1)
#     
#     perceptron = Lineaire(hinge, hinge_g, max_iter=5000, eps=0.05)
#     perceptron.fit(train_x, train_y)
#     print("Erreur : train %f, test %f"% (perceptron.score(train_x,train_y),perceptron.score(test_x,test_y)))
#     show_usps(perceptron.w[:-1])
# =============================================================================

#3eme point
# =============================================================================
#     train_datax, train_datay = load_usps("USPS_train.txt")    
#     test_datax, test_datay = load_usps("USPS_test.txt")
# 
#     valplus = 9
#     valminus = 6
# 
#     train_x = train_datax[np.logical_or(train_datay == valplus, train_datay == valminus), :]    
#     train_y = train_datay[np.logical_or(train_datay == valplus, train_datay == valminus)]
#     train_y = np.where(train_y == valplus, 1, -1)
#     
#     test_x = test_datax[np.logical_or(test_datay == valplus, test_datay == valminus), :]    
#     test_y = test_datay[np.logical_or(test_datay == valplus, test_datay == valminus)]
#     test_y = np.where(test_y == valplus, 1, -1)
#     
#     perceptron = Lineaire(hinge, hinge_g, max_iter=3000, eps=0.05)
#     erreurs = perceptron.fit(train_x, train_y, test_x, test_y)
#     print("Erreur : train %f, test %f"% (perceptron.score(train_x,train_y),perceptron.score(test_x,test_y)))
#     show_usps(perceptron.w[:-1])
#     
#     fig, ax = plt.subplots()
#     ax.plot(erreurs[:, 0], label = "Apprentissage")
#     ax.plot(erreurs[:, 1], label = "test")
#     ax.set_yscale("log")
#     ax.legend()
#     ax.set_xlabel("nb d'itérations")
#     ax.set_ylabel("erreur")
#     ax.grid(True)
#     ax.set_axisbelow(True)
#     fig.show()
# =============================================================================
    
# =============================================================================
#     train_datax, train_datay = load_usps("USPS_train.txt")    
#     test_datax, test_datay = load_usps("USPS_test.txt")
#     
#     valplus = 6
#     
#     train_x = train_datax 
#     train_y = np.where(train_datay == valplus, -1, 1)
#     
#     test_x = test_datax
#     test_y = np.where(test_datay == valplus, -1, 1)
#         
#     perceptron = Lineaire(hinge, hinge_g, max_iter=5000, eps=0.1)
#     erreurs = perceptron.fit(train_x, train_y, test_x, test_y)
#     print("Erreur : train %f, test %f"% (perceptron.score(train_x,train_y),perceptron.score(test_x,test_y)))
#     show_usps(perceptron.w[:-1])
#     
#     fig, ax = plt.subplots()
#     ax.plot(erreurs[:, 0], label = "Apprentissage")
#     ax.plot(erreurs[:, 1], label = "test")
#     ax.set_yscale("log")
#     ax.legend()
#     ax.set_xlabel("nb d'itérations")
#     ax.set_ylabel("erreur")
#     ax.grid(True)
#     ax.set_axisbelow(True)
#     fig.show()
# =============================================================================
    
#projection polynomiale

# =============================================================================
#     plt.close("all")
#     
#     trainx, trainy = gen_arti(nbex = 1000, data_type = 1, epsilon = 0.5)
#     testx, testy = gen_arti(nbex = 1000, data_type = 1, epsilon = 0.5)
#     
#     ntrainx = projection_polynomiale(trainx)
#     ntestx = projection_polynomiale(testx)
#     
#     perceptron = Lineaire(hinge, hinge_g, max_iter = 1000, eps = 0.05, bias = False)
#     perceptron.fit(ntrainx,trainy)
#    
#     print("Erreur : train %f, test %f"% (perceptron.score(ntrainx,trainy),perceptron.score(ntestx,testy)))
#     plt.figure()
#     plot_frontiere(trainx, lambda x: perceptron.predict(projection_polynomiale(x)), 200)
#     plot_data(trainx,trainy)
# =============================================================================

#projection gaussienne

    plt.close("all")
    
    trainx, trainy = gen_arti(nbex = 4000, data_type = 2)
    testx, testy = gen_arti(nbex = 4000, data_type = 2)
    
    x1min = trainx[:, 0].min()
    x2min = trainx[:, 1].min()
    x1max = trainx[:, 0].max()
    x2max = trainx[:, 1].max()
    n1 = 40
    n2 = 40
    ntrainx = projection_gaussienne(trainx, x1min, x1max, x2min, x2max, n1, n2)
    ntestx = projection_gaussienne(testx, x1min, x1max, x2min, x2max, n1, n2)
    
    perceptron = Lineaire(hinge, hinge_g, max_iter = 1000, eps = 0.05)
    perceptron.fit(ntrainx,trainy)
   
    print("Erreur : train %f, test %f"% (perceptron.score(ntrainx,trainy),perceptron.score(ntestx,testy)))
    plt.figure()
    plot_frontiere(trainx, lambda x: perceptron.predict(projection_gaussienne(x, x1min, x1max, x2min, x2max, n1, n2)), 200)
    plot_data(trainx,trainy)

