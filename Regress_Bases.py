#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nossaiba
"""

import numpy as np
import matplotlib.pyplot as plt

## datasets 
m = 200
x = np.linspace(0, 10, m).reshape((m, 1))
y = x +np.random.randn(m, 1)
#Affichage
plt.scatter(x, y,label='Data',c='g') 


## Variables 
X = np.hstack((x, np.ones(x.shape)))
theta = np.random.randn(2, 1)

#le modèle
def modele(X, theta):
    return X.dot(theta)

#l'erreur
def err_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((modele(X, theta) - y)**2)

#gradient
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(modele(X, theta) - y)

#desente du gradient 
def gradient_descent(X, y, theta, alpha, n_iterations):
    err_history = np.zeros(n_iterations)     
    for i in range(0, n_iterations):
        err_history[i] = err_function(X, y, theta)         
        theta = theta - alpha * grad(X, y, theta) 
    return theta, err_history

n_iterations = 300
alpha = 0.001
 
theta_optimal, err_history = gradient_descent(X, y, theta, alpha, n_iterations)

# La prédiction
predictions = modele(X, theta_optimal).reshape(m,1)

# Affiche des résultats 
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.show()



#Etude de performance
moyen=y.mean()
d1 = np.sum((predictions - y)**2)
d2 = np.sum((y - moyen)**2)
r2=1-d1/d2
print("the r-squared is: ",r2)


#afficher l'histrique de l'erreur
plt.plot(range(n_iterations), err_history)
plt.show()