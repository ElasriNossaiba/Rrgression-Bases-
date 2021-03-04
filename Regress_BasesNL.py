#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nossaiba
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0) 

## datasets 
m = 100 
x = np.linspace(0, 10, m).reshape((m, 1))
y = x +np.random.randn(m, 1)
y=y**2
#Affichage
plt.scatter(x, y,label='Data',c='g') 
## Variables 
x2=x**2
X = np.hstack((x2, x, np.ones(x.shape)))
theta = np.random.randn(3, 1)

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
        theta = theta - alpha * grad(X, y, theta) 
        err_history[i] = err_function(X, y, theta)         
    return theta, err_history

n_iterations = 100
alpha = 0.0001
 
theta_optimal, err_history = gradient_descent(X, y, theta, alpha, n_iterations)

# La prédiction
predictions = modele(X, theta_optimal)

# Affiche des résultats 
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.show()
plt.plot(range(n_iterations), err_history)
plt.show()