# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:27:21 2019
@author: SRINIDHI
"""
import matplotlib.pyplot as plt
import numpy as np
#import genfromtxt from numpy
data = np.genfromtxt('data.csv', delimiter=',')

def Costcomputation(A, B, theta):
    value = np.power(((A @ theta.T) - B), 2) # @ is the matrix multiplication of arrays. In order to use * for multiplication we should convert all arrays to matrices!
    #value calculates dotproduct of X and theta raised to power 2.
    return np.sum(value) / (2 * len(A))
#gradientDescent function is created to minimize cost function.
def gradientDescent(A, B, theta, alpha, iterations):
    for i in range(iterations):
        
        theta = theta - (alpha/len(A)) * np.sum((A @ theta.T - B) * A, axis=0)
        cost = Costcomputation(A, B, theta)
        
    return (theta, cost)

# consider small alpha value
alpha = 0.0002
iterations = 1500

#A is columns
A = data[:, 0].reshape(-1,1) # -1 tells numpy to figure out the dimension by itself
ones = np.ones([A.shape[0], 1])
A = np.concatenate([ones, A],1)

# theta is a row vector
theta = np.array([[1.0, 1.0]])

#B is a columns vector
B =data[:, 1].reshape(-1,1)

g, cost = gradientDescent(A, B, theta, alpha, iterations)  
print(g, cost)

plt.scatter(data[:, 0].reshape(-1,1), B)
axes = plt.gca()
x_values = np.array(axes.get_xlim()) 
y_values = g[0][0] + g[0][1]* x_values #Line equation
plt.plot(x_values, y_values, '-')
