# -*- coding: utf-8 -*-
"""
Logistic Regression from scratch
Author: Senna van Iersel
"""

import numpy as np

# Sigmoid function to output an array with values between 0 and 1
# Input: Z with Zi = w.T * x + b
#               where:  w  = (Nx,1)
#                       x  = (Nx,1)
#                       b  = (1,1)
# Output: sigmoid(Z) = (Nx,1)
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Calculate the loss
# Inputs    y     = True label (0/1)
#           y_hat = Predicted probability (between 0 and 1)
def loss(Y, Y_hat):
    loss = - (Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return loss

# Calculate cost
# Inputs:   Y
#           Y_hat
def cost(Y, A, m):
    losses = loss(Y, A)
    return (1 / m) * np.sum(losses)

# Perform gradient descent - update weights and bias
# Inputs:   w     = weights (Nx,1)
#           b     = bias (1,1)
#           alpha = learning rate
def gradient_descent(w, b, alpha, dw, db):
    w = w - alpha * dw
    b = b - alpha * db
    return w, b

def initialization(X):
    # Row length of X
    w_len = X.shape[0]
    # Initialize weights, bias and cost
    w = dw = np.zeros((w_len,1))
    b = db = cost = 0
    return w, b, cost, dw, db

# Train the actual logistic regression model
def LogReg(X, Y, alpha = 0.5, num_iterations = 10000):
    X = X.T
    Y = Y.reshape(1, X.shape[1])
    # Initialize weights, bias and cost
    w, b, J, dw, db = initialization(X)
    m               = X.shape[1]
    for i in range(1,num_iterations):
        A   = sigmoid(np.dot(w.T,X) + b)
        J   = cost(Y, A, m)
        if i % 1000 == 0:
            print(i, J)
        dZ  = A - Y
        dw  = (1 / m) * np.dot(X,dZ.T)
        db  = (1 / m) * np.sum(dZ)
        w,b = gradient_descent(w,b,alpha,dw,db)
    print(J)
    
    