import numpy as np
import math

def f1(X):
    x, y = X[0],X[1]
    return (x**2)*y

def f2(X):
    x, y = X[0],X[1]
    return 3*x + y**2

def g(X):
    x, y = X[0],X[1]
    return x**3 +2*x*y + y**2

def numerical_gradient_nd(f,X,h=1e-5):
    grad = np.zeros_like(X)

    for i in range(len(X)):
        x_nudge = X.copy()
        x_nudge[i] = x_nudge[i]+h
        grad[i] = (f(x_nudge)-f(X))/h

    return grad

def numerical_jacobian(f_list, X, h=1e-5):
    jocob_martix = np.zeros((len(f_list),len(X)))

    for i in range(len(f_list)):
        fn = f_list[i]
        grad  = numerical_gradient_nd(fn,X)
        jocob_martix[i]= grad
    
    return jocob_martix


X = np.array([2.0,3.0])


print(numerical_jacobian([f1,f2],X))


def numerical_hessian(g, X, h=1e-5):
    hessian_matrix = np.zeros((len(X),len(X)))


    for j in range(len(X)):
        ej = np.eye(len(X))[j]
        column = (numerical_gradient_nd(g,(X+h*ej)) - numerical_gradient_nd(g,X))/h
        hessian_matrix[:,j]  = column

    return hessian_matrix



print(numerical_hessian(g,X))