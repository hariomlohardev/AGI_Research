import numpy as np

def f(X):
    x_0 ,x_1 ,x_2 = X[0],X[1],X[2]
    result = x_0**2 + 3*x_0*x_1 + x_2**3

    return result

print(f(np.array([2,4,3])))


def numerical_gradient_nd(f, X, h=1e-5):
    grad = np.zeros_like(X)
    
    for i in range(len(X)):
        x_nudge = X.copy()
        x_nudge[i] = x_nudge[i] +h
        grad[i] = (f(x_nudge)-f(X))/h

    return grad

print(numerical_gradient_nd(f,np.array([2.0,4.0,3.0])))