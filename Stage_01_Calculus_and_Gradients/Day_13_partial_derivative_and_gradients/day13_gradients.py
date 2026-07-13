import math 
import numpy as np

def f(x,y):
    result = x**2*y + 2*x*y**2 + y**3
    return result

print(f(2,3))

def numerical_gradient(f, x, y, h=1e-5):
    df_dx = (f(x+h , y) -f(x,y))/h
    df_dy = (f(x , y+h) -f(x,y))/h

    return np.array([df_dx , df_dy])

print(numerical_gradient(f ,2,3))
    