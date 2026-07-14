import numpy as np 

def u(x):
    return 3*x

def v(x):
    return x**2

def z(val_u ,val_v ):
    return val_u**2 + val_v**2

def numerical_chain_rule(x, h=1e-5):
    dz_du = (z(u(x)+h , v(x)) - z(u(x),v(x)))/h
    dz_dv = (z(u(x), v(x)+h) - z(u(x),v(x)))/h
    du_dx = (u(x+h) - u(x))/h
    dv_dx = (v(x+h) - v(x))/h
    dz_dx = (dz_du*du_dx) +(dz_dv*dv_dx)
    return dz_dx

print(numerical_chain_rule(2.0))
