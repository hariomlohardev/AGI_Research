import numpy as np


def vjp_linear_backward(W : np.array, grad_y):
    grad_x = W.T @ grad_y
    return grad_x

W = np.array([[2,1],[1,3]])
grad_y = np.array([[4],[2]])

print(vjp_linear_backward(W ,grad_y))