import numpy as np 

x = np.array([2.0 , 3.0])

W = np.array([[2.0,0.0],
             [1.0,1.0]])

def loss(W ,x):
    y= W @ x
    loss = np.sum(y ** 2)

    return loss


def numerical_matrix_gradient(loss_fn, W, x, h=1e-5):
    grad_W = np.zeros_like(W)
    i_W ,j_W  = grad_W.shape
    for i in range(i_W):

        for j in range(j_W):
            W_nudge = W.copy()
            W_nudge[i,j] = W_nudge[i,j]  + h
            slop =  (loss_fn(W_nudge , x) - loss_fn(W , x))/ h
            grad_W[i,j]  = slop

    return grad_W

print(numerical_matrix_gradient(loss, W, x))
    