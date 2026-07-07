import numpy as np

A = np.array([[2, 1, 1],
              [4, 1, 0],
              [-2, 2, 1]])

b = np.array([1, -2, 7])


for i in range(len(b)):
    A_i = A.copy()
    A_i[:,i] = b
    det_a_i =np.linalg.det(A_i)
    det_A = np.linalg.det(A)
    x_i = det_a_i/det_A

    print(x_i)


print(np.linalg.solve(A,b))