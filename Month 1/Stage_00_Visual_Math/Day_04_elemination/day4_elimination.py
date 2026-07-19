import numpy as np

# 1. Define the system of equations
A = np.array([[2, 1, 1],
              [4, 1, 0],
              [-2, 2, 1]])

b = np.array([1, -2, 7])

# 2. Solve instantly
x = np.linalg.solve(A, b)
# print(f"The solution [x, y, z] is: {x}")

result = []
col_1 = A[:,0]
col_2 = A[:,1]
col_3 = A[:,2]

combination = (x[0]*col_1) +(x[1]*col_2) + (x[2]*col_3)

# print(combination)

from scipy.linalg import null_space

A = np.array([[1, 2],
              [3, 6]])

subspace = null_space(A)

result = A @ subspace

print(result)