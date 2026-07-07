import numpy as np

A = np.array([[1,2],
              [2,1]])

W_explode = np.array([[1.5, 0.5], [0.5, 1.5]])

W_vanish = np.array([[0.5, 0.2], [0.2, 0.5]])


def matrix_power(A ,n):
    eigne_values , S = np.linalg.eig(A)

    Lambda = np.diag(eigne_values)

    S_inverse = np.linalg.inv(S)

    A_n = S @ (Lambda ** n) @ S_inverse
    
    return A_n

print(f""" for W_explode
program calculated A_5 : {matrix_power(W_explode,50)}
via numpy : {np.linalg.matrix_power(W_explode, 50)}

{'-'*20}

for W_vanish
program calculated A_5 : {matrix_power(W_vanish,50)}
via numpy : {np.linalg.matrix_power(W_vanish, 50)}
""")