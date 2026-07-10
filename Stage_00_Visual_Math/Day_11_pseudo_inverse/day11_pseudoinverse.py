import numpy as np

A = np.array([[1,2],
              [2,4]])

b = np.array([3,6])


def pseudo_inverse(A):
    U ,S ,Vt = np.linalg.svd(A)
    # S = np.diag(S)
    S_pluss = np.zeros((2,2))
    for i in range(len(S)):
        if S[i]> 0.00001:
            S_pluss[i,i] =    1/S[i] 
    A_pluss = Vt.T @ S_pluss @ U.T

    return A_pluss


A_plus = pseudo_inverse(A)

x_solved = A_plus @ b

print(f"""
MY pseudo inverse : {A_plus}
np pseudo inverse : {np.linalg.pinv(A)}

solved X ={x_solved}
""")