import numpy as np

M = np.array([
    [1, 2],
    [3, 4]
])


INVERSE =np.linalg.inv(M)

identity_matrix = INVERSE @ M

print(identity_matrix)

