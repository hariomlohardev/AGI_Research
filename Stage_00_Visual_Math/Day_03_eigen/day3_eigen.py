import numpy as np


A = np.array([[1,2],[2,1]])

eigenvalue  ,eigenvector = np.linalg.eig(A)

print(f"""eigne_vector : {eigenvector}
      eigne_value : {eigenvalue}""")

v = eigenvector[:,0]

lam = eigenvalue[0]

LHS =A @ v

RHS = lam * v

print(LHS , RHS)