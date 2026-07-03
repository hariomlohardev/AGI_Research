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


## How does an AI use eigenvectors to compress data?
'''AI uses eigenvectors when the dataset is too large and we want to compress it down
 while remaing all the important information then we use eigenvectors then one with highest eignevalue'''