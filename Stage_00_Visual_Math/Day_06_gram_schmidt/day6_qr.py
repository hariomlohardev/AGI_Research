import numpy as np

def gram_schmidt(A):
    n, m = A.shape
    Q = np.zeros((n, m))
    print(Q)
    
    # Loop through every column of A (index j)
    for j in range(m):
        # 1. Start with the original column vector
        v = A[:, j].astype(float) 
        
        # 2. Subtract projections of all PREVIOUS columns we already solved (index i)
        for i in range(j):
            q_prev = Q[:, i]
            print(q_prev,f'i : {j}')
            # Calculate projection of v onto q_prev
            # Since q_prev already has a length of 1, the projection formula is very simple:
            # proj = dot_product(v, q_prev) * q_prev
            proj = np.dot(v, q_prev) * q_prev

            print(proj)
            
            # Subtract the projection to make v orthogonal
            v = v - proj
            
        # 3. Normalize the vector (make its length = 1)
        # HINT: Divide v by its norm using np.linalg.norm(v)
        Q[:, j] = v / np.linalg.norm(v)
        
    return Q

A = np.array([[1,2],
              [3,4]])

# 1. Run your function to get Q
Q = gram_schmidt(A)

# 2. Calculate the upper triangular matrix R
R = Q.T @ A

# 3. Prove that Q @ R reconstructs your original matrix A!
A_reconstructed = Q @ R

# 4. Prove that Q is orthogonal (Q.T @ Q should be the Identity Matrix)
identity_proof = Q.T @ Q

# 5. Run NumPy's built-in QR solver to compare
q_np, r_np = np.linalg.qr(A)

print("\n--- RESULTS ---")
print("Your Q:\n", Q)
print("NumPy's Q:\n", q_np)
print("\nYour R:\n", R)
print("NumPy's R:\n", r_np)
print("\nReconstructed A (Q @ R):\n", A_reconstructed)
print("\nIdentity Proof (Q.T @ Q):\n", np.round(identity_proof, 5))