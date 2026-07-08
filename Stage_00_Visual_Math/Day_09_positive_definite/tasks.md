
---

# 🧠 DAY 9: Symmetric Matrices, Minima & The Loss Landscape

### 🛑 RULES FOR TODAY:
1. **The Spectral Proof:** You must prove that a symmetric matrix can be diagonalized using its transpose instead of its inverse ($A = Q \Lambda Q^T$).
2. **The Definiteness Test:** You will write a Python function that tests if any symmetric matrix is Positive Definite (representing a stable minimum) or not.

---

### 🎥 BLOCK 1: Theory (MIT Lectures)
Watch these two lectures to understand symmetric space and how we mathematically define a "minimum":
*   [ ] **Watch MIT Lecture 25:** Symmetric Matrices and Orthogonal Eigenvectors.
*   [ ] **Watch MIT Lecture 27:** Positive Definite Matrices and Minima. *(Crucial: Focus on the connection between quadratic equations and positive eigenvalues).*

---

### 📝 BLOCK 2: Pen & Paper Lab (Testing Positive Definiteness)
Let's test our favorite symmetric matrix:
$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

1.  **Test 1: The Eigenvalue Test**
    *   We already know the eigenvalues are $\lambda_1 = 3$ and $\lambda_2 = 1$. Are they both strictly greater than 0?
2.  **Test 2: The Determinant (Pivot) Test**
    *   Check the top-left 1x1 determinant: $d_1 = |2| = 2$. Is it $> 0$?
    *   Check the full 2x2 determinant: $d_2 = |A| = (2\times2) - (1\times1) = 3$. Is it $> 0$?
3.  **Test 3: The Quadratic Form ($x^T A x > 0$)**
    *   Let $x = [x_1, x_2]^T$ be any non-zero vector. 
    *   Multiply out the quadratic form:
        $$x^T A x = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$
    *   Show that this algebraic expression ($2x_1^2 + 2x_1x_2 + 2x_2^2$) is always positive for any values of $x_1$ and $x_2$ (except when both are 0).

---

### 💻 BLOCK 3: Coding Lab (The "Loss Bowl" Surface Test)
*Create a file called `day9_positive_definite.py`.*

#### **Task 1: Prove the Spectral Theorem ($A = Q \Lambda Q^T$)**
Because $A$ is symmetric, its eigenvector matrix $S$ is actually orthogonal ($Q$). This means $Q^{-1}$ is just $Q^T$. Transposing a matrix takes **zero** computational time, while inverting a matrix is very slow. This is a massive speed hack in AI!

1.  Define a symmetric matrix in NumPy:
    `A = np.array([[4, 2], [2, 5]])`
2.  Calculate `eigenvalues, Q = np.linalg.eig(A)`.
3.  Reconstruct $A$ using the Spectral Theorem:
    `A_reconstructed = Q @ np.diag(eigenvalues) @ Q.T` *(Notice we use `Q.T` instead of `np.linalg.inv(Q)`!).*
4.  Verify `A_reconstructed` matches `A`.

#### **Task 2: Write the `is_positive_definite` Validator**
Write a function `is_positive_definite(matrix)` that takes a symmetric matrix and returns `True` if it is positive definite, and `False` otherwise.

*   *Rule to implement:* It must calculate the eigenvalues of the matrix. If **all** eigenvalues are strictly greater than 0, return `True`. Otherwise, return `False`.
*   **Test Matrix 1 (The Stable Bowl):** 
    `M1 = np.array([[2, 1], [1, 2]])` (Should return `True`).
*   **Test Matrix 2 (The Saddle Point):** 
    `M2 = np.array([[1, 2], [2, 1]])` (Should return `False` because its eigenvalues are 3 and -1).

---