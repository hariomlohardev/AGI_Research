
---

# 🧠 DAY 11: The Pseudo-Inverse & Solving Unsolvable Systems

### 🛑 RULES FOR TODAY:
1. **The No-Crash Rule:** You must prove that you can solve a singular, non-invertible system that causes standard inverse functions to crash.
2. **The First-Principles SVD Construction:** You will build the $A^+$ matrix manually using your SVD outputs and verify it against NumPy's built-in `np.linalg.pinv()`.

---

### 🎥 BLOCK 1: Theory (MIT Lectures)
Watch this final lecture on SVD applications to understand how we map the pseudo-inverse geometrically:
*   [ ] **Watch MIT Lecture 33:** The Pseudo-Inverse. *(Pay close attention to how it maps the row space to the column space, bypassing the null space).*

---

### 📝 BLOCK 2: Pen & Paper Lab (Calculating $A^+$)
Let's find the pseudo-inverse of the rectangular $3\times2$ matrix $A$ from yesterday:
$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$$

Recall your Day 10 SVD results:
*   $U = I_{3\times3}$ (Identity matrix)
*   $\Sigma = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$
*   $V^T = I_{2\times2}$ (Identity matrix)

1.  **Construct $\Sigma^+$ ($2\times3$):**
    *   Transpose $\Sigma$ to get a $2\times3$ matrix.
    *   Replace the non-zero elements ($3$ and $2$) with their reciprocals ($1/3$ and $1/2$).
2.  **Calculate the Pseudo-Inverse $A^+$ ($2\times3$):**
    *   Multiply: $A^+ = V \Sigma^+ U^T$
3.  **The Identity Check:**
    *   Multiply $A \times A^+ \times A$ on paper. 
    *   **The Proof:** Show that this multiplication successfully returns your original matrix $A$. (In SVD theory, this is the defining property of a pseudo-inverse).

---

### 💻 BLOCK 3: Coding Lab (Solving the "Unsolvable" System)
*Create a file called `day11_pseudoinverse.py`.*

We want to solve the system $Ax = b$ where $A$ is a singular, non-invertible matrix:
$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$
*(Determinant is 0. Standard inversion is impossible).*

Let the target vector be $b = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$.

#### **Your Task:**
1.  **Prove the Crash:**
    *   Write a `try/except` block attempting to run `np.linalg.inv(A)`. Print the error message to prove standard math fails.
2.  **Decompose with SVD:**
    *   Run SVD on $A$ to get `U`, `S`, and `Vt`.
3.  **Construct $\Sigma^+$ manually:**
    *   Create a $2\times2$ matrix of zeros (the same shape as $A^T$).
    *   For each non-zero singular value in `S`, place its reciprocal ($1/S_i$) on the diagonal of your zero matrix to form `Sigma_plus`.
4.  **Build $A^+$ from scratch:**
    *   Multiply the SVD elements together using your custom `Sigma_plus`:
        $$A^+ = V \Sigma^+ U^T$$
        *   *Hint:* Remember that `np.linalg.svd` returns `Vt` (which is $V^T$), so you need to transpose it back to get $V$ before multiplying.
5.  **Solve the System:**
    *   Calculate the solution vector: `x_solved = A_plus @ b`.
6.  **Verify:**
    *   Verify your custom $A^+$ matches NumPy's built-in pseudo-inverse: `np.linalg.pinv(A)`. 

---
