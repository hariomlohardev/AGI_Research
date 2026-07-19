
---

# 🧠 DAY 8: Diagonalization, Matrix Powers & Deep Learning Gradients

### 🛑 RULES FOR TODAY:
1. **The Power Shortcut Rule:** You must prove that $A^k = S \Lambda^k S^{-1}$ in your code.
2. **The Gradient Simulation:** You will write a simulation that shows how eigenvalues $> 1$ blow up to infinity (exploding) and eigenvalues $< 1$ collapse to zero (vanishing) over 50 layers.

---

### 🎥 BLOCK 1: Theory (MIT Lectures)
Watch these two lectures to understand how eigenvalues govern what happens to data over time:
*   [ ] **Watch MIT Lecture 21:** Eigenvalues and Diagonalization.
*   [ ] **Watch MIT Lecture 22:** Diagonalization and Powers of A. *(Pay close attention to how he solves the Fibonacci sequence using matrix powers!)*

---

### 📝 BLOCK 2: Pen & Paper Lab (Calculating Matrix Powers)
Let's diagonalize our favorite matrix from Day 3:
$$A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$$

On Day 3, we calculated:
*   Eigenvalues: $\lambda_1 = 3$, $\lambda_2 = -1$
*   Eigenvectors: $v_1 = [1, 1]^T$, $v_2 = [1, -1]^T$

1.  **Construct the Eigenvector Matrix $S$:**
    $$S = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$
2.  **Construct the Diagonal Eigenvalue Matrix $\Lambda$:**
    $$\Lambda = \begin{bmatrix} 3 & 0 \\ 0 & -1 \end{bmatrix}$$
3.  **Find the Inverse $S^{-1}$ by hand:**
    $$S^{-1} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & -0.5 \end{bmatrix}$$
4.  **Calculate $A^3$ using the Diagonalization Shortcut:**
    $$A^3 = S \Lambda^3 S^{-1} = S \begin{bmatrix} 3^3 & 0 \\ 0 & (-1)^3 \end{bmatrix} S^{-1}$$
    *(Multiply this out on paper to find the final matrix $A^3$).*

---

### 💻 BLOCK 3: Coding Lab (The Exploding/Vanishing Gradient Simulation)
*Create a file called `day8_diagonalization.py`.*

We will simulate what happens to gradients as they pass through 50 layers of a neural network.

1.  **Task 1: Prove the Power Shortcut**
    *   Define $A = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$ in NumPy.
    *   Calculate `eigenvalues, S = np.linalg.eig(A)`.
    *   Construct the diagonal matrix `Lambda = np.diag(eigenvalues)`.
    *   Calculate $A^5$ using the shortcut: `A_power_5 = S @ (Lambda ** 5) @ np.linalg.inv(S)`.
    *   Verify it matches NumPy's built-in power function: `np.linalg.matrix_power(A, 5)`.
2.  **Task 2: Simulate Exploding Gradients (Eigenvalues > 1)**
    *   Define a weight matrix `W_explode` whose eigenvalues are greater than 1:
        `W_explode = np.array([[1.5, 0.5], [0.5, 1.5]])` (Eigenvalues are 2.0 and 1.0).
    *   Calculate `W_explode` raised to the power of 50. What happens to the numbers?
3.  **Task 3: Simulate Vanishing Gradients (Eigenvalues < 1)**
    *   Define a weight matrix `W_vanish` whose eigenvalues are less than 1:
        `W_vanish = np.array([[0.5, 0.2], [0.2, 0.5]])` (Eigenvalues are 0.7 and 0.3).
    *   Calculate `W_vanish` raised to the power of 50. What happens to the numbers?

---
