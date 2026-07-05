---

# 🧠 DAY 6: Gram-Schmidt & QR Decomposition (Straightening Data)

### 🛑 RULES FOR TODAY:
1. **The First-Principles Rule:** You must write the Gram-Schmidt loop manually in NumPy first. No using `np.linalg.qr` until you have proven your manual algorithm works.
2. **The Orthogonal Proof:** You must mathematically prove that $Q^T Q = I$ (the Identity Matrix) in your code.

---

### 🎥 BLOCK 1: Theory (MIT Lecture)
Watch this lecture to understand how to mathematically "straighten" space:
*   [ ] **Watch MIT Lecture 17:** Orthogonal Matrices and Gram-Schmidt.

---

### 📝 BLOCK 2: Pen & Paper Lab (Manual Gram-Schmidt)
Let's take two vectors in 2D space: **$a = [3, 1]$** and **$b = [2, 2]$**. They are not perpendicular. Let's make them orthonormal.

1.  **Keep the first vector as your base:**
    $$A = a = [3, 1]$$
2.  **Subtract the projection of $b$ onto $A$ from vector $b$:**
    $$B = b - \text{proj}_A(b) = b - \frac{A^T b}{A^T A} A$$
    *(Calculate this. Vector $B$ is now perfectly perpendicular to vector $A$).*
3.  **Normalize them (make their length equal to 1):**
    *   Divide $A$ by its length: $q_1 = \frac{A}{\|A\|}$
    *   Divide $B$ by its length: $q_2 = \frac{B}{\|B\|}$
4.  **Verify:** Calculate the dot product $q_1 \cdot q_2$. It must equal **0** (perpendicular).

---

### 💻 BLOCK 3: Coding Lab (The QR Factorization from Scratch)
*Create a file called `day6_qr.py`.*

You are going to write a function `gram_schmidt(A)` that performs this process for any size matrix.

#### **The Algorithm Design:**
For each column vector $v$ in matrix $A$:
1. Start with the original column vector.
2. Loop through all the orthogonal vectors you have already created, and subtract their projections from $v$.
3. Normalize the resulting vector and append it to your list of orthogonal vectors.

#### **The Tasks:**
1.  **Write the manual function:**
    ```python
    import numpy as np

    def gram_schmidt(A):
        # A is a 2D numpy array where columns are your vectors
        n, m = A.shape
        Q = np.zeros((n, m))
        
        # Write your loop logic here to fill Q with orthonormal columns
        # ...
        return Q
    ```
2.  **Calculate the $R$ Matrix:**
    Since $A = QR$, and $Q$ is orthogonal ($Q^T Q = I$), we can solve for $R$:
    $$R = Q^T A$$
    In your code, calculate `R = Q.T @ A`.
3.  **The Built-in Verification:**
    Use NumPy's built-in QR solver to verify your manual math:
    `q_np, r_np = np.linalg.qr(A)`
    Compare your `Q` and `R` with NumPy's `q_np` and `r_np`.
4.  **The Identity Proof:**
    Verify that `Q.T @ Q` equals the Identity Matrix (within floating-point limits).

---
