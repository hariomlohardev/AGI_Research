
---

# 🧠 DAY 7: Determinants, Cramer's Rule & Volume

### 🛑 RULES FOR TODAY:
1. **The Pure Math Rule:** Understand the algebraic definition of the Matrix Inverse using Cofactors:
   $$A^{-1} = \frac{1}{\det(A)} C^T$$
   *(No high-level shortcuts. You will solve this by hand first).*
2. **The Column Replacement Rule:** In your code, you must dynamically replace columns of a matrix using NumPy slicing to implement Cramer's Rule.

---

### 🎥 BLOCK 1: Theory (MIT Lectures)
Watch these three lectures to understand the core algebraic properties of determinants and space:
*   [x] **Watch MIT Lecture 18:** Properties of Determinants. *(The 3 foundational rules).*
*   [x] **Watch MIT Lecture 19:** Formulas for Determinants & Cofactors.
*   [x] **Watch MIT Lecture 20:** Cramer's Rule, Inverse Matrix, and Volume.

---

### 📝 BLOCK 2: Pen & Paper Lab (The Cofactor Inverse)
Let's find the inverse of a $2\times2$ matrix $A$ using only the Cofactor formula:
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

1.  **Calculate the Determinant:**
    $$\det(A) = (1\times4) - (2\times3) = -2$$
2.  **Find the Cofactor Matrix $C$:**
    *   For a 2x2 matrix, the cofactor matrix is:
        $$C = \begin{bmatrix} 4 & -3 \\ -2 & 1 \end{bmatrix}$$
3.  **Transpose the Cofactor Matrix ($C^T$):**
    $$C^T = \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix}$$
4.  **Calculate the Inverse:**
    $$A^{-1} = \frac{1}{\det(A)} C^T = -\frac{1}{2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}$$
5.  **Verify:** Multiply $A \times A^{-1}$ on paper. Do you get the Identity Matrix?

---

### 💻 BLOCK 3: Coding Lab (Cramer's Rule from Scratch)
*Create a file called `day7_determinants.py`.*

You are going to solve a $3\times3$ system of equations ($Ax = b$) using **Cramer's Rule**.

**The Math of Cramer's Rule:**
To find the $i$-th variable ($x_i$), you create a new matrix $A_i$ by replacing the $i$-th column of $A$ with the target vector $b$. Then:
$$x_i = \frac{\det(A_i)}{\det(A)}$$

#### **Your Task:**
1.  **Define your system in NumPy:**
    ```python
    import numpy as np

    A = np.array([[2, 1, 1],
                  [4, 1, 0],
                  [-2, 2, 1]])

    b = np.array([1, -2, 7])
    ```
2.  **Implement Cramer's Rule Loop:**
    Write a loop that:
    *   Loops through each column index `i` (0, 1, 2).
    *   Creates a copy of $A$ (use `A_i = A.copy()`).
    *   Replaces the $i$-th column of `A_i` with `b` (using slicing: `A_i[:, i] = b`).
    *   Calculates the determinant of `A_i` using `np.linalg.det(A_i)`.
    *   Divides it by `np.linalg.det(A)` to solve for $x_i$.
3.  **Verify:** Compare your Cramer's Rule output vector with `np.linalg.solve(A, b)`. They must match exactly!

---

### 📊 BLOCK 4: The Geometry of Volume
In Lecture 20, Strang proves that the determinant of a $3\times3$ matrix is exactly the **volume of the box (parallelepiped)** spanned by its column vectors.

1.  In `day7_determinants.py`, print the determinant of your matrix $A$.
2.  Add a comment in your code explaining: *If we have three vectors that lie on the same flat 2D sheet (coplanar), what will their determinant be, and why?* (Hint: What is the volume of a flat sheet?)

---
