
---

# 🧠 DAY 4: MIT Linear Algebra – Gaussian Elimination from Scratch

### 🛑 RULES FOR TODAY:
1. **The No-Cheat Rule:** You are **not** allowed to use `np.linalg.solve` or any built-in solver today. You must write the actual algebraic row-reduction algorithm yourself.
2. **The Rigor Rule:** You must take detailed, written notes on the difference between the "Row Picture" and the "Column Picture" of matrix multiplication.

---

### 🎥 BLOCK 1: Theory (MIT Lectures)
Watch the first two lectures of MIT 18.06. Take physical notes. If you don't understand a concept, rewind. This is Professor Gilbert Strang at his best.
*   [ ] **Lecture 1:** The Geometry of Linear Equations (Row picture vs. Column picture).
*   [ ] **Lecture 2:** Elimination with Matrices (Gaussian Elimination, Elimination Matrices $E_{ij}$, and Permutations).

---

### 📝 BLOCK 2: Pen & Paper Lab (MIT Level Math)
*Before you code, solve this exact system on paper using Gaussian Elimination.*

You have the system of equations:
$$2x + y + z = 1$$
$$4x + y = -2$$
$$-2x + 2y + z = 7$$

1.  **Write the Augmented Matrix $[A | b]$:**
    $$[A | b] = \begin{bmatrix} 2 & 1 & 1 & | & 1 \\ 4 & 1 & 0 & | & -2 \\ -2 & 2 & 1 & | & 7 \end{bmatrix}$$
2.  **Perform Row Operations by hand:**
    *   Find the first pivot (Row 1, Column 1 = `2`).
    *   Eliminate the `4` in Row 2 (Subtract $2 \times$ Row 1 from Row 2).
    *   Eliminate the `-2` in Row 3 (Add $1 \times$ Row 1 to Row 3).
    *   Find the second pivot and eliminate the rest.
3.  **Back-Substitution:**
    *   Once you have the upper triangular matrix $U$, solve for $z$, then $y$, then $x$.
    *   **Keep your final answers $[x, y, z]$** to test your code later.
4.  **Find the Elimination Matrices:**
    *   Write down the $3\times3$ Elimination Matrix **$E_{21}$** that performs the first step (subtracting $2 \times$ Row 1 from Row 2).

---

### 💻 BLOCK 3: The Coding Challenge (in `day4_elimination.py`)
*This is the big one. You must write an algorithm that can take ANY 3x3 matrix and solve it using Gaussian Elimination.*

**The Requirements for `gaussian_elimination(A, b)`:**
*   **Input:** A $3\times3$ NumPy array `A`, and a $3\times1$ NumPy array `b`.
*   **The Elimination Phase:** Your code must dynamically calculate the multipliers, perform row operations on both `A` and `b`, and convert `A` into an **Upper Triangular Matrix $U$** (all zeros below the main diagonal).
*   **The Back-Substitution Phase:** Your code must loop backward from the bottom row to the top row, solving for the variables one by one.
*   **Output:** A list or NumPy array `[x, y, z]` with the correct solutions.

---

### 🚀 BLOCK 4: The Operator Verification
In Lecture 2, Strang teaches that row operations are actually just **multiplying $A$ by an Elimination Matrix $E$**.

1.  Create the original matrix `A` in NumPy:
    ```python
    A = np.array([[2, 1, 1],
                  [4, 1, 0],
                  [-2, 2, 1]])
    ```
2.  Create the Elimination Matrix **$E_{21}$** that you calculated on paper in Block 2.
3.  Multiply them in NumPy: `result = E21 @ A`.
4.  Verify that `result` has a `0` in the Row 2, Column 1 position (just like your first step of hand-written math).

---
