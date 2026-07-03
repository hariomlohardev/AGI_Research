
---

# 🧠 DAY 3: The Hidden Skeleton of Data (Eigenvalues & Eigenvectors)

### 🛑 RULES FOR TODAY:
1. **The Core Equation:** Today you will learn the most important equation in AI: 
   $$A \vec{v} = \lambda \vec{v}$$
   *(Matrix $\times$ Vector = Number $\times$ Vector)*
2. **The Visual Goal:** Understand why some vectors "stay still" while the rest of the world rotates around them.

---

### 🎥 BLOCK 1: Theory (Watch & Visualize)
Go to YouTube -> **3Blue1Brown - Essence of Linear Algebra**.
*   [x] **Watch Video 14:** Eigenvalues and eigenvectors.
*   **The AGI Intuition:** When a Neural Network transforms data (rotates/stretches space), almost all vectors get knocked off their original path. **Eigenvectors** are the special vectors that **do not change direction**. They only stretch or shrink. The amount they stretch is called the **Eigenvalue ($\lambda$)**. They represent the "axes" of your data.

---

### 📝 BLOCK 2: Pen & Paper Lab (12th Grade Revision)
*Let's solve the characteristic equation on paper.*

You have Matrix $A$:
$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

1.  **Find the Eigenvalues ($\lambda$):**
    *   Solve the equation: $\det(A - \lambda I) = 0$
    *   This becomes: $\det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix} = 0$
    *   Solve the quadratic equation: $(2-\lambda)(2-\lambda) - 1 = 0$
    *   **Find the two values of $\lambda$** (Hint: they are integers).
2.  **Find the Eigenvector ($\vec{v}$) for the largest $\lambda$:**
    *   Plug your largest $\lambda$ back into $(A - \lambda I)\vec{v} = 0$ and solve for vector $\vec{v} = [x, y]$.

---

### 💻 BLOCK 3: Python Coding Lab (in `day3_eigen.py`)
*Now, we let NumPy do the heavy lifting, but we will PROVE the math.*

1.  **Task 1: Calculate with NumPy**
    *   Create matrix `A = np.array([[2, 1], [1, 2]])`.
    *   Use `eigenvalues, eigenvectors = np.linalg.eig(A)` to calculate them instantly.
    *   Print both. Do the eigenvalues match your paper math? (Note: NumPy normalizes eigenvectors to have a length of 1, so they might look like decimals like `0.7071`).
2.  **Task 4: The Ultimate Proof ($A \vec{v} = \lambda \vec{v}$)**
    *   Let's prove the core equation of AI.
    *   Take your matrix `A`.
    *   Take the first eigenvector: `v = eigenvectors[:, 0]` (Column 0).
    *   Take the first eigenvalue: `lam = eigenvalues[0]`.
    *   **Calculate Left Side:** `LHS = A @ v`
    *   **Calculate Right Side:** `RHS = lam * v`
    *   **The Proof:** Print both `LHS` and `RHS`. They should be **exactly identical!**

---

### 🚀 BLOCK 4: The AGI Application (Image Compression concept)
Why do we care about eigenvectors in AGI? Because of **Dimensionality Reduction (PCA)**. 

If a dataset has 1,000 features, it's too big for an AI to process quickly. We calculate the eigenvectors of the data. The eigenvectors with the **largest eigenvalues** contain the most information. We throw away the rest. 

*   [x] In `day3_eigen.py`, write a short comment explaining in your own words: *How does an AI use eigenvectors to compress data?*

---
