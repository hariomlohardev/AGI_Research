
---

# 🧠 DAY 10: Singular Value Decomposition & Image Compression

### 🛑 RULES FOR TODAY:
1. **The Pure Math Setup:** You must calculate the eigenvalues of $A^T A$ to find the singular values of a non-square matrix on paper first.
2. **The Compression Task:** You will write a Python function that reconstructs an image using only the top $k$ singular values.

---

### 🎥 BLOCK 1: Theory (MIT Lectures)
Watch these two lectures to understand how SVD maps any rectangular space:
*   [x] **Watch MIT Lecture 29:** Singular Value Decomposition. *(Understand the math of $U \Sigma V^T$).*
*   [x] **Watch MIT Lecture 30:** SVD and Principal Component Analysis. *(How SVD is used to find the most important directions of data).*

---

### 📝 BLOCK 2: Pen & Paper Lab (Calculating $U \Sigma V^T$)
Let's find the SVD of a simple rectangular $3\times2$ matrix $A$:
$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$$

Since $A$ is $3\times2$, our SVD matrices must be:
*   $U$ dimensions: $3\times3$
*   $\Sigma$ dimensions: $3\times2$
*   $V^T$ dimensions: $2\times2$

1.  **Find $V$ and $\Sigma$ (using $A^T A$):**
    *   Calculate the $2\times2$ symmetric matrix $A^T A$.
    *   Find the eigenvalues ($\lambda_1, \lambda_2$) of $A^T A$.
    *   The **singular values** ($\sigma_1, \sigma_2$) of $A$ are the square roots of these eigenvalues: $\sigma = \sqrt{\lambda}$.
    *   Find the normalized eigenvectors of $A^T A$. These eigenvectors are the columns of $V$.
2.  **Find $U$ (using the SVD relationship):**
    *   For any non-zero singular value, the columns of $U$ are calculated as:
        $$u_i = \frac{A v_i}{\sigma_i}$$
    *   Calculate $u_1$ and $u_2$ using this formula. (The third column $u_3$ will be orthogonal to the first two).

---

### 💻 BLOCK 3: Coding Lab (SVD Image Compressor)
*Create a file called `day10_svd.py`.*

Your goal is to load a real grayscale image, perform SVD, compress it by keeping only the top $k$ singular values, and reconstruct it.

#### **Your Task:**
1.  **Load a Grayscale Image:**
    Use `matplotlib.image.imread` or `PIL` to load any image on your computer. If it is an RGB image, convert it to grayscale by averaging the color channels.
2.  **Run SVD using NumPy:**
    Use `U, S, Vt = np.linalg.svd(image_matrix)` to decompose your image matrix. 
    *   *Note:* NumPy returns `S` as a flat 1D array of singular values (not a 2D diagonal matrix).
3.  **Write the Compression Function:**
    Write a function `compress_image(U, S, Vt, k)` that:
    *   Takes the $U$, $S$, and $V^T$ matrices, along with an integer $k$ (the number of singular values to keep).
    *   Reconstructs the image matrix using **only** the first $k$ columns of $U$, the first $k$ singular values in $S$, and the first $k$ rows of $V^T$.
    *   *Hint:* You will need to slice the matrices (e.g., `U[:, :k]`, `Vt[:k, :]`) and use matrix multiplication `@` to reconstruct.
4.  **Visualize the Compression Levels:**
    Use `plt.subplot()` to plot the original image next to the compressed images at $k = 5, 20,$ and $50$. 

---
