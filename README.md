
# 🧠 First Principles AI & AGI Roadmap
**Author:** Hariom lohar | **Location:** India | **Goal:** Building AGI systems from scratch in 18 months.

### 🎯 Objective
This repository documents my daily, 8-hour deep work journey from 12th-grade mathematics to advanced Artificial General Intelligence (AGI) research. I do not rely on high-level libraries until I have built the underlying mathematical engines from scratch in pure Python and NumPy.

### 📅 Progress Tracker
- [x] **Stage 0:** Visual Intuition & Linear Algebra *(In Progress)*
- [x] **Stage 1:** Calculus & Gradients
- [ ] **Stage 2:** Probability & Statistics 
- [ ] **Stage 3:** Machine Learning Algorithms (Scratch)
- [ ] **Stage 4:** Deep Learning & Neural Networks
- [ ] **Stage 5:** Reinforcement Learning & Agents
- [ ] **Stage 6:** Transformers & LLMs

### 🛠️ Daily Log
*   **Day 1:** Coded Vector addition, Dot products, and Matrix Multiplication from scratch using pure Python. Built the foundation for linear transformations.
*   **Day 2:** Mastered NumPy, matrix determinants, and benchmarked C-optimized vectorization (NumPy was 759x faster than pure Python!). Built matrix inverse operations.
*   **Day 3:** Mastered Eigenvalues & Eigenvectors. Mathematically proved the core AI equation $A\vec{v} = \lambda\vec{v}$ in NumPy and learned the theory of data compression/dimensionality reduction (PCA). **Graduated Stage 0!**
*   **Day 4:** Started MIT Linear Algebra. Mastered the "Column Picture" (linear combinations/Transformer Attention intuition) and used `np.linalg.solve`. Mastered Vector Spaces, Subspaces, and calculated the "Null Space" (blind spots of matrices) using SciPy, proving $A \times \text{NullSpace} = 0$.
*   **Day 5:** Mastered Orthogonality, Vector Projections, and the Least Squares method. Coded the **Normal Equation** $(\theta = (X^T X)^{-1} X^T y)$ from scratch using NumPy to build a Linear Regression model, and visualized the line of best fit using Matplotlib.

*   **Day 6:** Coded the **Gram-Schmidt Orthogonalization** algorithm and **QR Decomposition** ($A = QR$) from scratch using NumPy. Mathematically proved that $Q$ is orthogonal ($Q^T Q = I$) and reconstructed the original matrix $A = QR$ with zero numerical loss.
*   **Day 7:** Mastered the algebraic properties of Determinants & Cofactors. Coded **Cramer's Rule** from scratch in NumPy to solve a $3\times3$ system of equations using only determinant calculations, and verified its accuracy against `np.linalg.solve`.
*   **Day 8:** Mastered Matrix Diagonalization ($A = S\Lambda S^{-1}$) and calculated matrix powers. Wrote a dynamic simulation proving the **Vanishing & Exploding Gradients** problem in Deep Learning—showing how eigenvalues $\lambda > 1$ cause exponential explosion ($10^{14}$) and $\lambda < 1$ cause exponential collapse ($10^{-9}$) over 50 layers.
*   **Day 9:** Mastered Symmetric Matrices & Positive Definite Matrices. Proved the **Spectral Theorem** ($A = Q\Lambda Q^T$) in NumPy using orthogonal transpose operations. Coded a robust Positive Definite matrix validator to evaluate if a multi-dimensional error surface represents a stable minimum (convex bowl) or an unstable saddle point. 
*   **Day 10:** Mastered Singular Value Decomposition (SVD) ($A = U\Sigma V^T$). Calculated SVD elements manually for non-square matrices on paper, and coded an SVD-based Image Compressor in NumPy utilizing Low-Rank Approximation (truncating singular values to $k$) to compress data with zero perceptual loss.
*   **Day 11:** Mastered the **Moore-Penrose Pseudo-Inverse** ($A^+$) using SVD ($A^+ = V \Sigma^+ U^T$). Built a robust pseudo-inverse solver from scratch in NumPy, implemented a numerical threshold filter to handle floating-point noise/division-by-zero traps, and successfully solved a singular, non-invertible system of equations.
*   **Day 12 (Stage 1 Graduation Project):** Built a custom **Principal Component Analysis (PCA)** class from scratch in NumPy using SVD. Successfully compressed a 4-dimensional biological dataset of flower features down to 2D, and visualized the natural clustering of distinct species using Matplotlib. **Graduated Stage 1!**
*   **Day 12.5 (Independent Challenge):** Independently built a Handwritten Digits (MNIST-lite) classifier in Google Colab. Programmed the entire data splitting, mean-centering, SVD projection, and Euclidean distance-matching pipeline. Wrote an automated evaluation loop from scratch to test 270 unseen digits and calculated the final system accuracy.
*   **Day 13 (Stage 2 Start):** Mastered Partial Derivatives and the Gradient Vector ($\nabla f$). Coded a **Numerical Gradient Estimator** from scratch in Python using the Finite Difference Method (approximating multi-dimensional slopes by nudging variables by $10^{-5}$) and mathematically verified its accuracy against algebraic calculus.
*   **Day 14:** Scaled calculus to $N$-Dimensions by writing a **Dynamic Vectorized Gradient Estimator** in NumPy. The algorithm uses deep vector copies and dynamic index-nudging to calculate the complete multidimensional slope vector ($\nabla f$) for inputs of any arbitrary size.
*   **Day 15:** Mastered Jacobian & Hessian Matrices. Coded a dynamic **Numerical Jacobian** (mapping layer-to-layer error flow) and a **Numerical Hessian** (measuring local loss surface curvature) from scratch in NumPy using second-order finite differences and vector coordinate perturbations (`np.eye`).
*   **Day 16:** Mastered the **Multivariable Chain Rule**. Coded a multi-path backpropagation simulator from scratch in NumPy, mathematically proving that when an error signal splits across multiple branching pathways, we must sum the gradients of all paths ($\frac{dz}{dx} = \frac{\partial z}{\partial u}\frac{du}{dx} + \frac{\partial z}{\partial v}\frac{dv}{dx}$) to find the true total derivative.
*   **Day 18:** Mastered the **Vector-Jacobian Product (VJP)** and the Vector Chain Rule. Coded a lightweight backpropagation layer backward pass ($W^T \cdot \nabla_y L$) in NumPy, demonstrating how deep learning frameworks propagate gradients backward across layers without storing giant, memory-intensive Jacobian matrices.
*   **Day 19 (The Autograd Engine):** Built a custom **Scalar Automatic Differentiation Engine (Autograd)** from scratch in Python. Implemented a `Value` class with dynamic graph construction (DAG), magic methods (`__add__`, `__mul__`, `__sub__`), a post-order topological sorting algorithm (DFS) to traverse the computational graph, and automatic backpropagation.

*   **Day 20 (Stage 2 Graduation Project):** Fully completed the custom **Scalar Autograd Engine**. Built object-oriented classes for `Neuron`, `Layer`, and `MLP` (Multilayer Perceptron) using the scratch-built `Value` class, recursively gathered all 41 network parameters, and successfully trained the MLP on non-linear dataset targets, dropping loss to near-zero. **Graduated Stage 2!**

*   **Day 23:** Scaled the optimization framework by implementing **SGD with Momentum** and **RMSprop** from scratch. Extended the `Value` class to track historical velocity and squared gradient variance, allowing the training loop to dynamically scale and smooth weight updates on steep loss surfaces.