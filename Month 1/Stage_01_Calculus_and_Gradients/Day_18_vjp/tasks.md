

---

# 🧠 DAY 18: The Vector Chain Rule & Vector-Jacobian Products (VJP)

### 🛑 RULES FOR TODAY:
1. **The Dimension Match Rule:** You must verify on paper that the dimensions of your backward gradient vector $\nabla_x L$ exactly match the dimensions of your original input vector $x$. 
2. **The VJP implementation:** You will write a Python function that propagates an incoming gradient vector backward through a linear layer using the Vector-Jacobian Product formula ($W^T \cdot \nabla_y L$).

---

### 🎥 BLOCK 1: Theory (Vector-Jacobian Products)
To understand how PyTorch avoids storing massive matrices during backpropagation, search and watch:
*   [x] **Watch Video (Search on YT):** `Vector Jacobian Products PyTorch autodiff`  
    *(Working Link: [PyTorch Autodiff and Vector-Jacobian Products](https://www.youtube.com/watch?v=MswxJw-8PvE)) [3]*
    *(Pay close attention to why we multiply the vector on the left or the right of the Jacobian depending on row/column vector conventions!)*

---

### 📝 BLOCK 2: Pen & Paper Lab (The Vector Chain Rule)
*Open your notebook. Let's calculate how error flows backward through a layer to the inputs.*

We have:
*   Input vector: $x = \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}$ (dimension $2\times1$)
*   Weight Matrix: $W = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$ (dimension $2\times2$)
*   Output vector: $y = Wx$ (dimension $2\times1$)

Suppose the layer above sends us an incoming loss gradient with respect to $y$:
$$\nabla_y L = \begin{bmatrix} 4 \\ 2 \end{bmatrix}$$

*   [x] **Step 1:** Write down the transpose of the weight matrix, $W^T$.
*   [x] **Step 2:** Apply the VJP formula to calculate the gradient of the loss with respect to the input $x$:
    $$\nabla_x L = W^T \cdot \nabla_y L$$
    *(Evaluate this matrix-vector multiplication to get a $2\times1$ numerical vector).*
*   [x] **Step 3: Dimension Verification:** Does the shape of your calculated $\nabla_x L$ ($2\times1$) exactly match the shape of your starting input vector $x$ ($2\times1$)? (It must! You cannot add gradients to variables if their shapes don't match).

---

### 💻 BLOCK 3: Coding Lab (The VJP Engine)
*Create a file called `day18_vjp.py`.*

#### **Your Task:**
1.  **Define your inputs and weights in NumPy:**
    *   `W = np.array([[2.0, 1.0], [1.0, 3.0]])`
    *   `grad_y = np.array([4.0, 2.0])`
2.  **Implement `vjp_linear_backward(W, grad_y)`:**
    *   **Input:** The weight matrix `W` and the incoming gradient vector `grad_y`.
    *   **Output:** The backward gradient with respect to the input $x$ (`grad_x`).
    *   *Hint:* Implement the $W^T \cdot \nabla_y L$ formula using NumPy's transpose `.T` and matrix multiplication `@`.
3.  **Verify the Math:**
    *   Print your calculated `grad_x`.
    *   Confirm it matches your hand-written vector from Block 2!

---