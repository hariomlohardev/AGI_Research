
---

# 🧠 DAY 17: Matrix Derivatives & The Backprop Equation

### 🛑 RULES FOR TODAY:
1. **The Outer Product Proof:** You must prove on paper that arranging the individual partial derivatives of the weights into a matrix is exactly equal to multiplying the output gradient vector by the transpose of the input vector ($\nabla_y L \cdot \vec{x}^T$).
2. **The Matrix Nudger:** You will write a nested loop in Python to nudge each individual element of a $2\times2$ weight matrix $W$ one-by-one to calculate the numerical matrix gradient.

---

### 🎥 BLOCK 1: Theory (Matrix Derivatives)
Watch this high-yield video explaining how backpropagation handles vectors and matrices under the hood:
*   [x] **Watch Video (Search on YT):** `Backpropagation with Vectors and Matrices Stanford CS231n`  
    *(Working Link: [CS231n Lecture 4 - Backpropagation with Vectors](https://www.youtube.com/watch?v=dB-u7o528gM))*
    *(Pay close attention to the shape matching of the gradients—the gradient of a matrix must always have the exact same shape as the original matrix!).*

---

### 📝 BLOCK 2: Pen & Paper Lab (The Matrix Backprop Proof)
*Open your notebook. Let's solve a $2\times2$ matrix derivative.*

We have an input vector $\vec{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$.
We have a weight matrix $W$:
$$W = \begin{bmatrix} w_{00} & w_{01} \\ w_{10} & w_{11} \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 1 & 1 \end{bmatrix}$$

The forward pass is $\vec{y} = W \vec{x}$:
$$y_0 = w_{00}x_0 + w_{01}x_1$$
$$y_1 = w_{10}x_0 + w_{11}x_1$$
*(Verify that for our starting values, $\vec{y} = [4, 5]^T$).*

Suppose our loss function is $L(\vec{y}) = y_0^2 + y_1^2$.

*   [x] **Step 1:** Calculate the gradient of the loss with respect to the output vector $\vec{y}$:
    $$\nabla_y L = \begin{bmatrix} \frac{\partial L}{\partial y_0} \\ \frac{\partial L}{\partial y_1} \end{bmatrix} = \begin{bmatrix} 2y_0 \\ 2y_1 \end{bmatrix}$$
    Evaluate this numerically at our point $\vec{y} = [4, 5]^T$.
*   [x] **Step 2:** Calculate the partial derivative of $L$ with respect to each weight in $W$ using the chain rule:
    *   $\frac{\partial L}{\partial w_{00}} = \frac{\partial L}{\partial y_0} \frac{\partial y_0}{\partial w_{00}}$
    *   $\frac{\partial L}{\partial w_{01}} = \frac{\partial L}{\partial y_0} \frac{\partial y_0}{\partial w_{01}}$
    *   $\frac{\partial L}{\partial w_{10}} = \frac{\partial L}{\partial y_1} \frac{\partial y_1}{\partial w_{10}}$
    *   $\frac{\partial L}{\partial w_{11}} = \frac{\partial L}{\partial y_1} \frac{\partial y_1}{\partial w_{11}}$
*   [x] **Step 3:** Evaluate these 4 partial derivatives numerically.
*   [x] **Step 4: The Matrix Proof:** Arrange these 4 numbers into a $2\times2$ matrix representing $\frac{\partial L}{\partial W}$:
    $$\frac{\partial L}{\partial W} = \begin{bmatrix} \frac{\partial L}{\partial w_{00}} & \frac{\partial L}{\partial w_{01}} \\ \frac{\partial L}{\partial w_{10}} & \frac{\partial L}{\partial w_{11}} \end{bmatrix}$$
    Show that this matrix is **exactly** equal to the matrix multiplication of our column vector $\nabla_y L$ and our row vector $\vec{x}^T$:
    $$\frac{\partial L}{\partial W} = (\nabla_y L) \cdot \vec{x}^T$$
    *(Keep this $2\times2$ matrix of integers to test your code!).*

---

### 💻 BLOCK 3: Coding Lab (The Matrix Gradient Validator)
*Create a file called `day17_matrix_gradients.py`.*

#### **Your Task:**
1.  **Define your inputs and functions:**
    *   Define `x = np.array([2.0, 3.0])`.
    *   Define your weight matrix `W = np.array([[2.0, 0.0], [1.0, 1.0]])`.
    *   Write a function `loss(W, x)` that:
        *   Computes `y = W @ x`.
        *   Returns the scalar loss: `np.sum(y ** 2)`.
2.  **Implement the Matrix Gradient Estimator:**
    *   Write a function `numerical_matrix_gradient(loss_fn, W, x, h=1e-5)` that:
        *   Initializes `grad_W = np.zeros_like(W)`.
        *   Uses a nested loop (looping through row `i` and column `j`) to nudge each element `W[i, j]` by `h` one-by-one.
        *   *Hint:* Make a copy of `W` inside the nested loop (`W_nudge = W.copy()`), add `h` to `W_nudge[i, j]`, calculate the slope `(loss_fn(W_nudge, x) - loss_fn(W, x)) / h`, and store it in `grad_W[i, j]`.
        *   Returns `grad_W`.
3.  **Run the Verification:**
    *   Call `numerical_matrix_gradient(loss, W, x)`. Print the result.
    *   *Note:* Ensure you handle the NumPy integer trap by making sure `W` is declared as floats!
    *   Confirm your printed $2\times2$ matrix matches your manual calculation from Block 2!

---