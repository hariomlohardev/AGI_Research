
---

# 🧠 DAY 15: Jacobian & Hessian Matrices (Mapping Curve & Flow)

### 🛑 RULES FOR TODAY:
1. **No Completed Code:** You must write both matrix-building functions yourself without any libraries besides basic NumPy.
2. **The Verification Test:** I have not given you the final numerical matrices in this prompt. You must solve the math on paper first, then write the code, and prove they match!

---

### 🎥 BLOCK 1: Theory (Watch & Visualize)
Watch these two videos to lock in the physical curvature and error-flow intuition:
*   [ ] **Video 1 (Search on YT):** `Partial derivatives, introduction Khan Academy`  
    *(Working Link: https://www.youtube.com/watch?v=AXqhWeUEtQU)*
*   [ ] **Video 2 (Search on YT):** `Gradient Khan Academy`  
    *(Working Link: https://www.youtube.com/watch?v=tIpKfDc295M)*
*   [ ] **Video 3 (Search on YT):** `The Hessian matrix Multivariable calculus Khan Academy`  
    *(Working Link: https://www.youtube.com/watch?v=n9hGhgulnWw)*

---

### 📝 BLOCK 2: Pen & Paper Lab (Analytical Derivatives)
*Grab your notebook. Solve these equations to find your target verification matrices.*

**Problem 1: The Jacobian Matrix (Error-Flow Grid)**
You have a 2-output vector function $\vec{f}(X) = [f_1(X), f_2(X)]^T$ where:
$$f_1(X) = x_0^2 x_1$$
$$f_2(X) = 3x_0 + x_1^2$$

*   [ ] **Step 1:** Calculate the algebraic partial derivatives: $\frac{\partial f_1}{\partial x_0}$, $\frac{\partial f_1}{\partial x_1}$, $\frac{\partial f_2}{\partial x_0}$, and $\frac{\partial f_2}{\partial x_1}$ on paper.
*   [ ] **Step 2:** Construct the algebraic Jacobian matrix:
    $$J(X) = \begin{bmatrix} \frac{\partial f_1}{\partial x_0} & \frac{\partial f_1}{\partial x_1} \\ \frac{\partial f_2}{\partial x_0} & \frac{\partial f_2}{\partial x_1} \end{bmatrix}$$
*   [ ] **Step 3:** Evaluate $J(X)$ at the point **$X = [2.0, 3.0]$** to get a $2\times2$ matrix of integers. Keep this to verify your code.

**Problem 2: The Hessian Matrix (Curvature of Error Surface)**
You have a single scalar-valued loss function:
$$g(X) = x_0^3 + 2x_0 x_1 + x_1^2$$

*   [ ] **Step 1:** Calculate all second-order partial derivatives: $\frac{\partial^2 g}{\partial x_0^2}$, $\frac{\partial^2 g}{\partial x_1^2}$, and the mixed partials $\frac{\partial^2 g}{\partial x_0 \partial x_1}$ and $\frac{\partial^2 g}{\partial x_1 \partial x_0}$.
*   [ ] **Step 2:** Construct the algebraic Hessian matrix $H(X)$. Verify that the mixed partials are equal, making the Hessian symmetric.
*   [ ] **Step 3:** Evaluate $H(X)$ at the point **$X = [2.0, 3.0]$** to get a $2\times2$ matrix of integers. Keep this to verify your code.

---

### 💻 BLOCK 3: Python Coding Lab (in `day15_matrices.py`)
*Create a file called `day15_matrices.py` and write your numerical matrix solvers.*

#### **Task 1: Define your functions**
*   Write `f1(X)` and `f2(X)` to match Problem 1.
*   Write `g(X)` to match Problem 2.
*   Paste your `numerical_gradient_nd` function from yesterday—you will need it!

#### **Task 2: Code `numerical_jacobian(f_list, X, h=1e-5)`**
*   **Input:** A list of functions `f_list` (e.g., `[f1, f2]`) and a 1D NumPy array `X`.
*   **Output:** A 2D NumPy array representing the Jacobian.
*   *Hint:* The $i$-th row of the Jacobian is simply the gradient of the $i$-th function in `f_list`. Loop through the functions, calculate their gradient at `X` using your Day 14 code, and append/stack them.

#### **Task 3: Code `numerical_hessian(g, X, h=1e-5)`**
*   **Input:** A scalar function `g` and a 1D NumPy array `X`.
*   **Output:** A 2D NumPy array representing the Hessian.
*   *Hint:* The Hessian is just the Jacobian of the gradient vector! To find the $j$-th column of the Hessian, you can calculate how much the *entire gradient vector* changes when you nudge $x_j$ by $h$:
    $$\text{column}_j = \frac{\nabla g(X + h \cdot e_j) - \nabla g(X)}{h}$$
    Where $e_j$ is a unit vector pointing in the $j$-th coordinate direction.

#### **Task 4: Run the Verification**
*   Call `numerical_jacobian([f1, f2], np.array([2.0, 3.0]))` and print.
*   Call `numerical_hessian(g, np.array([2.0, 3.0]))` and print.
*   Verify that both printed $2\times2$ matrices match your hand-written math!

---
