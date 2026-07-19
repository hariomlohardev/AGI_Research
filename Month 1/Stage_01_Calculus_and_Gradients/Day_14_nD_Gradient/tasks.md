
---

# 🧠 DAY 14: Multidimensional Slopes (The Vectorized Gradient)

### 🛑 RULES FOR TODAY:
1. **The $N$-Dimension Rule:** Your gradient function must be completely dynamic. It should work for a 3-dimensional vector, a 10-dimensional vector, or a 1,000-dimensional vector.
2. **The Safe Copy Rule:** When nudging the $i$-th element of a vector in Python, you must make a deep copy (or `.copy()`) of the vector so you don't permanently alter the original inputs.

---

### 🎥 BLOCK 1: Theory (The Vectorized Gradient)

Imagine we represent our inputs as a single vector $X$:
$$X = [x_0, x_1, x_2, \dots, x_{n-1}]$$

If we have a function $f(X)$ that takes this vector and returns a single number, the gradient $\nabla f(X)$ is a vector of the exact same size:
$$\nabla f(X) = \begin{bmatrix} \frac{\partial f}{\partial x_0} \\ \frac{\partial f}{\partial x_1} \\ \dots \\ \frac{\partial f}{\partial x_{n-1}} \end{bmatrix}$$

To calculate this numerically in a computer:
1. Initialize an empty gradient array of the same shape as $X$ filled with zeros.
2. Loop through each index $i$ from $0$ to $n-1$.
3. Create a temporary copy of $X$ and add a tiny $h$ (like `1e-5`) **only** to the $i$-th element.
4. Calculate the slope at that specific index using the finite difference method:
   $$\text{grad}[i] = \frac{f(X_{\text{nudged}}) - f(X)}{h}$$
5. Store that slope, and move to the next index.

---

### 📝 BLOCK 2: Pen & Paper Lab (Multidimensional Gradient)
*Open your notebook. Let's solve a 3D system algebraically.*

You have a 3D vector input $X = [x_0, x_1, x_2]$.
The function is:
$$f(X) = x_0^2 + 3x_0 x_1 + x_2^3$$

*   [ ] **Step 1:** Calculate the algebraic partial derivative $\frac{\partial f}{\partial x_0}$ on paper.
*   [ ] **Step 2:** Calculate the algebraic partial derivative $\frac{\partial f}{\partial x_1}$ on paper.
*   [ ] **Step 3:** Calculate the algebraic partial derivative $\frac{\partial f}{\partial x_2}$ on paper.
*   [ ] **Step 4:** Write down your final gradient vector $\nabla f(X)$.
*   [ ] **Step 5:** Evaluate this gradient at the point **$X = [2.0,  4.0,  3.0]$** to get a 3-element numerical vector. Keep these three numbers to test your code!

---

### 💻 BLOCK 3: Coding Lab (The Dynamic $N$-D Gradient Estimator)
*Create a file called `day14_nd_gradient.py`.*

#### **Your Task:**
1.  **Define the Vectorized Function:**
    *   Write a Python function `f(X)` that takes a single 1D NumPy array `X` of size 3 and returns the value of $x_0^2 + 3x_0 x_1 + x_2^3$.
    *   *Example:* Inside your function, $x_0$ is `X[0]`, $x_1$ is `X[1]`, and $x_2$ is `X[2]`.
2.  **Implement the $N$-D Estimator:**
    *   Write a function `numerical_gradient_nd(f, X, h=1e-5)` that:
        *   Accepts the function `f`, the input vector `X` (a NumPy array), and step size `h`.
        *   Initializes `grad = np.zeros_like(X)` (creates a vector of zeros matching the shape of `X`).
        *   Loops through every index `i` in `range(len(X))`.
        *   Inside the loop, creates a copy of `X` (use `X_nudge = X.copy()`).
        *   Adds `h` to `X_nudge[i]`.
        *   Calculates the partial derivative at index `i` and stores it in `grad[i]`.
        *   Returns the final `grad` array.
3.  **Run the Verification:**
    *   Call `numerical_gradient_nd(f, np.array([2.0, 4.0, 3.0]))`.
    *   Print the result. It must match your 3-element vector from Block 2!

---