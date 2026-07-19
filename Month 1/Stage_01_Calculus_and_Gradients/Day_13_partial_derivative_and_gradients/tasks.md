
---

# 🧠 DAY 13: Partial Derivatives & The Numerical Gradient Estimator

### 🛑 RULES FOR TODAY:
1. **The Partial Rule:** When calculating a partial derivative with respect to $x$, treat all other variables (like $y$) as constant frozen numbers.
2. **The Numerical Approximation Rule:** You will code the **Finite Difference Method** in Python to approximate slopes without using algebraic calculus, and prove it matches your paper math.

---

### 📖 BLOCK 1: The Concept of the Gradient

If you have a function of two variables:
$$f(x, y) = x^2 + 3xy + y^2$$

1.  **Partial Derivative with respect to $x$ ($\frac{\partial f}{\partial x}$):** 
    Treat $y$ as a constant number (like `5`). Differentiate normally with respect to $x$:
    $$\frac{\partial f}{\partial x} = 2x + 3y + 0 = 2x + 3y$$
2.  **Partial Derivative with respect to $y$ ($\frac{\partial f}{\partial y}$):** 
    Treat $x$ as a constant number. Differentiate normally with respect to $y$:
    $$\frac{\partial f}{\partial y} = 0 + 3x + 2y = 3x + 2y$$
3.  **The Gradient Vector ($\nabla f$):**
    The gradient is simply a vector containing all the partial derivatives:
    $$\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x + 3y \\ 3x + 2y \end{bmatrix}$$
    *   **AGI Intuition:** The gradient vector points in the direction of the **steepest uphill slope**. If an AI wants to minimize its error, it calculates the gradient and walks in the **opposite direction** ($-\nabla f$). This is the entire foundation of **Gradient Descent**!

---

### 📝 BLOCK 2: Pen & Paper Lab (Calculating the Slope Vector)

You have the 3D surface function:
$$f(x, y) = x^2 y + 2xy^2 + y^3$$

1.  Calculate the algebraic partial derivative $\frac{\partial f}{\partial x}$ on paper.
2.  Calculate the algebraic partial derivative $\frac{\partial f}{\partial y}$ on paper.
3.  Write down the gradient vector $\nabla f(x, y)$.
4.  Evaluate the gradient vector at the point **$(x, y) = (2, 3)$**. 
    *   *Your output should be a numerical vector $[a, b]$. Keep this to verify your code.*

---

### 💻 BLOCK 3: Coding Lab (The Numerical Gradient Estimator)
*Create a file called `day13_gradients.py`.*

How do computers calculate gradients when they don't know algebraic calculus? They use the **Finite Difference Method**:
$$\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x, y)}{h}$$
Where $h$ is an extremely small step size (like $10^{-5}$ or `1e-5`).

#### **Your Task:**
1.  **Define the Mathematical Function:**
    Write a Python function `f(x, y)` that returns the value of $x^2 y + 2xy^2 + y^3$.
2.  **Implement the Estimator:**
    Write a function `numerical_gradient(f, x, y, h=1e-5)` that:
    *   Approximates $\frac{\partial f}{\partial x}$ using the finite difference formula (nudge $x$ by $h$ while keeping $y$ unchanged).
    *   Approximates $\frac{\partial f}{\partial y}$ using the finite difference formula (nudge $y$ by $h$ while keeping $x$ unchanged).
    *   Returns the approximated gradient vector as a list or NumPy array `[df_dx, df_dy]`.
3.  **Run the Verification:**
    Call your function at the point `(2.0, 3.0)` with `h=1e-5`. Print the result.

---