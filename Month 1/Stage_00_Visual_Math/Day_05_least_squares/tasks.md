
---

# 🧠 DAY 5: Projections & The Normal Equation (Your First ML Predictor)

### 🛑 RULES FOR TODAY:
1. **The Math Rule:** Understand the most important formula in classical ML: 
   $$\theta = (X^T X)^{-1} X^T y$$
   *(This is the mathematical shortcut to draw a perfect line through messy data).*
2. **The Visual Rule:** You must plot your data and your line using `matplotlib` to visually see your math succeed.

---

### 🎥 BLOCK 1: Theory (MIT Lectures)
These three lectures are the foundation of all predictive AI. Watch them carefully:
*   [x] **Lecture 14:** Orthogonal Vectors and Subspaces. *(Why orthogonal vectors have zero similarity).*
*   [x] **Lecture 15:** Projections onto Subspaces. *(How we project high-dimensional data onto a lower-dimensional screen).*
*   [x] **Lecture 16:** Projection Matrices and Least Squares. *(The math of finding the best-fit line).*

---

### 📝 BLOCK 2: Pen & Paper Lab (The Projection Shortcut)
*Let's do a simple projection on paper.*

Imagine you have a vector $b = [1, 2, 2]$ (a point in 3D space), and you want to project it onto a 1D line represented by vector $a = [2, 3, 4]$.

1.  Calculate the projection scalar $x$ (the recipe for how far to go along line $a$):
    $$x = \frac{a^T b}{a^T a}$$
    *(Hint: $a^T b$ is just the dot product of $a$ and $b$. $a^T a$ is the dot product of $a$ with itself).*
2.  Calculate the projection vector $p$ (the actual "shadow" on the line):
    $$p = x \cdot a$$
3.  **Think:** Why is the vector $e = b - p$ (the error) orthogonal to the line $a$? (Hint: Draw it).

---

### 💻 BLOCK 3: Coding Lab (The Normal Equation from Scratch)
*Create a file called `day5_least_squares.py`.*

Imagine we have 4 data points representing **[House Size (in 1000 sq ft), Price (in crores)]**:
*   Size = 1, Price = 2
*   Size = 2, Price = 3
*   Size = 3, Price = 5
*   Size = 4, Price = 6

There is no straight line that can pass through `(1,2), (2,3), (3,5), and (4,6)` perfectly. We will use the **Normal Equation** to find the best possible line ($y = mx + c$).

1.  **Define your data matrices in NumPy:**
    We need a matrix $X$ (with a column of 1s for the intercept $c$, and a column for sizes) and vector $y$ (prices):
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # House sizes (X) and intercept column
    X = np.array([[1, 1],
                  [1, 2],
                  [1, 3],
                  [1, 4]]) # First column is 1s, second column is house size

    y = np.array([2, 3, 5, 6]) # Prices
    ```
2.  **Code the Normal Equation:**
    Write the NumPy code to solve for the weights $\theta = [c, m]$ (where $c$ is the intercept and $m$ is the slope):
    $$\theta = (X^T X)^{-1} X^T y$$
    *   *Hint:* Use `X.T` for transpose, `np.linalg.inv()` for inverse, and `@` for matrix multiplication.
3.  **Print the weights:**
    Print your calculated slope ($m$) and intercept ($c$).

---

### 📊 BLOCK 4: Plot your first Machine Learning Model
Now, let's plot the actual data points and the line you just calculated.

1.  **Extract your parameters:**
    `c, m = theta[0], theta[1]`
2.  **Generate the line:**
    `line_y = m * X[:, 1] + c`
3.  **Plot it using Matplotlib:**
    ```python
    plt.scatter(X[:, 1], y, color='red', label='Real Data') # Plot data points
    plt.plot(X[:, 1], line_y, color='blue', label='Best Fit Line') # Plot your ML line
    plt.xlabel('House Size')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    ```

---