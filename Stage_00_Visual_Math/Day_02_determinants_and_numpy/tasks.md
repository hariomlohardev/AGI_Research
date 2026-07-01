
---

# 🧠 DAY 2: The 3D World & The Power of NumPy

### 🛑 RULES FOR TODAY:
1. **The Visual Rule:** When watching the videos today, think about how AI handles 3D space. 
2. **The Speed Rule:** Today, you finally get to use `NumPy`. We are going to test how much faster it is than your code from yesterday.

---

### 🎥 BLOCK 1: Theory (Watch & Visualize)
Go back to the **3Blue1Brown - Essence of Linear Algebra** playlist.
*   [ ] **Watch Video 5:** Three-dimensional linear transformations. *(Crucial: This is how AI processes RGB images—Red, Green, Blue arrays!)*
*   [ ] **Watch Video 6:** The Determinant. 
*   **AGI Intuition for Determinants:** In 12th grade, you just cross-multiplied numbers. In AI, the determinant tells you if your Neural Network is destroying data. If the determinant is `0`, it means your matrix squashed the whole 2D world into a single 1D line. The AI just lost a dimension of information!

---

### 📝 BLOCK 2: Pen & Paper Lab
*Open your notebook. Just one math concept today before we code.*

**Question 1: The Determinant**
*   Given Matrix `A = [[3, 2], [1, 4]]`
*   Calculate the determinant by hand. *(Formula: ad - bc)*
*   **Think:** Is the area of the data stretching or shrinking?

**Question 2: The "Data Destruction" Matrix**
*   Given Matrix `B = [[4, 2], [2, 1]]`
*   Calculate the determinant. 
*   What happens to the data? (Hint: It equals 0. The space collapses).

---

### 💻 BLOCK 3: Python Coding Lab (in `day2_numpy.py`)
*Today we upgrade. Open your terminal in VS Code and type: `pip install numpy`*
*Create a new file called `day2_numpy.py`.*

*   [ ] **Task 1: Import NumPy** 
    Write `import numpy as np` at the top of your file.
*   [ ] **Task 2: Arrays**
    Create `v1 = np.array([4, -2])` and `v2 = np.array([1, 5])`.
*   [ ] **Task 3: Do Day 1 in 4 Lines of Code**
    *   Print Vector Addition: `v1 + v2`
    *   Print Scalar Mult: `3 * v1`
    *   Print Dot Product: `np.dot(v1, v2)`
    *   Print Matrix * Matrix: `np.matmul(m1, m2)` or just `m1 @ m2`.

*Mind-blowing, right? What took you 30 lines of code yesterday takes 1 line today.*

---

### 🚀 BLOCK 4: The Boss Level (The Benchmark Test)
Since you are an advanced Python developer, I am giving you a special challenge today. 
We are going to race your Day 1 code against NumPy.

*   [ ] At the top of `day2_numpy.py`, write `import time`.
*   [ ] Copy your `matrix_matrix_mult` function from `day1_math.py` and paste it into `day2_numpy.py`.
*   [ ] Create two MASSIVE matrices randomly using NumPy:
    ```python
    # Creates two 200x200 matrices filled with random numbers
    big_matrix_1 = np.random.rand(200, 200)
    big_matrix_2 = np.random.rand(200, 200)
    ```
*   [ ] **The Race:**
    1. Use `time.time()` to record the start time.
    2. Multiply the massive matrices using *your* Day 1 function. (Pass them in as `big_matrix_1.tolist()`).
    3. Record the end time and print how many seconds it took.
    4. Now, do the same thing using `np.matmul(big_matrix_1, big_matrix_2)`.
    5. Print how many seconds NumPy took.

---
