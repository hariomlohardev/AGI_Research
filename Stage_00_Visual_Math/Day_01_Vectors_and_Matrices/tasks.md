

# 🧠 DAY 1: The Geometry of AI (Vectors & Matrices)

### 🛑 RULES FOR TODAY:
1. **No Libraries:** Do not use `import numpy` or `import math`. Everything must be written in raw Python using standard `lists` and `loops`.
2. **Visualize:** When doing the math, draw the x and y axes on paper. See the vectors moving. 
3. **No Sleep:** You do not sleep until the Python code matches your paper math.

---

### 🎥 BLOCK 1: Theory (Watch & Visualize)
Go to YouTube and search for: **"Essence of linear algebra" by 3Blue1Brown**. 
Watch these 4 videos carefully. Do not memorize formulas; focus on what the shapes are doing.
*   [ ] **Video 1:** Vectors, what even are they?
*   [ ] **Video 2:** Linear combinations, span, and basis vectors.
*   [ ] **Video 3:** Linear transformations and matrices. *(Crucial: Notice how a matrix is just a "movement" of space).*
*   [ ] **Video 4:** Matrix multiplication as composition. *(Crucial: Notice how multiplying two matrices means doing two movements one after the other).*

---

### 📝 BLOCK 2: Pen & Paper Lab (Do this manually)
*Open your notebook. Calculate these by hand to prove your brain understands the logic before the computer does it.*

**Question 1: Vector Arithmetic**
*   Given `v1 = [4, -2]` and `v2 = [1, 5]`
*   Calculate: `v1 + v2`
*   Calculate: `3 * v1`
*   Calculate: `v1 - v2`

**Question 2: Dot Product**
*   Given `a = [3, 4]` and `b = [2, -1]`
*   Calculate the dot product: `a · b`

**Question 3: Matrix-Vector Multiplication**
*   Given Matrix `M = [[2, 0], [0, 3]]` and Vector `x = [2, 1]`
*   Calculate: `M * x` *(Remember: take the dot product of the matrix rows with the vector)*

**Question 4: Matrix-Matrix Multiplication (Composition)**
*   Given Matrix `A = [[1, 2], [3, 4]]` and Matrix `B = [[2, 0], [1, 2]]`
*   Calculate: `A * B` *(Row by Column)*

---

### 💻 BLOCK 3: Python Coding Lab (in `day1_math.py`)
*Write these 5 functions from scratch using pure Python logic (`for` loops, lists, or list comprehensions). I am giving you the specifications, you write the logic.*

*   [ ] **Task 1:** Write `add_vectors(v1, v2)` -> Returns a new list.
*   [ ] **Task 2:** Write `scale_vector(scalar, v)` -> Returns a new list multiplied by the scalar.
*   [ ] **Task 3:** Write `dot_product(v1, v2)` -> Returns a single number.
*   [ ] **Task 4:** Write `matrix_vector_mult(matrix, vector)` -> Returns a new list (the transformed vector). 
*   [ ] **Task 5 (Boss Level):** Write `matrix_matrix_mult(m1, m2)` -> Returns a new list of lists (the new composed matrix).

**Final Step (The Verification):**
At the bottom of `day1_math.py`, write `print()` statements that pass the exact numbers from **Block 2** into your functions. Run the file in your terminal (`python day1_math.py`). 

If your terminal output perfectly matches your notebook math... **You win Day 1.**
