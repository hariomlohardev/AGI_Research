
---

# 🧠 DAY 19: The Autograd Engine – The Value Class & Graph Construction

### 🛑 RULES FOR TODAY:
1. **The Class Builder:** You must implement the class structure, `__add__`, and `__mul__` yourself.
2. **The Manual Graph Walk:** You will build a simple mathematical expression using your class and manually execute the backward chain rule steps on each node.

---

### 🎥 BLOCK 1: Theory (Building Micrograd)
We are going to use the absolute gold-standard video for this project. 
*   [x] **Watch Video:** [The spelled-out intro to neural networks and backpropagation: building micrograd (Min 15 to 45)](https://www.youtube.com/watch?v=VMj-3S1tku0)
    *(Watch how Karpathy sets up the `Value` class, how the parents are stored in a set, and how we manually propagate gradients through addition and multiplication nodes).*

---

### 📝 BLOCK 2: Pen & Paper Lab (The Graph Walk)
Let's trace a simple graph on paper:
*   $a = 2.0$
*   $b = -3.0$
*   $c = 10.0$
*   $e = a \times b$
*   $d = e + c$ (So $d = (a \times b) + c = 4.0$).

Assuming the final loss is $d$ itself, let's backpropagate manually:
1.  **Output Gradient:** $\frac{\partial d}{\partial d} = \mathbf{1.0}$. (This is your starting gradient, `d.grad = 1.0`).
2.  **Addition Node ($d = e + c$):** 
    *   What are the local derivatives $\frac{\partial d}{\partial e}$ and $\frac{\partial d}{\partial c}$? (Hint: The derivative of addition is always $1$).
    *   Use the chain rule to calculate `e.grad` and `c.grad`.
3.  **Multiplication Node ($e = a \times b$):**
    *   What are the local derivatives $\frac{\partial e}{\partial a}$ and $\frac{\partial e}{\partial b}$? (Hint: The derivative of $a \times b$ with respect to $a$ is $b$. The derivative with respect to $b$ is $a$).
    *   Use the chain rule ($\text{parent.grad} += \text{child.grad} \times \text{local\_derivative}$) to calculate `a.grad` and `b.grad`.
4.  **Keep these final numerical gradients** to verify your Python code!

---

### 💻 BLOCK 3: Coding Lab (The `Value` Class Structure)
*Create a file called `day19_autograd.py`.*

#### **Your Task:**
1.  **Define the `Value` Class:**
    *   In `__init__(self, data, _children=(), _op='')`:
        *   Store `self.data` as a float.
        *   Initialize `self.grad = 0.0`.
        *   Store `self._prev = set(_children)`.
        *   Store `self._op = _op`.
        *   Initialize `self._backward = lambda: None` (A placeholder function).
    *   Implement a `__repr__` method so it prints nicely, e.g., `Value(data=2.0, grad=0.0)`.
2.  **Implement Addition (`__add__(self, other)`):**
    *   Ensure `other` is converted to a `Value` if it is a raw number (for now, assume both are `Value` objects).
    *   Create a new node `out = Value(self.data + other.data, (self, other), '+')`.
    *   Define the inner backward function for this specific addition operation:
        ```python
        def _backward():
            self.grad += out.grad  # Local derivative is 1, so we just pass out.grad
            other.grad += out.grad
        out._backward = _backward
        ```
    *   Return `out`.
3.  **Implement Multiplication (`__mul__(self, other)`):**
    *   Create a new node `out = Value(self.data * other.data, (self, other), '*')`.
    *   Define the inner backward function:
        ```python
        def _backward():
            # Use the local product rule derivatives:
            # self.grad gets updated using other.data
            # other.grad gets updated using self.data
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        ```
    *   Return `out`.
4.  **Run the Manual Graph Verification:**
    *   Recreate the Block 2 math in code:
        ```python
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)
        
        # Forward pass (this automatically builds the graph)
        e = a * b
        d = e + c
        
        # Manual backward pass (triggering the step backward one-by-one)
        d.grad = 1.0 # Initialize the final gradient
        d._backward() # Propagates gradient to e and c
        e._backward() # Propagates gradient to a and b
        ```
    *   Print the gradients: `a.grad`, `b.grad`, `c.grad`, `e.grad`.
    *   **The Proof:** Confirm that your Python terminal prints the exact same gradients you calculated in your notebook!

---
