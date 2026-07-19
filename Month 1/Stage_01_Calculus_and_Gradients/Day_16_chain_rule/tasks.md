
---

# 🧠 DAY 16: The Multivariable Chain Rule & Multi-Path Backprop

### 🛑 RULES FOR TODAY:
1. **The Dual Verification:** You must calculate the derivative using two different methods on paper: first using the multivariable chain rule, and second using direct substitution. They must match.
2. **The Numerical Chain Checker:** You will code a numerical simulator that calculates each individual slope along the branches and combines them using the chain rule formula.

---

### 🎥 BLOCK 1: Theory (Watch & Visualize)
Watch this short, highly visual video to understand how derivatives flow along branching pathways:
*   [ ] **Watch Video (Search on YT):** `Multivariable chain rule | Multivariable calculus | Khan Academy`  
    *(Working Link: [Khan Academy - Multivariable Chain Rule](https://www.youtube.com/watch?v=A8v_K6vS8U4))*
    *(Pay close attention to the "tree diagram" of variables—this is exactly what we call a computational graph in AI!).*

---

### 📝 BLOCK 2: Pen & Paper Lab (Calculating Multi-path Derivatives)
*Open your notebook. Let's calculate the derivative of a branching network.*

We have a system where the output $z$ depends on $u$ and $v$:
$$z(u, v) = u^2 + v^2$$

And both $u$ and $v$ depend on a single starting weight $x$:
$$u(x) = 3x$$
$$v(x) = x^2$$

1.  **Method A: The Multivariable Chain Rule**
    *   Calculate the partial derivatives: $\frac{\partial z}{\partial u}$, $\frac{\partial z}{\partial v}$, $\frac{du}{dx}$, and $\frac{dv}{dx}$.
    *   Apply the Multivariable Chain Rule formula:
        $$\frac{dz}{dx} = \frac{\partial z}{\partial u} \frac{du}{dx} + \frac{\partial z}{\partial v} \frac{dv}{dx}$$
    *   Substitute $u(x) = 3x$ and $v(x) = x^2$ back into your final expression so it is entirely in terms of $x$.
    *   Evaluate $\frac{dz}{dx}$ at the point **$x = 2.0$**. Keep this integer!
2.  **Method B: Direct Substitution (The Proof)**
    *   Substitute $u(x)$ and $v(x)$ directly into the equation for $z(u, v)$ to get a single function $z(x)$ containing only the variable $x$.
    *   Differentiate $z(x)$ using standard single-variable calculus.
    *   Evaluate your derivative at **$x = 2.0$**.
    *   **The Proof:** Confirm that Method A and Method B yield the exact same numerical answer.

---

### 💻 BLOCK 3: Coding Lab (The Multi-path Gradient Checker)
*Create a file called `day16_chain_rule.py`.*

#### **Your Task:**
1.  **Define your math functions:**
    *   Write `u(x)` returning $3x$.
    *   Write `v(x)` returning $x^2$.
    *   Write `z(u_val, v_val)` returning $u\_val^2 + v\_val^2$.
2.  **Implement `numerical_chain_rule(x, h=1e-5)`:**
    *   Calculate the individual numerical slopes using the finite difference method:
        *   `dz_du`: Nudge $u\_val$ in function `z` while keeping $v\_val$ constant. (Hint: Pass `u(x) + h` and `v(x)`).
        *   `dz_dv`: Nudge $v\_val$ in function `z` while keeping $u\_val$ constant. (Hint: Pass `u(x)` and `v(x) + h`).
        *   `du_dx`: Nudge $x$ in function `u`.
        *   `dv_dx`: Nudge $x$ in function `v`.
    *   Combine these 4 individual slopes using the multivariable chain rule formula:
        `dz_dx = (dz_du * du_dx) + (dz_dv * dv_dx)`
    *   Return `dz_dx`.
3.  **Run the Verification:**
    *   Call `numerical_chain_rule(2.0)` and print the result.
    *   Confirm it matches your hand-written calculations from Block 2!

---