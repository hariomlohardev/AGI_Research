
---

# 🧠 DAY 20: Building the Multilayer Perceptron (MLP) from Scratch

### 🛑 RULES FOR TODAY:
1. **The Object-Oriented Rule:** You must implement the class structures for `Neuron`, `Layer`, and `MLP` yourself using your `Value` class. 
2. **The Parameter Collector:** Your classes must implement a `.parameters()` method that recursively gathers every weight and bias `Value` object in the network. This is how optimizers "see" what they need to update.

---

### 🎥 BLOCK 1: Theory (Building Neurons and Layers)
Go back to Andrej Karpathy's Micrograd video on YouTube:
*   [x] **Watch Segment:** [Building a Multi-Layer Perceptron (MLP) (Min 45 to 1:15)](https://www.youtube.com/watch?v=VMj-3S1tku0)
    *(Watch how Karpathy designs the three classes, how he uses `random.uniform` to initialize weights as `Value` objects, and how he flattens lists to collect parameters).*

---

### 📝 BLOCK 2: Architectural Specifications (No Code, Just Blueprint)

You must write your neural classes to meet these exact requirements. Import Python's built-in `random` module to initialize weights and biases.

#### **1. The `Neuron(nin)` Class**
*   **Initialization:** Takes an integer `nin` (number of inputs). 
    *   Create a list of weights `self.w` containing `nin` random `Value` objects (use `random.uniform(-1, 1)`).
    *   Create a single bias `self.b` initialized as a random `Value` object.
*   **The Activation Function:** Implement `__call__(self, x)` which takes an input list `x`. It must calculate the weighted sum plus the bias:
    $$\text{act} = \sum(x_i \cdot w_i) + b$$
    And return `act.sigmoid()` (your custom sigmoid function from yesterday).
*   **Parameters:** Implement `parameters(self)` which returns a list containing all its weights and its bias: `self.w + [self.b]`.

#### **2. The `Layer(nin, nout)` Class**
*   **Initialization:** Takes `nin` (inputs to each neuron) and `nout` (number of neurons in this layer).
    *   Create a list `self.neurons` containing `nout` `Neuron` objects.
*   **Feedforward:** Implement `__call__(self, x)` which passes the input list `x` through each neuron in the layer.
    *   *Requirement:* If `nout == 1`, return just the single `Value` output instead of a list of size 1. If `nout > 1`, return a list of output `Value` objects.
*   **Parameters:** Implement `parameters(self)` which returns a flattened list of all parameters across all its neurons.

#### **3. The `MLP(nin, nouts)` Class**
*   **Initialization:** Takes `nin` (inputs to the first layer) and a list of integers `nouts` representing the sizes of all successive layers (e.g., `nouts = [4, 4, 1]` means a 3-layer network).
    *   Create a list of `self.layers` where the output size of one layer becomes the input size of the next.
*   **Feedforward:** Implement `__call__(self, x)` which takes an input vector `x` and passes it sequentially through all layers in `self.layers`.
*   **Parameters:** Implement `parameters(self)` which returns a flattened list of all parameters across all layers in the MLP.

---

### 💻 BLOCK 3: Coding Lab (The First Backprop through a Network)
*Create a file called `day20_mlp.py`. Copy your `Value` class from yesterday to the top of the file.*

#### **Your Task:**
1.  Implement the `Neuron`, `Layer`, and `MLP` classes based on the blueprints above.
2.  **Test the Forward Pass:**
    *   Instantiate an MLP: `n = MLP(3, [4, 4, 1])` (takes 3 inputs, has two hidden layers of 4 neurons, and 1 output neuron).
    *   Define a dummy 3D input: `xs = [2.0, 3.0, -1.0]`.
    *   Pass the input through: `ypred = n(xs)`.
    *   Print `ypred`. It should be a single `Value` object representing your network's prediction.
3.  **Test the Backward Pass:**
    *   Call `ypred.backward()`.
    *   Check how many parameters are in your network: `print(len(n.parameters()))`. (For a `3 -> [4, 4, 1]` network, there should be exactly **41 parameters**: weights + biases. Verify this count!).
    *   **The Ultimate Test:** Print the gradient of the first weight of the first neuron in your first layer: `print(n.layers[0].neurons[0].w[0].grad)`.
    *   *Proof:* Verify that this gradient is **not** `0.0`. If it is a non-zero decimal, it means your backpropagation successfully flowed through all 3 layers and updated the parameters at the very beginning!

---
