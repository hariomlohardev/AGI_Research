
---

# 🧠 DAY 12: Stage 1 Graduation Project – Custom PCA from Scratch

### 🛑 RULES FOR TODAY:
1. **The OOP Rule:** You must write your PCA algorithm as a reusable Object-Oriented Python class (`class PCA`).
2. **The No-Library Rule:** You are only allowed to use `sklearn` to load the raw Iris dataset. You must not use any of `sklearn`'s math or decomposition functions. All math must be done with your custom SVD logic in NumPy.

---

### 💻 THE GRADUATION TASK: Building `day12_pca.py`

Create a file called `day12_pca.py` and implement the following structure:

#### **Step 1: Load the Raw Data**
Use this exact code at the top of your file to get the raw 4D data of the flowers:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the classic Iris Dataset
iris = load_iris()
X = iris.data       # Shape: (150, 4) -> 4D data
y = iris.target     # Shape: (150,) -> 0, 1, or 2 representing the 3 flower species
```

#### **Step 2: Implement the `PCA` Class**
Write your custom class. It must have this exact structure:

```python
class PCA:
    def __init__(self):
        self.mean = None
        self.components = None

    def fit(self, X):
        # 1. Calculate the mean of each column (feature) and save it as self.mean
        # 2. Subtract the mean from X to center the data (X_centered)
        # 3. Run SVD on X_centered using np.linalg.svd
        # 4. Extract and store the right singular vectors (V) as self.components
        # Hint: np.linalg.svd returns Vt. You need to transpose it to get V!
        pass

    def transform(self, X, k):
        # 1. Center the input data X using the stored self.mean
        # 2. Slice self.components to keep only the first k columns
        # 3. Project the centered data onto these k columns using matrix multiplication (@)
        # 4. Return the projected (compressed) data matrix
        pass
```

#### **Step 3: Run the Compression**
*   Instantiate your class: `pca = PCA()`
*   Fit the data: `pca.fit(X)`
*   Compress the 4D data down to 2D: `X_projected = pca.transform(X, k=2)`
*   *Verification Print:* Print `X_projected.shape`. It must be exactly `(150, 2)`.

#### **Step 4: Visualize the Compressed Clusters**
Plot your newly compressed 2D data points using Matplotlib:
```python
# Scatter plot the 2D projected data
# We color the points using 'y' so we can see if our math grouped the flower species correctly!
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Custom PCA from Scratch (4D -> 2D)')
plt.colorbar(label='Species Label')
plt.show()
```

---