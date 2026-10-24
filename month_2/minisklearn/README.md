# minisklearn

### What is this BTW

This is an mini version of sklearn( `scikit-learn` ).
Build only using python inbuld libraries and nothings else.

Its an part of my AI / AGI resercher journey chech that on [hariomlohardev.github.io](https://hariomlohardev.github.io/)



## Installation 

This is simple as hell :)

### Step 1: Clone the repo 

```bash
git clone https://github.com/hariomlohardev/mini-sklearn.git
```

### Step 2: Install minisklearn


```bash
cd mini-sklearn

pip install -e . # for windows 
#OR
pip3 install -e . # for mac and linux
```

### Step 3: Import

```python
import minisklearn
from minisklearn import LinearRegression , LogisticRegression
```

### Step 4: Fit and Predict

```python
X_train = ... # training data
y_train = ... # training labels 

X_test = ... # test data
y_test = ... # true labels 

model = LinearRegression() # Model linear regression

# fit
model.fit(X_train , y_test) # fiting the model on data

# predict
y_pred = model.predict(X_test)
```

### Step 5: Accuracy

```python
from minisklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Model Accuracy
accuracy  = accuracy_score(y_true = y_test, y_pred = y_pred)

# Precision
precision  = precision_score(y_true = y_test, y_pred = y_pred)

# Recall
recall_score  = recall_score(y_true = y_test, y_pred = y_pred)

# F1 score 
f1  = f1_score(y_true = y_test, y_pred = y_pred)

# F1 score 
confusion  = confusion_matrix(y_true = y_test, y_pred = y_pred)
```

## Usage
```python
from minisklearn.linear_model import LogisticRegression
from minisklearn.metrics import accuracy_score, f1_score

# toy data: two blobs, labels 0 and 1
X_train = [[0.1], [0.2], [0.3], [2.1], [2.2], [2.3]]
y_train = [0, 0, 0, 1, 1, 1]
X_test = [[0.25], [2.0]]
y_test = [0, 1]

model = LogisticRegression(lr=0.1, epochs=500)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("accuracy:", accuracy_score(y_test, preds))
print("f1:", f1_score(y_test, preds))
```

## Model list 

- LinearRegression
- LogisticRegression 
- DecisionTreeClassifier
- RandomForestClassifier
- KMeans
- LinearSVMGD





## sklearn comparison

Titanic split (`test_ratio=0.2, seed=0`), both sides scored with
`minisklearn.metrics`. Classifiers report accuracy / precision / recall /
F1; K-Means reports inertia (no labels, so no accuracy).

| model | mine | sklearn | gap | verdict |
|---|---|---|---|---|
| LogisticRegression | 0.684 / 0.714 / 0.294 / 0.417 | 0.847 / 0.806 / 0.794 / 0.800 | 0.164 | investigate — tiny-lr logistic underfits unscaled features |
| DecisionTree | 0.706 / 0.682 / 0.441 / 0.536 | 0.780 / 0.716 / 0.706 / 0.711 | 0.073 | investigate — exact ID3 splits fragment continuous features |
| RandomForest | 0.695 / 0.652 / 0.441 / 0.526 | 0.831 / 0.828 / 0.706 / 0.762 | 0.136 | investigate — same ID3 weakness, bagged |
| KMeans | inertia 742754.6 | inertia 742754.6 | 0.0 | match — Lloyd loop is correct |

Pattern: my classifiers trail sklearn on recall (0.29–0.44 vs 0.71–0.79) —
they under-predict "survived". KMeans matches exactly.