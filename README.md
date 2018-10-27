```python
%load_ext autoreload
%autoreload 2
```

## Load data

```python
import pandas
from decision_tree import *

df = pandas.read_csv('01_homework_dataset.csv', delimiter=',')

X = df.iloc[:,:3].values
y = df.iloc[:,3].values

d = DecisionTree(2, X, y, df.columns.values)
d.auto_split()
```

## Problem 1:
Each line represents one node.
Format of each line:
0. condition
from parent node was matched?
1. splitting criterion for the next level
2. class
distribution
3. gini index for current node

The gini index was calculated as
follows:
$i_G(t) = 1 - \sum_{c_i \in C} \pi_{c_i}^2$

The code for the
implemented decision tree has been attached

```python
d.draw()
```

## Problem 2
Predict: $x_a=(4.1, -0.1,2.2)^T$ and $x_b=(6.1, 0.4,1.3)^T$
Following the tree we see that $x_{a,1} \leq 4.1$ and our distribution in the
respective node is pure since it only contains data with the label 1.

`True |
distr: {1: 1.0}             gini: 0.0`

$\hat{y}_a = 1$ and since the node is
pure: $p(c=\hat{y}_a | x_a, T) = 1$


Path for $x_b$:

$x_{b,1} \leq 4.1 ? =>
False$

$x_{b,1} \leq 6.9 ? => True$

`True | distr: {0: 0.33, 2: 0.67}   gini:
0.44`

$\hat{y}_b = 2$ and $p(c=\hat{y}_b | x_b, T) \approx 0.67$

```python
d.predict([4.1,-0.1,2.2])
```

```python
d.predict([6.1,0.4,1.3])
```
