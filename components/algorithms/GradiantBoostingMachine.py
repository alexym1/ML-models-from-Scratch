### 10.3 Algorithm - Gradiant Tree Boosting Machine

### Hastie, T. et al. (2009), The Elements of Statistical Learning, 
### Second Edition, DOI 10.1007/b94608_10, Springer Science+Business Media, LLC.

### Special thanks to https://randomrealizations.com/posts/gradient-boosting-machine-from-scratch/

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingMachineFS():
    
    def __init__(self, ntree = 100, learning_rate = 0.1, max_depth = 1):
        self.ntree = ntree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.trees = []
        
        # 1. Initialize f0(x)
        self.F0 = y.mean()
        Fm = self.F0 
        
        for _ in range(self.ntree):
          
          # 2.a) Compute pseudo residuals (Table 10.2)
          rm = y - Fm
          
          # 2.b) Fit a regression tree to the targets r im giving terminal regions
          tree = DecisionTreeRegressor(max_depth=self.max_depth)
          tree.fit(X, rm)
          
          # 2.c) Output value for gamma that minimizes the summation
          gamma = tree.predict(X)
          self.trees.append(tree)
          
          # 2.d) Update fm(x)
          Fm += self.learning_rate * gamma
          
    
    # 3) Output    
    def predict(self, X):
        return self.F0 + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
