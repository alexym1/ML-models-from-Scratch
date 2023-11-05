### 15.1 Algorithm - RandomForest

### Hastie, T. et al. (2009), The Elements of Statistical Learning, 
### Second Edition, DOI 10.1007/b94608_10, Springer Science+Business Media, LLC.


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier:
    
    def __init__(self, ntree=100):
        self.ntree = ntree
        self.R_F = []
        self.Xsubset = {}
        self.Ysubset = {}

    def fit(self, X, y):
        '''
        Fit model. Arguments:
        X: independent variables
        y: Dichotomus variable (int)
        ntree: number of decision trees. Default is 100
        
        Default:
        Decision tree: CART
        Loss function: Gini impurity
        
        Example:
          
          from components.datasets.DataScratch import dataset_for_MLscratch
          from components.algorithms.randomforest import RandomForestClassifier
          from sklearn.model_selection import train_test_split
          
          df = dataset_for_MLscratch()
          X = df.drop(["Species"], axis = 1)
          y = df["Species"].to_numpy()
          
          X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
          classifier = RandomForestClassifier(ntree = 100)
          classifier.fit(X_train, y_train)
          y_pred = classifier.predict(X_test)
          classifier.OOB(y_test,y_pred)
          
        '''
        
        target = pd.DataFrame(y, columns = ["target"])
        df = pd.concat([X.reset_index(drop=True),target],axis=1)

        # Generate B decision trees
        for tree in range(self.ntree):
          
          # a) Draw a bootstrap sample Z* of size N from the training data
          z = np.random.randint(low = 1, high = len(df))
          boostrap_sample = df.sample(n=z, replace=False)
          
          # b) Grow a RandomForest tree Tb to the boostrapped data
          
          # i) Select m variables at random from the P variables
          p = np.random.randint(low = 1, high = len(X.columns))
          new_X = boostrap_sample.drop(["target"],axis=1).sample(n=p, axis='columns', replace=False)
          new_y = boostrap_sample["target"].to_numpy()
          
          # ii) Pick the best variable/split-point among the m
          # & iii) Split the node into two daughter nodes
          # Repeat the ii) & ii) steps recursively
          classifier = DecisionTreeClassifier()
          classifier.fit(new_X, new_y)
          
          self.R_F.append(classifier)
          self.Xsubset[tree] = new_X
          self.Ysubset[tree] = new_y
          
          
    def predict(self, X):
        '''
        Predict using fitted model. Arguments:
        X: independent variables
        '''

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.ntree)) 

        # Predict class label for each weak classifier
        for m in range(self.ntree):
          model = self.R_F[m]
          y_pred_m = model.predict(X[model.feature_names_in_.tolist()])
          weak_preds.loc[:,m] = y_pred_m


        # Output the ensemble of trees
        # Classification -> Majority vote 
        y_preds = []
        for vote in range(len(X)):
          preds, counts = np.unique(weak_preds.iloc[vote,:], return_counts=True)
          y_pred = preds[np.argmax(counts)]
          y_preds.append(y_pred)
        
        return y_preds

 
    def OOB(self, y_true, y_pred):
        '''
        Get the out-of-bag error.
        y_true: y_train or y_test
        y_pred: y predicton from the classifier
        '''
        error = int()
        for b in range(len(y_true)):
          if not y_true[b] == y_pred[b]:
            error += 1
            
        OOB_err = error / len(y_true)
        
        return OOB_err
