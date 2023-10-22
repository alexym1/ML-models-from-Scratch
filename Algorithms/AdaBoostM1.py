### 10.1 Algorithm - Adaboost M1 Classifier

### Hastie, T. et al. (2009), The Elements of Statistical Learning, 
### Second Edition, DOI 10.1007/b94608_10, Springer Science+Business Media, LLC.

### Special thanks to https://github.com/AlvaroCorrales/AdaBoost


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def compute_error(y, y_pred, w_i):
    '''
    Calculate the error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w_i: individual weights for each observation

    
    Note that all arrays should be the same length
    '''
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)


def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    return np.log((1 - error) / error)


def update_weights(w_i, alpha, y, y_pred):
    ''' 
    Update individual weights w_i after a boosting iteration. Arguments:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate y_pred
    '''  
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))


class AdaBoostClassifierFS:
    
    def __init__(self, M=100):
        self.M = M
        self.alphas = []
        self.G_M = []
        self.training_errors = []
        self.prediction_errors = []

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
          
          from DataScratch import dataset_for_MLscratch
          from AdaBoostM1 import AdaBoostClassifierFS
          from sklearn.model_selection import train_test_split
          
          df = dataset_for_MLscratch()
          X = df.drop(["Species"], axis = 1)
          y = df["Species"].to_numpy()
          
          X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
          classifier = AdaBoostClassifierFS(M = 100)
          classifier.fit(X_train, y_train)
          y_pred = classifier.predict(X_test)
          
        '''
        
        self.alphas = [] 
        self.training_errors = []

        # Iterate over M weak classifiers
        for m in range(self.M):
            
            if m == 0:
              # Initialize the observation weights wi = 1/N, i = 1,2,...,N.
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
              # Update wi
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            
            # a) Fit a classifier Gm(x) to the training data using weights wi
            # Weak learners in AdaBoost are typically decision trees with a single level (=stump trees)
            G_m = DecisionTreeClassifier(max_depth = 1)
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            self.G_M.append(G_m)

            # b) Compute error
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # c) Compute alpha
            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)


    def predict(self, X):
        '''
        Predict using fitted model. Arguments:
        X: independent variables
        '''

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # Estimate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
      
    def error_rates(self, X, y):
        '''
        Get the error rates of each weak classifier. Arguments:
        X: independent variables
        y: target variables associated to X
        '''
        
        self.prediction_errors = []
        
        # Predict class label for each weak classifier
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X)          
            error_m = compute_error(y = y, y_pred = y_pred_m, w_i = np.ones(len(y)))
            self.prediction_errors.append(error_m)
