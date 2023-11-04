### Example of dataset

import pandas as pd
from sklearn.datasets import load_iris

def dataset_for_MLscratch():
  
  iris = load_iris()
  X = pd.DataFrame(iris.data, columns = iris.feature_names)
  y = pd.DataFrame(iris.target, columns = ["Species"])
  new_iris = pd.concat([X,y], axis = 1)
  
  return new_iris
