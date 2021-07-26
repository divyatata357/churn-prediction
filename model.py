# Importing the libraries
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('spam.csv')

dataset['class'] = dataset['class'].map({'G1': 0, 'AC': 1})

X = dataset.iloc[:,:-1]
y = dataset['class']
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model1.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))