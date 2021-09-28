import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/tissue.csv")
print(df)


x = df.iloc[:,0:2].values
y = df.iloc[:,-1].values
print(x)
print(y)


#for converting in number format

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#machine learning algorithm

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3,metric = 'euclidean')
san = classifier.fit(x,y)
print(san)

sa = classifier.predict([[3,7]])
print(sa)
