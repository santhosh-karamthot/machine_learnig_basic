import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/AdultIncome.csv")
print(df)



#for checking the null values

tchknull = df.isnull().sum()

print(tchknull)


#get dummies methods(for data frame)

gdm = pd.get_dummies(df,drop_first=True)
print(gdm)


#for input output(spliting x and y)
x = df.iloc[:, :1].values
y = df.iloc[:, 1].values
print(x)
print(y)

#FEATURE SCALLING(IF THE ARE NOT IN STANDARAED FORM THEN WE HAVE TO DO THE FEATURE SCALLING)

#splitting and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

print(x_test)


#decision tree with classifier

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(x_train,y_train)
print(dtc)


pred_y = dtc.predict(x_test)
print(pred_y)


#for to check the accuracy and confusion matrics

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,pred_y)
print(cm)

sa = accuracy_score(y_test,pred_y)
print(sa)


#to get more  accuracy we have go for this method


#ENSEMBLE TECHNIQES:

#1.BAGGING:RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)
rs = rfc.fit(x_train,y_train)
print(rs)

y_predrfc = rfc.predict(x_test)

cms = confusion_matrix(y_test,y_predrfc)
print(cms)
saa = accuracy_score(y_test,y_predrfc)
print(saa)











