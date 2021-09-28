import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/diabetes.csv')
print(df)

# x = df.iloc[:,0].values.reshape(-1,1)
# y = df.iloc[:,1].values
# print(x)
# print(y)

x = df.iloc[:,0].values.reshape(-1,1)
opp = x.shape
print(opp)
print(x)
y = df.iloc[:,1].values
print(y)


#CORELATION

s = df.corr()
print(s)

#training/testing sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.2,random_state=0)
d = x_test
print(d)




#feature scalling


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print(x_test)



#logistic regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
a = classifier.fit(x_train,y_train)
print(a)

pred_y = classifier.predict(x_test)
print(pred_y)


print(y_test)


#classification matrics


from sklearn.metrics import accuracy_score
w = accuracy_score(y_test,pred_y)
print(w)


from sklearn.metrics import confusion_matrix
q = confusion_matrix(y_test,pred_y)
print(q)




#error evaluation



#1
#MAE-mean absolute error
#2
#MSE-mean squared error

m = pred_y - y_test
print(m)

# n = np.abs(pred_y - test_y)
# print(n)

n = np.abs(pred_y - y_test).mean()
print(n)



from sklearn import metrics
c = metrics.mean_absolute_error(y_test,pred_y)
print(c)

s = metrics.mean_squared_error(y_test,pred_y)
print(s)


#DATA VISUALISATION


# plt.scatter(x,y)
plt.plot(x,y,color = 'g',marker = "*",markersize = 10)
rav = plt.show()
print(rav)

