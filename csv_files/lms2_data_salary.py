import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/Salary_Data.csv')
print(df)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values
print(x)
print(y)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size= 0.2,random_state= 0)
print(test_x)


Lin = LinearRegression()
Lin.fit(train_x,train_y)
print(Lin)

pred_y = Lin.predict(test_x)
print(pred_y)


san = Lin.predict([[15]])
print(san)



##error evaluation



#1
#MAE-mean absolute error
#2
#MSE-mean squared error

m = pred_y - test_y
print(m)

# n = np.abs(pred_y - test_y)
# print(n)

n = np.abs(pred_y - test_y).mean()
print(n)



from sklearn import metrics
c = metrics.mean_absolute_error(test_y,pred_y)
print(c)

s = metrics.mean_squared_error(test_y,pred_y)
print(s)


