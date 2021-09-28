
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/LSM1.csv")
print(df)

# x = df.iloc[:,0].values
# opp = x.shape
# print(opp)
x = df.iloc[:,0].values.reshape(5,1)
opp = x.shape
print(opp)

print(x)

y = df.iloc[:,1].values

print(y)

# from sklearn.linear_model import LinearRegression
# lin = LinearRegression()
# print(lin)

# from sklearn import linear_model
# reg = linear_model.LinearRegression()
#1
from sklearn import linear_model
reg = linear_model.LinearRegression()
oppp = reg.fit(x,y)    #train the model
lin = reg.predict([[2.5]])   #testing the model 
print(oppp)
print(lin)
ram = reg.coef_  #for checking the coefficient of x
print(ram)
san = reg.intercept_  #for print the intercept of c     #equqtion is y = mx+c
print(san)


# plt.scatter(x,y)
# ren = plt.show()
# print(ren)

pred_y = reg.predict(x)
# print(pred_y)
plt.scatter(x,y)
plt.plot(x,pred_y,color = 'g',marker = "*",markersize = 10)
rav = plt.show()
print(rav)


#2


# from sklearn.linear_model import LinearRegression
# lin = LinearRegression()
# lin.fit(x,y)
# lin.predict([[2.5]])
# print(lin)



#example for linearregression with comment for understand perpose

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# # Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print('Mean squared error: %.2f'
#       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(diabetes_y_test, diabetes_y_pred))

# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
