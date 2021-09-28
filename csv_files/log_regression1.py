from typing import ValuesView
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/Social_Network_Ads.csv")
print(df)


s = df.corr()
print(s)

x = df[["Age","EstimatedSalary"]].values
print(x)

y = df["Purchased"].values
print(y)

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

#to calculate individual values

a = classifier.predict(sc.transform([[43,125000]]))
print(a)





#DATA VISUALAZATION


# import numpy as np
# from matplotlib import pyplot as plt

# ys = 200 + np.random.randn(100)
# x = [x for x in range(len(ys))]

# plt.plot(x, ys, '-')
# plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

# plt.title("Sample Visualization")
# plt.show()


