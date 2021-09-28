import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/K MEANS2.csv")
print(df)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_pred = km.fit_predict(df[['Age', 'Income($)']])
print(y_pred)


sse = []
k_rng = range(1,10)
for k in k_rng:
  km = KMeans(n_clusters=k)
  km.fit(df[['Age','Income($)']])
  sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()


#