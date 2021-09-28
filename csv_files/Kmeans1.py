from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
 
df = pd.read_csv("C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/k means 1.csv")
print(df)


plt.scatter(df['X'],df['Y'])
plt.show()
# print(plt)


#K MEANS CLUSTTERING

from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)
y_pred = km.fit_predict(df)
print(y_pred)



#adding new column with name clustter

df['cluster'] = y_pred
print(df)

#divide data frame into two parts

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]

print(df1)
print(df2)



#for label and colurs for the axies and data sets
plt.scatter(df1['X'],df1['Y'],c ="green")
plt.scatter(df2['X'],df2['Y'],c ="red")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


#for finding the centroid of the data

c1 = km.cluster_centers_[:,0]
c2 = km.cluster_centers_[:,1]
print(c1)
print(c2)

plt.scatter(df1['X'],df1['Y'],c ="green")
plt.scatter(df2['X'],df2['Y'],c ="red")
plt.scatter(c1,c2,c="purple",marker="*")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()







