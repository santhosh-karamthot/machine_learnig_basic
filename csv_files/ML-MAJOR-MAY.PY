# importing libraries  
import numpy as np 
import matplotlib.pyplot as plt  
import pandas as pd  
  
#importing datasets  
df= pd.read_csv("C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/digit_svm.csv")  
print(df)


#Extracting Independent and dependent Variable  
x= df.iloc[:, 0].values.reshape(-1,1) 
y= df.iloc[:, 2].values 
print(x)
print(y)  



# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0) 
# y_train = y_train.astype(np.int8)
print(x_test)


#feature Scaling  
from sklearn.preprocessing import StandardScaler    
sc= StandardScaler()

x_train= sc.fit_transform(x_train)    
x_test= sc.fit_transform(x_test) 
print(x_test)



# #support vector machine algorithm

# from sklearn.svm import SVC 
# clf = SVC(kernel="linear", random_state=0)
# # self.classes_ = np.unique(y)
# # np.unique(y_train)
# clf.fit(x_train,y_train)
# y.value_counts()  
# print(clf)



# # #Predicting the test set result  
# y_pred = clf.predict(x_test) 




# #divide data frame into two parts

# df1 = df[df.cluster==0]
# df2 = df[df.cluster==1]

# print(df1)
# print(df2)

# #for label and colurs for the axies and data sets
# plt.scatter(df1['X'],df1['Y'],c ="green")
# plt.scatter(df2['X'],df2['Y'],c ="red")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()




# # #Creating the Confusion matrix 


# from sklearn.metrics import confusion_matrix  
# cm= confusion_matrix(y_test,y_pred) 
# print(cm)



# #classification matrics


# from sklearn.metrics import accuracy_score
# w = accuracy_score(y_test,y_pred)
# print(w)







import numpy as np
from PIL import Image

number = int(input("what's the number: "))

img = Image.open('C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/three.png').convert('L')

#print np.array(img)
img_arr = np.array(img)

#print img_arr.flatten()

WIDTH, HEIGHT = img.size

data = list(img.getdata()) # convert image data to a list of integers
# convert that to 2D list (list of lists of integers)
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# individually using data[row][col].

# For example:
for row in data:
    print(' '.join('{:3}'.format(value) for value in row))



import numpy as np
from PIL import Image
import pandas as pd

#for i in range(0,10,1):

num = int(input("what's the number: "))

img = Image.open('C:/Users/SANTHOSH KUMAR/Documents/machine learning/csv files/three.png').convert('L')

print(np.array(img))
img_arr = np.array(img)

print(img_arr.flatten())

flat_img_arr = img_arr.flatten()

#print "check whether understanding of flattening works or not: "

print(img_arr[3,15], flat_img_arr[99]) # works quite well 

img_list = flat_img_arr.tolist()

print(type(img_list))
print(img_list[99], "length of the list: ", len(img_list))

# add the number in the list of pixels as a form of label
img_list.insert(0, num)

print("length of the list: ", len(img_list))

print(img_list)



