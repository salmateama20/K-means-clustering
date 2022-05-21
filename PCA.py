#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import preprocessing
data = pd.read_csv("F:\group.csv")

data=data.iloc[:1000]
normal_data=preprocessing.normalize(data)
data=pd.DataFrame(normal_data)
df=pd.DataFrame(data)
df.columns=['Height','Weight','BMI','Length between the shoulders','Length of the Arms']
X = df[["Height","Weight","BMI","Length between the shoulders","Length of the Arms"]]
def PCA(dataf , num_components):
  X_meaned = dataf - np.mean(X , axis = 0)
  cov_mat = np.cov(X_meaned , rowvar = False)
  eigen_values,eigen_vectors = np.linalg.eigh(cov_mat)
#sort the eigenvalues in descending order
  sorted_index = np.argsort(eigen_values)[::-1]
  sorted_eigenvalue = eigen_values[sorted_index]
#similarly sort the eigenvectors 
  sorted_eigenvectors = eigen_vectors[:,sorted_index]
# select the first n eigenvectors, n is desired dimension
# of our final reduced data.
  n_components = 2 #you can select any number of components.
  eigenvector_subset = sorted_eigenvectors[:,0:n_components]
  X_reduced = np.dot(eigenvector_subset.transpose(),X_meaned.transpose()).transpose()
  return X_reduced
  X_reduced
x = df.iloc[:,:]
 
#prepare the target
target = df.iloc[:,:]
 
#Applying it to PCA function
mat_reduced = PCA(x , 2)
 
#Creating a Pandas DataFrame of reduced Dataset
PCA_reduced = pd.DataFrame(mat_reduced , columns = ['Height','Weight'])
 
#Concat it with target variable to create a complete Dataset
PCA_reduced = pd.concat([PCA_reduced] , axis = 1)
plt.figure(figsize = (5,5))
sb.scatterplot(data = PCA_reduced , x = 'Height',y = 'Weight' )


# In[3]:


data=data.iloc[:1000]
Reduced = PCA_reduced[["Height","Weight"]]
#Visualise data points

K=3

Centroids = (PCA_reduced.sample(n=K))
plt.scatter(Reduced["Height"],Reduced["Weight"],c='black')
plt.scatter(Centroids["Height"],Centroids["Weight"],c='red')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
Centroids = (Reduced.sample(n=K))
diff = 1
j=0

while(diff!=0):
    XD=Reduced
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Height"]-row_d["Height"])**2
            d2=(row_c["Weight"]-row_d["Weight"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        Reduced[i]=ED
        i=i+1

    C=[]
    for index,row in Reduced.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    Reduced["Cluster"]=C
    Centroids_new = Reduced.groupby(["Cluster"]).mean()[["Height","Weight"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = ((Centroids_new['Height'] - Centroids['Height']).sum()) + ((Centroids_new['Weight'] - Centroids['Weight']).sum())
        print(diff.sum())
    Centroids = Reduced.groupby(["Cluster"]).mean()[["Height","Weight"]]
    color=['yellow','green','cyan','orange','purple']
for k in range(K):
    data=Reduced[Reduced["Cluster"]==k+1]
    plt.scatter(data["Height"],data["Weight"],c=color[k])
plt.scatter(Centroids["Height"],Centroids["Weight"],c='red')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()


# In[ ]:





# In[ ]:




