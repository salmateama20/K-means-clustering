#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
data = pd.read_csv("F:\group.csv")
normal_data=preprocessing.normalize(data)
data=pd.DataFrame(normal_data)
df=pd.DataFrame(data)
df.columns=['Height','Weight','BMI','Length between the shoulders','Length of the Arms']
X = df[["Height","Weight","BMI","Length between the shoulders","Length of the Arms"]]
data=data.iloc[:1000]
def K_Mean(X,K):
    global run_time
    start = time.time()
    Centroids = (X.sample(n=K))
    diff = 1
    j=0

    while(diff!=0):
        XD=X
        i=1
        for index1,row_c in Centroids.iterrows():
            ED=[]
            for index2,row_d in XD.iterrows():
                d1=(row_c["Height"]-row_d["Height"])**2
                d2=(row_c["Weight"]-row_d["Weight"])**2
                d3=(row_c["BMI"]-row_d["BMI"])**2
                d4=(row_c["Length between the shoulders"]-row_d["Length between the shoulders"])**2
                d5=(row_c["Length of the Arms"]-row_d["Length of the Arms"])**2
                d=np.sqrt(d1+d2+d3+d4+d5)
                ED.append(d)
            X[i]=ED
            i=i+1

        C=[]
        for index,row in X.iterrows():
            min_dist=row[1]
            pos=1
            for i in range(K):
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        X["Cluster"]=C
        Centroids_new = X.groupby(["Cluster"]).mean()[["Height","Weight","BMI","Length between the shoulders","Length of the Arms"]]
        if j == 0:
            diff=1
            j=j+1
        else:
            
            diff = ((Centroids_new['Height'] - Centroids['Height']).sum()) + ((Centroids_new['Weight'] - Centroids['Weight']).sum())+ ((Centroids_new['BMI'] - Centroids['BMI']).sum()) + ((Centroids_new['Length between the shoulders'] - Centroids['Length between the shoulders']).sum()) + ((Centroids_new['Length of the Arms'] - Centroids['Length of the Arms']).sum())
            print(diff.sum())
        Centroids = X.groupby(["Cluster"]).mean()[["Height","Weight","BMI","Length between the shoulders","Length of the Arms"]]


    stop = time.time()
    run_time.append(stop - start)
N = [100,200,500,1000]
K = [3,5]

for i in K:
    run_time = []
    for j in N:
        tmp_data = data.iloc[:j]
        normal_data = preprocessing.normalize(tmp_data)
        df = pd.DataFrame(normal_data)
        df.columns=['Height','Weight','BMI','Length between the shoulders','Length of the Arms']
        K_Mean(df,i)
    
    plt.plot(N,run_time)
    plt.xlabel('different N values')
    plt.ylabel('Run Time')
    plt.title(f'run time for k={i}')
    plt.show()


# In[ ]:





# In[ ]:




