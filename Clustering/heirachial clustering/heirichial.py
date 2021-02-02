import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Clustering\heirachial clustering\Mall_Customers.csv')
X = dataset.iloc[:, 3:].values

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=3)
y_pred=hc.fit_predict(X)

print(y_pred)

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='green',label='Cluster 3')
plt.title('K-Means')
plt.xlabel('Salary')
plt.ylabel('Score')
plt.show()