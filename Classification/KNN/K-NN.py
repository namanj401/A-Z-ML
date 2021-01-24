import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Classification\KNN\Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)  

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)

print(classifier.predict(sc.transform([[30,87000]])))

y_pred=classifier.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix
cn = confusion_matrix(y_test,y_pred)
print(cn)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

 