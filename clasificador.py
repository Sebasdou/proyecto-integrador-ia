import pandas as pd
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('card_evaluation_result.csv')

x = df.iloc[:, :-2]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)

clf = svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)
y_pred_svm = clf.predict(x_test)

print("Precisión con SVN:", accuracy_score(y_test, y_pred_svm))

cm = confusion_matrix(y_test,y_pred_svm)
print("Verdadero positivo:", cm[0][0])
print("Falso positivo:", cm[0][1])
print("Verdadero negativo:", cm[1][0])
print("Falso negativo:", cm[1][1])


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)
print("Precisión con KNN:", accuracy_score(y_test, y_pred_knn))

cm = confusion_matrix(y_test,y_pred_knn)
print("Verdadero positivo:", cm[0][0])
print("Falso positivo:", cm[0][1])
print("Verdadero negativo:", cm[1][0])
print("Falso negativo:", cm[1][1])
