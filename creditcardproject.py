import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss,EditedNearestNeighbours
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
datetime.now()

df = pd.read_csv(r'E:\Dyche Data Science class\Data Mining in python\File_1\creditcard.csv')
df.Class.value_counts()

df.drop('Time',axis=1,inplace=True)
x = df.drop('Class',axis=1)
y = df.Class
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=142,stratify=y)
x_train.shape
x_test.shape
print(x_train.shape[0]/x_train.shape[1])
print(x_test.shape[0]/x_test.shape[1])
nrm = MinMaxScaler()
x_train = nrm.fit_transform(x_train)
x_test = nrm.fit_transform(x_test)

clf = RandomForestClassifier()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
y_pred = clf.predict(x_test)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)


sm = SMOTE()
# seed = 100
# k = 3
x_train_res,y_train_res = sm.fit_sample(x_train,y_train)
y_train.shape
y_train_res.shape

mn = MultinomialNB()
mn.fit(x_train_res,y_train_res)
y_pred_balanced = mn.predict(x_test)
accuracy_score(y_test,y_pred_balanced)
precision_score(y_test,y_pred_balanced)
recall_score(y_test,y_pred_balanced)
f1_score(y_test,y_pred_balanced)
plot_confusion_matrix(mn,x_test,y_test)

#under sampling
un = NearMiss()
x_train_res_un,y_train_res_un = un.fit_sample(x_train,y_train)
y_train.shape
y_train_res_un.shape

mn = MultinomialNB()
mn.fit(x_train_res_un,y_train_res_un)
y_pred_balanced = mn.predict(x_test)
accuracy_score(y_test,y_pred_balanced)
precision_score(y_test,y_pred_balanced)
recall_score(y_test,y_pred_balanced)
f1_score(y_test,y_pred_balanced)

enn = EditedNearestNeighbours()
x_train_res_enn,y_train_res_enn = enn.fit_sample(x_train,y_train)
y_train.shape
y_train_res_un.shape

mn = MultinomialNB()
mn.fit(x_train_res_enn,y_train_res_ennn)
y_pred_balanced = mn.predict(x_test)
accuracy_score(y_test,y_pred_balanced)
precision_score(y_test,y_pred_balanced)
recall_score(y_test,y_pred_balanced)
f1_score(y_test,y_pred_balanced)