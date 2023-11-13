# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.
## Program:
```PYTHON
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: GANESH R
RegisterNumber:  212222240029
*/

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy  

```

## Output:
## Result:
![ML1](https://github.com/ganesha360/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120884552/386a9482-fd2a-41bb-ada4-6fc09ec0a71e)

## Data.head():
![ML2](https://github.com/ganesha360/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120884552/222d9616-12b4-4629-b61f-fd8c42c0dccb)

## data.info():
![ML3](https://github.com/ganesha360/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120884552/f6f4d59b-be5c-4665-bc44-a5a3868993ae)

## data.isnull().sum():
![ML4](https://github.com/ganesha360/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120884552/2b4fda67-fba8-4725-b4fc-81c9a3593d96)

## Y prediction value:
![ML5](https://github.com/ganesha360/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120884552/3c31e0fc-4ca1-4c30-92f6-6eeece0ed698)

## Accuracy value:
![ML6](https://github.com/ganesha360/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120884552/8e9aed2f-e2a8-4523-88f7-d1cbb72f8fd7)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
