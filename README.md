# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices. 5.Display the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Zafreen J
RegisterNumber:  212223040252
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

data = pd.read_csv('Placement_Data.csv')
data1 = data.drop(["sl_no", "salary"], axis=1)

data1["gender"] = LabelEncoder().fit_transform(data1["gender"])
data1["ssc_b"] = LabelEncoder().fit_transform(data1["ssc_b"])
data1["hsc_b"] = LabelEncoder().fit_transform(data1["hsc_b"])
data1["hsc_s"] = LabelEncoder().fit_transform(data1["hsc_s"])
data1["degree_t"] = LabelEncoder().fit_transform(data1["degree_t"])
data1["workex"] = LabelEncoder().fit_transform(data1["workex"])
data1["specialisation"] = LabelEncoder().fit_transform(data1["specialisation"])
data1["status"] = LabelEncoder().fit_transform(data1["status"])

x = data1.iloc[:, :-1]
y = data1["status"]

imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)

print(lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1,85]]))

```


## Output:



![Screenshot 2024-09-16 160816](https://github.com/user-attachments/assets/32c24e06-c7be-4991-9fed-63a30fe6cf2a)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
