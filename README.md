### Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Yuvabharathi.B
RegisterNumber:  212222230181

import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test, y_pred)
accuracy

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

```

## Output:

data.head()

![31](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/4d28c2e4-8695-423d-bedf-c0f5872e071a)


data.info()

![32](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/03176d38-096a-4c6f-84dc-bd4e472e9e74)


isnull() and sum()

![33](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/6b200641-325e-4ed4-8b32-4b764c485a69)


data value counts()

![34](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/a34c3f8a-d8a7-45ef-946b-f700dba26503)


data.head() for salary

![35](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/1f88e6ab-a445-439c-8a70-c6f2424f79bb)


x.head()

![36](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/20db7b70-4424-4878-b361-9ad56702356d)


accuracy value 

![37](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/462f0d52-25e1-40c9-bb34-90ee54878974)


data prediction

![38](https://github.com/hariprasath5106/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/111515488/ee3041d3-f28b-43ae-b3a8-098df9d89f28)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
