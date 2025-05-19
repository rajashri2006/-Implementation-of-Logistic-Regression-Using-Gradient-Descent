# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.
. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Rahini A
RegisterNumber: 212223230165
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

## Dataset
![Screenshot 2025-03-28 054010](https://github.com/user-attachments/assets/07b49a1f-5e77-49d3-8522-0e00b603e02c)

## Dataset.dtypes
![Screenshot 2025-03-28 054036](https://github.com/user-attachments/assets/abe933de-b735-4af6-bfef-177d13779949)

## Dataset
![Screenshot 2025-03-28 054057](https://github.com/user-attachments/assets/4aed41ba-908b-4a3b-8f8b-e127a365a444)

## Y
![Screenshot 2025-03-28 054115](https://github.com/user-attachments/assets/308e1e2c-9eab-495b-bd36-2d29b6de4a62)

## Accuracy
![Screenshot 2025-03-28 054133](https://github.com/user-attachments/assets/d03b3b3f-f953-48ae-8b52-4b41d76bfd08)

## Y_pred
![Screenshot 2025-03-28 054152](https://github.com/user-attachments/assets/8e64ec14-f99f-48c9-944c-feeea32ba173)

## Y
![Screenshot 2025-03-28 054213](https://github.com/user-attachments/assets/ee0fd264-a705-4af2-91fc-fd2258fafb98)

## Y_prednew
![Screenshot 2025-03-28 054230](https://github.com/user-attachments/assets/6fa90528-b035-44df-a2c1-d0632ffa8e2a)

# Y_prednew
![Screenshot 2025-03-28 054249](https://github.com/user-attachments/assets/fb81a7d9-de98-4ea4-bd3d-6845540ab08b)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

