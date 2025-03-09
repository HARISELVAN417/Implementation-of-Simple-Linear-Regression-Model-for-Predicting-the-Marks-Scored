# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. .Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Datatest:
![image](https://github.com/user-attachments/assets/f8986e1e-0a81-45ea-afe8-e5ea30523a38)

Headvalues:
![image](https://github.com/user-attachments/assets/740c3e98-d789-4f54-befd-86cd5b4f266f)

Tail values:
![image](https://github.com/user-attachments/assets/9cde1a5a-d2f3-41fe-8ea8-b726e5a5f3bf)

X and Y values:
![image](https://github.com/user-attachments/assets/b1de4311-5024-43f1-ad43-781a0b47e4c8)

pridication values of x and y:
![image](https://github.com/user-attachments/assets/0afb6ac4-1d36-42d8-8260-aeca74eb3ca3)

MSE,MAE and RMSE:
![image](https://github.com/user-attachments/assets/b115e3f8-3045-431d-824b-f64758212560)

Training set:
![image](https://github.com/user-attachments/assets/dba57365-346a-4ed2-8e2d-c01c978528b0)

Testing set:
![image](https://github.com/user-attachments/assets/ad0bb3e5-1d05-4883-99bb-5173e941ae60)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
