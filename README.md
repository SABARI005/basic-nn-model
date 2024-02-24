# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://github.com/SABARI005/basic-nn-model/assets/118660461/414b5b8e-ff25-42e2-9970-e368e8cdd3a6)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM:

### Name: SABARI S
### Register Number: 212222240085

```python
from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('dlexp1').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'X':'float'})
df = df.astype({'Y':'float'})
df.head(10)

X = df[['X']].values
Y = df[['Y']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train = Scaler.transform(X_train)

AI_brain = Sequential([
    Dense(units = 4, activation = 'relu',input_shape = [1]),
    Dense(units = 3, activation = 'relu'),
    Dense(units = 1)
    ])

AI_brain.summary()

AI_brain.compile(optimizer = 'rmsprop', loss = 'mse')

AI_brain.fit(X_train, Y_train, epochs = 9000)

loss_df = pd.DataFrame(AI_brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

X_new = [[9]]

X_neww = Scaler.transform(X_new)

AI_brain.predict(X_neww)
```
## Dataset Information
![image](https://github.com/SABARI005/basic-nn-model/assets/118660461/a81ebfe2-49d3-43e1-a285-ddf876679ae0)




## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/SABARI005/basic-nn-model/assets/118660461/b34d7bd6-421e-4e65-8df1-071dd448daca)


### Test Data Root Mean Squared Error
![image](https://github.com/SABARI005/basic-nn-model/assets/118660461/4b8e1d7c-38f4-4c21-b9f7-f9e15f84aa17)


### New Sample Data Prediction
![image](https://github.com/SABARI005/basic-nn-model/assets/118660461/c881a688-54e1-4959-835c-d6d75e3fe7e4)


## RESULT
A neural network regression model for the given dataset has been developed Sucessfully.
