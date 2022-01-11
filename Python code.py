"""Importing Libraries"""

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

"""Reading data from url"""

url="https://bit.ly/w-data"
data=pd.read_csv(url)
print("Data imported successfuly")
print(data)

"""Plotting the distribution of scores"""

data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

"""Divide data into attributes and labels"""

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 

"""Splitting the data and model training"""

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")

"""Plotting the regression line"""

line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line,color='red');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

"""Making predictions"""

y_pred=regressor.predict(X_test)
print(X_test)
print(y_pred)

"""Comparison between actual and predicted values"""  

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df

"""Evaluating training and test score"""

print("Training Score:",regressor.score(X_train,y_train))
print("Test score:",regressor.score(X_train,y_train))

"""Actual vs predicted value plot"""

df.plot(kind='bar',figsize=(7,5))
plt.show()

"""Taking number of hours studied as input"""

hours=float(input("Please enter hours studied = "))
result=np.array([hours])
result=result.reshape(-1,1)
prediction=regressor.predict(result)
print("number of hours studied= {}".format(hours) )
print("predicted percentage score= {}".format(prediction[0]) )

"""Finally evaluating the model"""

print('Mean Squared Error=', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error=', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error=',metrics.mean_absolute_error(y_test, y_pred)) 
print('R2 Score=', metrics.r2_score(y_test, y_pred))