#importing all the libraries for the program
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#using the aplha vantage api to get the stockmarket data of a company
key = "VIYRYDIX8IIRZP99"
ts = TimeSeries(key, output_format='pandas')
ti = TechIndicators(key)
aapl_data, aapl_meta_data = ts.get_daily_adjusted(symbol='INFY') # you can change the company code to get the data of another company
aapl_sma, aapl_met_sma = ti.get_sma(symbol='INFY')
print('Data Collected Successfully from the Alpha Vantage API...')


# reshaping and adjusting data into arrays to feed into the machine learning model
price = aapl_data.iloc[:, 0]
y = np.array(price).reshape((-1,1))
numbers = list(range(1,101))
x = np.array(numbers).reshape((-1,1))

# splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = SEED)


# creating and training the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('Model Created Successfully...')
print('The Intercept is:', regressor.intercept_)


# making predictions using the model
predictions = regressor.predict([[101]])
print('The Predicted Value is:',predictions)
ynew1 = np.append(x, [101])
ynew = np.array(ynew1).reshape((-1,1))
predicted = np.append(y, [predictions])

# testing the effectiveness of the model using rmse value 
y_pred = regressor.predict(X_test)
print('The RMSE Value of Model is: (Should be less than 1 for the Model to be Effective)', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # if value is less than 1 then model can be used but we preffer a value under 0.8

# creating array to store all the predicted values
predicted_y = regressor.predict(x)
y_pred_1 = predicted_y

#plotting the graph to show the values
plt.plot(x, y_pred_1)
plt.plot(ynew, predicted)
plt.plot(x, y)
plt.xlabel('Number')
plt.ylabel('Stock Price')
plt.legend(['Previously Predicted Values', 'Next Prediction', 'True Values'])
plt.grid()
plt.show()











