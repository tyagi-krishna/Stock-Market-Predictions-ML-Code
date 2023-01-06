#importing all the libraries for the program
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

#using the aplha vantage api to get the stockmarket data of a company
key = "VIYRYDIX8IIRZP99"
ts = TimeSeries(key, output_format='pandas')
ti = TechIndicators(key)
aapl_data, aapl_meta_data = ts.get_daily_adjusted(symbol='aapl') # you can change the company code to get the data of another company
aapl_sma, aapl_met_sma = ti.get_sma(symbol='aapl')
print('Data Collected Successfully from the Alpha Vantage API...')


#reshaping and adjusting data into arrays to feed into the machine learning model
price = aapl_data.iloc[:, 0]
y = np.array(price).reshape((-1,1))
numbers = list(range(1,101))
x = np.array(numbers).reshape((-1,1))
x_predict = np.array([101]).reshape((-1,1))

#splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = SEED)


poly = PolynomialFeatures(degree=10, include_bias=False)
poly_feature = poly.fit_transform(X_train)
poly_feature1 = poly.fit_transform(x)
poly_predict = poly.fit_transform(x_predict)
poly_y_test = poly.fit_transform(y_test)


# creating and training the linear regression model
regressor = LinearRegression()
regressor.fit(poly_feature, y_train)
print('Model Created Successfully...')
print('The Intercept is:', regressor.intercept_)

loss = log_loss(X_test, regressor.predict_proba(X_test), eps=1e-15)
print("The Loss is "loss)


#making predictions using the model
#predictions = regressor.predict(np.array([101]).reshape(1,-1))
predictions = regressor.predict(poly_predict)
print('The Predicted Value is:',predictions)
ynew1 = np.append(x, [101])
ynew = np.array(ynew1).reshape((-1,1))
predicted = np.append(y, [predictions])

# creating array to store all the predicted values
predicted_y = regressor.predict(poly_feature1)
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
