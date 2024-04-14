#!/usr/bin/env python

# Standard library imports
import pickle

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

size = 300
X = np.random.rand(size) * 5 - 2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4 * (X**4) + w3 * (X**3) + w2 * (X**2) + w1 * X + w0 + np.random.randn(size) * 8 - 4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv', index=None)
df.plot.scatter(x='x', y='y')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

reg_lin_test = lin_reg.predict(X_test)
reg_lin_train = lin_reg.predict(X_train)

plt.scatter(X_test, y_test)
plt.plot(X_test, reg_lin_test, color='green')
plt.show()

knn_reg3 = KNeighborsRegressor(n_neighbors=3)
knn_reg3.fit(X_train, y_train)
knn3_test = knn_reg3.predict(X_test)
knn3_train = knn_reg3.predict(X_train)

plt.scatter(X_test, y_test)
plt.scatter(X_test, knn3_test, color='green')
plt.show()

knn_reg5 = KNeighborsRegressor(n_neighbors=5)
knn_reg5.fit(X_train, y_train)
knn5_test = knn_reg5.predict(X_test)
knn5_train = knn_reg5.predict(X_train)

plt.scatter(X_test, y_test)
plt.scatter(X_test, knn5_test, color='green')
plt.show()

poly_features2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly_features2.fit_transform(X_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_train_poly2, y_train)

poly2_test = lin_reg2.predict(poly_features2.transform(X_test))
poly2_train = lin_reg2.predict(poly_features2.transform(X_train))

plt.scatter(X_test, y_test)
plt.scatter(X_test, poly2_test, color='green')
plt.show()

poly_features3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly3 = poly_features3.fit_transform(X_train)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_train_poly3, y_train)

poly3_test = lin_reg3.predict(poly_features3.transform(X_test))
poly3_train = lin_reg3.predict(poly_features3.transform(X_train))

plt.scatter(X_test, y_test)
plt.scatter(X_test, poly3_test, color='green')
plt.show()

poly_features4 = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly4 = poly_features4.fit_transform(X_train)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_train_poly4, y_train)

poly4_test = lin_reg4.predict(poly_features4.transform(X_test))
poly4_train = lin_reg4.predict(poly_features4.transform(X_train))

plt.scatter(X_test, y_test)
plt.scatter(X_test, poly4_test, color='green')
plt.show()

poly_features5 = PolynomialFeatures(degree=5, include_bias=False)
X_train_poly5 = poly_features5.fit_transform(X_train)
lin_reg5 = LinearRegression()
lin_reg5.fit(X_train_poly5, y_train)

poly5_test = lin_reg5.predict(poly_features5.transform(X_test))
poly5_train = lin_reg5.predict(poly_features5.transform(X_train))

plt.scatter(X_test, y_test)
plt.scatter(X_test, poly5_test, color='green')
plt.show()

data = [
    [mean_squared_error(y_train, reg_lin_train), mean_squared_error(y_test, reg_lin_test)], 
    [mean_squared_error(y_train, knn3_train), mean_squared_error(y_test, knn3_test)], 
    [mean_squared_error(y_train, knn5_train), mean_squared_error(y_test, knn5_test)], 
    [mean_squared_error(y_train, poly2_train), mean_squared_error(y_test, poly2_test)], 
    [mean_squared_error(y_train, poly3_train), mean_squared_error(y_test, poly3_test)], 
    [mean_squared_error(y_train, poly4_train), mean_squared_error(y_test, poly4_test)], 
    [mean_squared_error(y_train, poly5_train), mean_squared_error(y_test, poly5_test)]
]

mse = pd.DataFrame(data, columns=['train_mse', 'test_mse'], 
                   index=['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg'])
print(mse)
with open('mse.pkl', 'wb') as output_file:
    pickle.dump(mse, output_file)

regressors = [
    (lin_reg, None), (knn_reg3, None), (knn_reg5, None), 
    (lin_reg2, poly_features2), (lin_reg3, poly_features3), 
    (lin_reg4, poly_features4), (lin_reg5, poly_features5)
]

with open('reg.pkl', 'wb') as output_file:
    pickle.dump(regressors, output_file)
