import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()
df = pd.DataFrame(boston['data'])
df.columns = boston['feature_names']
df['Price'] = boston['target']

x = df.drop('Price', axis=1)
x_crm = df['CRIM'].values[: np.newaxis]
y = df['Price'].values

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.33, random_state = 5)

# X_train = X_train.reshape(-1,1)
# X_test = X_test.reshape(-1,1)

lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

plt.scatter(Y_test, Y_pred, color='black')

plt.title('Crime vs Price')
plt.ylabel('Predicted Price')
plt.xlabel('Crime Rate')
plt.show()

print("R squared : ", sklearn.metrics.r2_score(Y_test, Y_pred))
print("Mean Absolute Error : ", sklearn.metrics.mean_absolute_error(Y_test, Y_pred))
print("Mean Squared Error : ", sklearn.metrics.mean_squared_error(Y_test, Y_pred))
print("Root Mean Square : ", np.sqrt(sklearn.metrics.mean_squared_error(Y_test, Y_pred)))

pd.plotting.scatter_matrix(df)
plt.show()