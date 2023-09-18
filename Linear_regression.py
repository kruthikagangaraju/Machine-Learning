import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression

# LR USING SCIKIT LEARN

# df = pd.read_csv('placement.csv')
#
# X = df[['cgpa']]
# Y = df['package']
#
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# lr = LinearRegression()
# lr.fit(X_train, Y_train)
#
# Y_pred = lr.predict(X_test)
#
# x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
# y_range = lr.predict(x_range)
#
# m = lr.coef_
# c = lr.intercept_
#
#
# plt.scatter(X_train, Y_train, label='Training Data')
# plt.scatter(X_test, Y_test, color='red', label='Test Data')
# plt.plot(x_range, y_range, color='green', label='Linear Regression Line')
# plt.xlabel('CGPA')
# plt.ylabel('Package (in LPA)')
# plt.legend()
# plt.show()

class LR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, Y_train):
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean()) * (Y_train[i] - Y_train.mean()))
            den = den + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))

        self.m = num / den
        self.b = Y_train.mean() - (self.m * X_train.mean())

    def predict(self, X_test):
        return self.m * X_test + self.b

df = pd.read_csv('placement.csv')

X = df.iloc[:, 0].values
Y = df.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

lr = LR()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = lr.predict(x_range)

plt.scatter(X_train, Y_train, label='Training Data')
plt.scatter(X_test, Y_test, color='red', label='Test Data')
plt.plot(x_range, y_range, color='green', label='Linear Regression Line')
plt.xlabel('CGPA')
plt.ylabel('Package (in LPA)')
plt.legend()
plt.show()
