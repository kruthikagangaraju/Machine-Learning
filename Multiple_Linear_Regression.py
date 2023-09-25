from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, Y = make_regression(n_samples=100, n_features=2, n_informative=2, n_targets=1, noise=50)
df = pd.DataFrame({'feature1': X[:,0], 'feature2': X[:,1], 'target': Y})

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
lr = LinearRegression()
lr.fit(X_train, Y_train)
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
Y_pred = lr.predict(X_test)

print("MAE", mean_absolute_error(Y_test, Y_pred))
print("MSE", mean_squared_error(Y_test, Y_pred))
print("R2 Score", r2_score(Y_test, Y_pred))

x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
xGrid, yGrid = np.meshgrid(y, x)
final = np.vstack((xGrid.ravel().reshape(1, 100), yGrid.ravel().reshape(1, 100))).T
z_final = lr.predict(final).reshape(10, 10)
z = z_final

fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')
fig.add_trace(go.Surface(x=x, y=y, z=z))
fig.show()
