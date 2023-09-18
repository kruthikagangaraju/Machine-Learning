import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def kernel(x1, x2, l=1.0, sigma=1.0):
    d = (x1 - x2.T) ** 2
    return sigma * np.exp(-d/(2 * 1 ** 2))

np.random.seed(0)
n = 50
x_test = np.linspace(-5, 5, n).reshape(-1, 1)
mu = np.zeros(x_test.shape)
cov = kernel(x_test, x_test)
prior_samples = np.random.multivariate_normal(mu.reshape(-1), cov, 3)
print(prior_samples.shape)
plt.plot(x_test, prior_samples.T)

x_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
y_train = np.sin(x_train)
noise = 1e-8

K = kernel(x_train, x_train)
K_s = kernel(x_train, x_test)
K_ss = kernel(x_test, x_test)
K_inv = np.linalg.inv(K)
print(K.shape, K_s.shape, K_ss.shape, K_inv.shape)

mu_post = K_s.T.dot(K_inv).dot(y_train)
cov_post = K_ss - K_s.T.dot(K_inv).dot(K_s)
sd = np.sqrt(np.diag(cov_post)).reshape(mu_post.shape)

posterior_samples = np.random.multivariate_normal(mu_post.ravel(), cov_post, 3)

plt.plot(x_train, y_train, 'o', ms=8)
plt.plot(x_test, posterior_samples.T, '--')
plt.gca().fill_between(np.squeeze(x_test), np.squeeze(mu_post - 2*sd), np.squeeze(mu_post + 2*sd), color='aliceblue')
plt.plot(x_test, mu_post)

N = len(x_train)
L = np.linalg.cholesky(K)
L_s = np.linalg.solve(L, K_s)
mu_L = np.dot(L_s.T, np.linalg.solve(L, y_train))
sd = np.sqrt(np.diag(K_ss) - np.sum(L_s ** 2, axis=0)).reshape(mu_L.shape)
L_ss = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(L_s.T, L_s))

posterior_samples_L = mu_L.reshape(-1, 1) +np.dot(L, np.random.normal(size=(n,3)))

plt.gca().fill_between(np.squeeze(x_test), np.squeeze(mu_L - 2*sd), np.squeeze(mu_L + 2*sd), color='aliceblue')
plt.plot(x_test, mu_L)

plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.show()
