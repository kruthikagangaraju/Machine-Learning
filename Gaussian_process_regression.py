import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels, GaussianProcessRegressor

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

# POSTERIOR WITHOUT NOISE
x_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
y_train = np.sin(x_train)
noise = 1e-8

K = kernel(x_train, x_train)
K_s = kernel(x_train, x_test)
K_ss = kernel(x_test, x_test)
K_inv = np.linalg.inv(K)
print(K.shape, K_s.shape, K_ss.shape, K_inv.shape)

mu_post1 = K_s.T.dot(K_inv).dot(y_train)
cov_post1 = K_ss - K_s.T.dot(K_inv).dot(K_s)
sd = np.sqrt(np.diag(cov_post1)).reshape(mu_post1.shape)

posterior_samples = np.random.multivariate_normal(mu_post1.ravel(), cov_post1, 3)

plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.plot(x_train, y_train, 'o', ms=8)
plt.plot(x_test, posterior_samples.T, '--')
plt.gca().fill_between(np.squeeze(x_test), np.squeeze(mu_post1 - 2*sd), np.squeeze(mu_post1 + 2*sd), color='aliceblue')
plt.plot(x_test, mu_post1)
plt.title("Posterior (Without Noise)")

# POSTERIOR USING L AND CHOLESKY (NO NOISE)
N = len(x_train)
L = np.linalg.cholesky(K)
L_s = np.linalg.solve(L, K_s)
mu_L = np.dot(L_s.T, np.linalg.solve(L, y_train))
sd = np.sqrt(np.diag(K_ss) - np.sum(L_s ** 2, axis=0)).reshape(mu_L.shape)
L_ss = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(L_s.T, L_s))

posterior_samples_L = mu_L.reshape(-1, 1) + np.dot(L_ss, np.random.normal(size=(n,3)))

plt.subplot(1,2,2)
plt.gca().fill_between(np.squeeze(x_test), np.squeeze(mu_L - 2*sd), np.squeeze(mu_L + 2*sd), color='aliceblue')
plt.plot(x_test, mu_L)
plt.title("Posterior using L and Cholesky (Without Noise)")

# SCIKIT sklearn
kernel_ = [kernels.RBF(), kernels.RationalQuadratic(), kernels.ExpSineSquared(periodicity=10.0), kernels.DotProduct(sigma_0=1.0)**2, kernels.Matern()]

for kernel in kernel_:
    gp = GaussianProcessRegressor(kernel=kernel)
    mu_prior, sd_prior = gp.predict(x_test, return_std=True)
    samples_prior = gp.sample_y(x_test, 3)

    print('#'*50)
    print(kernel)
    print('#'*50)

    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    plt.plot(x_test, mu_prior)
    plt.fill_between(x_test.ravel(), mu_prior - sd_prior, mu_prior + sd_prior, color='aliceblue')
    plt.plot(x_test, samples_prior, '--')
    plt.title('Prior')

    gp.fit(x_train, y_train)

    mu_post2, sd_post2 = gp.predict(x_test, return_std=True)
    mu_post2 = mu_post2.reshape(-1)
    samples_post = np.squeeze(gp.sample_y(x_test, 3))

    plt.subplot(1,2,2)
    plt.plot(x_test, mu_post2)
    plt.fill_between(x_test.ravel(), mu_post2 - sd_post2, mu_post2 + sd_post2, color='aliceblue')
    plt.plot(x_test, samples_post, '--')
    plt.scatter(x_train, y_train, c='blue', s=50)
    plt.title('Posterior')

    print("gp.kernel_", gp.kernel_)
    print("gp.log_marginal_likelihood", gp.log_marginal_likelihood(gp.kernel_.theta))

    print('-'*50, '\n\n')

plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.show()
