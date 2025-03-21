import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Implementation of Gaussian basis functions
    :param centers: the centers of the basis functions, a numpy array with shape [M, d1, ...]
    :param beta: the variance of the Gaussian to use for the basis functions
    :return: a function that recieves as an input data points, a numpy array with shape [N, d1, ...]
             and returns the basis functions, a numpy array with shape [N, M]
    """
    y = centers.reshape(centers.shape[0], -1)
    def func(x: np.ndarray):
        x = x.reshape(x.shape[0], -1)
        D = np.sum(x*x, axis=1)[:, None] + np.sum(y*y, axis=1)[None, :] - 2*x@y.T
        D = D/y.shape[-1]
        return np.exp(-.5*D/beta)
    return func


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.mu = theta_mean                        # prior mean
        self.cov = theta_cov                        # prior covariance
        self.prec = np.linalg.inv(theta_cov)        # prior precision (inverse covariance)

        self.fit_mu = None                          # posterior mean
        self.fit_prec = None                        # posterior precision
        self.fit_cov = None                         # posterior covariance

        self.sig = sig                              # sample noise used to fit model
        self.h = basis_functions                    # basis functions used by the model

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        H = self.h(X)
        self.fit_mu = H.T@y[:, None]/self.sig + self.prec@self.mu[:, None]
        self.fit_prec = self.prec + H.T@H/self.sig
        self.fit_cov = np.linalg.inv(self.fit_prec)
        self.fit_mu = np.linalg.solve(self.fit_prec, self.fit_mu)[:, 0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        # if the model hasn't been trained, return the prior prediction
        if self.fit_mu is None: return (self.h(X) @ self.mu[:, None])[:, 0]

        # otherwise, return the MMSE prediction
        return (self.h(X) @ self.fit_mu[:, None])[:, 0]

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # if the model hasn't been trained, return the prior standard deviation
        if self.fit_mu is None: return np.sqrt(np.diagonal(self.h(X) @ self.cov @ self.h(X).T))

        # otherwise, return the variance of the MMSE prediction
        return np.sqrt(np.diagonal(self.h(X) @ np.linalg.solve(self.fit_prec, self.h(X).T)))


class Gaussian:
    def __init__(self, beta: float=0, nu: float=0):
        """
        Initialize a multivariate Gaussian model with a inverse-Wishart prior over the covariance
        :param beta: the value on the diagonal of the scale matrix (which is given by Psi=I*beta)
        :param nu: number of pseudo-observations of the scale matrix
        """
        self.mu = np.array([])
        self.cov = np.array([])
        self.beta, self.nu = beta if nu > 0 else 0, nu
        self.logdet = 0

    def fit(self, X: np.ndarray) -> 'Gaussian':
        """
        Fit the model to the training data X, using the ML solution for the mean and the MMSE solution for the
        covariance, under the prior defined when initializing the model
        :param X: training points as a numpy array with shape [N, ...], where N is the number of samples
        :return: the model after fitting it to the data
        """
        X = X.copy().reshape(X.shape[0], -1)
        self.mu = np.mean(X, axis=0)
        self.cov = (X-self.mu[None, :]).T @ (X-self.mu[None, :]) + self.beta*np.eye(X.shape[1])
        self.cov = self.cov/(X.shape[0] + self.nu)
        self.logdet = np.linalg.slogdet(self.cov)[1]
        return self

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the log-likelihood of the data points X under the trained model
        :param X: data points as a numpy array with shape [N, ...] where N is the number of samples
        :return: a numpy array of the log-likelihoods of all of the data points, as a numpy array of length N
        """
        m = X.copy().reshape(X.shape[0], -1) - self.mu[None, :]
        return -0.5*(np.sum(m.T * np.linalg.solve(self.cov, m.T), axis=0) + self.logdet + m.shape[-1]*np.log(np.pi*2))

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of the data points X under the trained model
        :param X: data points as a numpy array with shape [N, ...] where N is the number of samples
        :return: a numpy array of the likelihoods of all of the data points, as a numpy array of length N
        """
        return np.exp(self.log_likelihood(X))


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the accuracy of a given set of predictions with respect to the true labels
    :param preds: the predictions made by some model (not necessarily just -1 or 1), as a numpy array of length N
    :param labels: the true labels of the data, as a numpy array of length N
    :return: the accuracy of the predictions with respect to the true labels as a float
    """
    return np.mean(np.sign(preds) == labels)


def BLR_example():
    """
    Example usage of the BLR model and how to plot the decision boundary
    """
    N = 500
    np.random.seed(0)

    # sample N points from class 1
    x1 = np.sign(np.random.rand(N)-.5)[:, None]*np.array([1, -1])[None, :] + .5*np.random.randn(N, 2)
    # sample N points from class 2
    x2 = np.sign(np.random.rand(N)-.5)[:, None]*np.array([1, 1])[None, :] + .25*np.random.randn(N, 2)
    # labels for both classes
    y1 = np.ones(N)
    y2 = -np.ones(N)

    # concatenate the data into something we can pass to the BLR model (a train set and a test set)
    X = np.concatenate([x1[:-50], x2[:-50]], axis=0)
    y = np.concatenate([y1[:-50], y2[:-50]], axis=0)
    X_test = np.concatenate([x1[-50:], x2[-50:]], axis=0)
    y_test = np.concatenate([y1[-50:], y2[-50:]], axis=0)

    # fit a BLR model to the data
    beta, sigma = 2, .01
    bf = gaussian_basis_functions(centers=X[np.random.choice(X.shape[0], 20, replace=False)], beta=beta)
    blr = BayesianLinearRegression(theta_mean=np.zeros(20), theta_cov=np.eye(20), sig=sigma,
                                   basis_functions=bf).fit(X, y)

    # print train and test accuracies
    print(f'Train accuracy: {accuracy(blr.predict(X), y):.2f}')
    print(f'Test accuracy: {accuracy(blr.predict(X_test), y_test):.2f}')

    # create a meshgrid to show the decision contours
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    zz = [[blr.predict(np.array([xx[j, i], yy[j, i]])[None, :])[0] for i in range(xx.shape[1])]
          for j in range(xx.shape[0])]
    zz = np.array(zz)

    # find 25 points the model is least certain about
    d = np.abs(blr.predict(X)/blr.predict_std(X))
    ord = np.argsort(d)[:25]

    plt.figure()
    # plot the underlying contours
    plt.contourf(xx, yy, zz, alpha=.35, cmap='RdBu')
    # show scale of colors
    plt.colorbar()
    # plot the classes as points
    plt.scatter(x1[:, 0], x1[:, 1], 40, alpha=.7)
    plt.scatter(x2[:, 0], x2[:, 1], 40, alpha=.7)
    # plot the 25 points the model is least certain about as black points
    plt.scatter(X[ord, 0], X[ord, 1], 40, 'k', alpha=.7)
    # plot the decision boundary, i.e. when the regression returns 0
    plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='k')
    plt.show()


def QDA_example():
    """
    Example usage of the BLR model, as well as how to plot the decision boundary
    """
    N = 500
    np.random.seed(0)

    # sample N points from class 1
    x1 = np.sign(np.random.rand(N)-.5)[:, None]*np.array([1, -1])[None, :] + .5*np.random.randn(N, 2)
    # sample N points from class 2
    x2 = np.sign(np.random.rand(N)-.5)[:, None]*np.array([1, 1])[None, :] + .25*np.random.randn(N, 2)
    # labels for both classes
    y1 = np.ones(N)
    y2 = -np.ones(N)

    # make train and test sets
    X = np.concatenate([x1[:-50], x2[:-50]], axis=0)
    y = np.concatenate([y1[:-50], y2[:-50]], axis=0)
    X_test = np.concatenate([x1[-50:], x2[-50:]], axis=0)
    y_test = np.concatenate([y1[-50:], y2[-50:]], axis=0)

    # remove test points from training data
    x1, x2 = x1[:-50], x2[:-50]

    # fit a Gaussian to both classes
    nu = 100
    beta = .1*nu
    gauss1, gauss2 = Gaussian(beta, nu).fit(x1), Gaussian(beta, nu).fit(x2)

    # create function for fast predictions
    pred = lambda x: np.clip(gauss1.log_likelihood(x) - gauss2.log_likelihood(x), -25, 25)

    # print train and test accuracies
    print(f'Train accuracy: {accuracy(pred(X), y):.2f}')
    print(f'Test accuracy: {accuracy(pred(X_test), y_test):.2f}')

    # create a meshgrid to show the decision contours
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    zz = [[pred(np.array([xx[j, i], yy[j, i]])[None, :])[0] for i in range(xx.shape[1])]
          for j in range(xx.shape[0])]
    zz = np.array(zz)

    plt.figure()
    # plot the underlying contours
    plt.contourf(xx, yy, zz, alpha=.35, cmap='RdBu')
    # show scale of colors
    plt.colorbar()
    # plot the classes as points
    plt.scatter(x1[:, 0], x1[:, 1], 40, alpha=.7)
    plt.scatter(x2[:, 0], x2[:, 1], 40, alpha=.7)
    # plot the decision boundary, i.e. when the regression returns 0
    plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='k')
    plt.show()


def load_im_data():
    """
    Load the image data for the dogs vs. frogs classification task
    :return: the two tuples (dogs_train, dogs_test), (frogs_train, frogs_test) which are the tuples of train and
             test data
    """

    test_amnt = 250

    dogs = np.load('dogs.npy')
    dogs_test = dogs[-test_amnt:]
    dogs = dogs[:-test_amnt]

    frogs = np.load('frogs.npy')
    frogs_test = frogs[-test_amnt:]
    frogs = frogs[:-test_amnt]

    return (dogs, dogs_test), (frogs, frogs_test)


def plot_ims(image_list, title=''):
    """
    Plot images in a mosaic-style image
    :param image_list: a list/numpy array of images to plot
    :param title: the title of the plot
    """
    rows = np.where((len(image_list) % np.arange(1, np.floor(np.sqrt(len(image_list)) + 1))) == 0)[0][-1] + 1
    cols = len(image_list)//rows
    ims = np.concatenate([np.concatenate(image_list[i*rows:(i+1)*rows], axis=1) for i in range(cols)], axis=0)
    plt.figure(dpi=300)
    plt.imshow(ims)
    plt.axis('off')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    QDA_example()
    BLR_example()
    (dogs_train, dogs_t), (frogs_train, frogs_t) = load_im_data()
    plot_ims(dogs_train[:25], title='Dog examples')
    plot_ims(frogs_train[:25], title='Frog examples')

