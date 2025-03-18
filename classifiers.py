import numpy as np
from matplotlib import pyplot as plt
from utils import load_im_data, BayesianLinearRegression, gaussian_basis_functions, accuracy, Gaussian, plot_ims


def main():
    # ------------------------------------------------------ MMSE Estimate for µ_+ and µ_-
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_p, mu_m = np.array([1, 1]), np.array([-1, -1])

    # sample 5 points from each class
    np.random.seed(0)
    x_p = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_m = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)

    # Calculate MMSE estimates for µ_+ and µ_- using equation 1.3
    N_p, N_m = len(x_p), len(x_m)
    precision_p = N_p / sig + 1 / sig_0
    precision_m = N_m / sig + 1 / sig_0

    mu_p_post = (np.sum(x_p, axis=0) / sig + mu_p / sig_0) / precision_p
    mu_m_post = (np.sum(x_m, axis=0) / sig + mu_m / sig_0) / precision_m

    # Plot sampled points and mean decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(x_p[:, 0], x_p[:, 1], color='blue', label='x^(+)', s=50)
    plt.scatter(x_m[:, 0], x_m[:, 1], color='red', label='x^(-)', s=50)

    # Calculate and plot the mean decision boundary using equation 1.6
    mu_diff = mu_p_post - mu_m_post
    mu_norm_diff = (np.linalg.norm(mu_p_post) ** 2 - np.linalg.norm(mu_m_post) ** 2) / 2
    x_vals = np.linspace(-1.5, 1.5, 100)
    y_vals = (mu_norm_diff - mu_diff[0] * x_vals) / mu_diff[1]
    plt.plot(x_vals, y_vals, color='green', label='Mean Decision Boundary', linewidth=2)

    # Sample 10 decision boundaries
    np.random.seed(1)
    for _ in range(10):
        mu_p_sample = mu_p_post + np.random.randn(2) / np.sqrt(precision_p)
        mu_m_sample = mu_m_post + np.random.randn(2) / np.sqrt(precision_m)
        mu_diff_sample = mu_p_sample - mu_m_sample
        mu_norm_diff_sample = (np.linalg.norm(mu_p_sample) ** 2 - np.linalg.norm(mu_m_sample) ** 2) / 2
        y_sample = (mu_norm_diff_sample - mu_diff_sample[0] * x_vals) / mu_diff_sample[1]
        plt.plot(x_vals, y_sample, color='gray', alpha=0.5)

    # Finalize plot
    plt.title('Decision Boundaries with Sampled Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # ------------------------------------------------------ QDA & BLR
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()

    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])

    # ------------------------------------------------------ QDA Accuracy vs ν
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)

        # Fit Gaussians to the data for each class
        gauss_dogs = Gaussian(beta, nu).fit(dogs)
        gauss_frogs = Gaussian(beta, nu).fit(frogs)

        # Define the prediction function
        def predict(x):
            log_likelihood_dogs = gauss_dogs.log_likelihood(x)
            log_likelihood_frogs = gauss_frogs.log_likelihood(x)
            return np.clip(log_likelihood_dogs - log_likelihood_frogs, -25, 25)

        # Calculate training and test accuracies
        train_score[i] = accuracy(predict(train), labels)
        test_score[i] = accuracy(predict(test), labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel(r'Value of $\nu$')
    plt.title('QDA Accuracy vs $\nu$')
    plt.grid()
    plt.show()

    # Print specific test accuracies for ν = 0 and ν = 25
    print(f'Test accuracy for nu=0: {test_score[nus.index(0)]:.2f}')
    print(f'Test accuracy for nu=25: {test_score[nus.index(25)]:.2f}')

    # ------------------------------------------------------ BLR Accuracy vs M
    # define question variables
    beta = .02
    sigma = .1
    Ms = [250, 500, 750, 1000, 2000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ms)), np.zeros(len(Ms))

    blr = None
    for i, M in enumerate(Ms):
        print(f'Gaussian basis functions using {M} samples', end='', flush=True)

        # Randomly sample M centers from dogs and M from frogs (2M centers total)
        dog_centers = dogs[np.random.choice(dogs.shape[0], M, replace=False)]
        frog_centers = frogs[np.random.choice(frogs.shape[0], M, replace=False)]
        # centers = np.concatenate([dog_centers, frog_centers], axis=0)
        centers = np.concatenate([frog_centers, dog_centers], axis=0)

        # Define basis functions using the selected centers
        basis_functions = gaussian_basis_functions(centers=centers, beta=beta)

        # Initialize and train the Bayesian Linear Regression model
        blr = BayesianLinearRegression(
            theta_mean=np.zeros(2 * M),  # 2M centers
            theta_cov=np.eye(2 * M),  # Covariance size matches the number of centers
            sig=sigma,
            basis_functions=basis_functions
        ).fit(train, labels)

        # Calculate training and test accuracy
        train_score[i] = accuracy(blr.predict(train), labels)
        test_score[i] = accuracy(blr.predict(test), labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ms, train_score, lw=2, label='train')
    plt.plot(Ms, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('# of Samples (M)')
    plt.title('Training and Test Accuracy vs # of Basis Functions')
    plt.xscale('log')
    plt.show()

    # Print test accuracy for M = all
    print(f'Test accuracy for M=all (5750): {test_score[-1]:.2f}')

    # calculate how certain the model is about the predictions
    d = np.abs(blr.predict(dogs_t) / blr.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')


if __name__ == '__main__':
    main()







