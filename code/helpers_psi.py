import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import ipywidgets as widgets


def plot_univariate_gaussian_pdf(mu, sig):
    """
    Plots a probability density function of a univariate Gaussian.
    Inputs: parameter values of the distribution
    Outputs: parameter values of the distribution, figure and axes handles of the plot
    """
    x_min = -10; x_max = 10; n_points = 100;
    x = np.linspace(x_min, x_max, n_points)
    densities = stats.norm.pdf(x, loc=mu, scale=sig)
    fig, ax = plt.subplots(figsize=[10, 10])
    ax.plot(x, densities, linestyle='-', color='b', linewidth=5)
    ax.set_ylim([0, 1]); ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
    return mu, sig, fig, ax


def plot_and_specify_univariate_gaussian_interactively():
    """
    Specify and interact with distribution parameter values of a univariate Gaussian.
    Inputs: -
    Outputs: interactive plot instance
    """

    interactive_plot = widgets.interactive(plot_univariate_gaussian_pdf,
                                           mu=widgets.FloatSlider(min=-2, max=2, step=0.5, description='$\mu$'),
                                           sig=widgets.FloatSlider(min=0.5, max=2, step=0.5, description='$\sigma$'))
    return interactive_plot


def histogram_estimation_of_univariate_data_density(data, nbins=100, min_edge=-10, max_edge=10):
    """
    Calculates a normalized histogram of data, to get an approximate density representation.
    Inputs:
        - data: data
        - nbins: amount of histogram bins to use
        - min_edge: smallest considered value in data
        - max_edge: largest considered value in data 
    Outputs:
        - bin_centers: centers of the bins
        - normalized_bin_counts: normalized bin counts, approximate densities
    """

    # specify bins
    bin_edges = np.linspace(min_edge, max_edge, nbins+1)
    bin_centers = 0.5*(bin_edges[0:-1]+bin_edges[1:])

    # calculate a normalized histogram
    [normalized_bin_counts, _] = np.histogram(data, bin_edges, density=True);

    return bin_centers, normalized_bin_counts

def kl_divergence_between_two_univariate_gaussians(mu1, sigma1, mu2, sigma2):
    """
    Calculates Kullback Leibler divergence between two univariate Gaussians;
    D_KL(p || q), where p is Gaussian with parameters mu1 and sigma1 and
    q is Gaussian with parameters mu2 and sigma2.

    Inputs: parameters of the two distributions
    Outputs: the KL-divergence
    """

    return np.log(sigma2/sigma1)+(sigma1**2+(mu1-mu2)**2)/(2*sigma2**2)-0.5


def plot_bivariate_gaussian_pdf(mu1, mu2, sig1, sig2, rho):
    """
    Plots a probability density function of a bivariate Gaussian.
    Inputs: parameter values of the distribution
    Outputs: parameter values of the distribution, figure and axes handles of the plot
    """
    x1_min = -5; x1_max = 5; n_points_x1 = 100; x2_min = -5; x2_max = 5; n_points_x2 = 100
    x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, n_points_x1), np.linspace(x2_min, x2_max, n_points_x2))
    x = np.empty((n_points_x1 * n_points_x2, 2), dtype='float')
    x[:, 0:1] = x1.reshape((n_points_x1 * n_points_x2, 1), order='F')
    x[:, 1:2] = x2.reshape((n_points_x1 * n_points_x2, 1), order='F')
    mu = np.empty((1, 2), dtype='float'); mu[0, 0] = mu1; mu[0, 1] = mu2
    K = np.empty((2, 2), dtype='float')
    K[0, 0] = sig1 ** 2  # Var[X_1]
    K[1, 1] = sig2 ** 2  # Var[X_2]
    K[0, 1] = rho * sig1 * sig2  # Cov[X_1, X_2]
    K[1, 0] = rho * sig1 * sig2  # Cov[X_2, X_1]
    densities = stats.multivariate_normal.pdf(x, mean=mu.flatten(), cov=K)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=[12, 12])
    ax.plot_surface(x1, x2, densities.reshape((n_points_x1, n_points_x2), order='F'), linewidth=0.5, rstride=1,
                    cstride=1, color='white', edgecolor='gray', alpha=0.2)
    ax.contour(x1, x2, densities.reshape((n_points_x1, n_points_x2), order='F'))
    ax.set_zlim([0, 1]); ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$'); ax.set_zlabel(r'$p(x_1, x_2|\mu, \Sigma)$')

    return mu, K, fig, ax


def plot_and_specify_bivariate_gaussian_interactively():
    """
    Specify and interact with distribution parameter values of a bivariate Gaussian.
    Inputs: -
    Outputs: interactive plot instance
    """

    interactive_plot = widgets.interactive(plot_bivariate_gaussian_pdf,
                                           mu1=widgets.FloatSlider(min=-2, max=2, step=.5, description=r'$\mu_1$'),
                                           mu2=widgets.FloatSlider(min=-2, max=2, step=.5, description=r'$\mu_2$'),
                                           sig1=widgets.FloatSlider(min=.5, max=2, step=.5, description=r'$\sigma_1$'),
                                           sig2=widgets.FloatSlider(min=.5, max=2, step=.5, description=r'$\sigma_2$'),
                                           rho=widgets.FloatSlider(min=-.75, max=.75, step=.25, description=r'$\rho$'))

    return interactive_plot

def kl_divergence_between_two_multivariate_gaussians(mu1, K1, mu2, K2):
    """
    Calculates Kullback Leibler divergence between two multivariate Gaussians;
    D_KL(p || q), where p is Gaussian with parameters mu1 and K1 and
    q is Gaussian with parameters mu2 and K2.

    Inputs: parameters of the two distributions
    Outputs: the KL-divergence
    """
    inv_K2 = np.linalg.inv(K2)
    D = mu2.size
    KL_divergence = np.trace(np.dot(inv_K2, K1)) + np.dot(mu2 - mu1, np.dot(inv_K2, np.transpose(mu2 - mu1)))
    KL_divergence += np.log(np.linalg.det(K2)/np.linalg.det(K1))-D
    KL_divergence = 0.5*KL_divergence

    return KL_divergence
