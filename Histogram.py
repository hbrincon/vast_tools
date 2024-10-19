import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

# PCI from Garwood, 1936
def poisson_central_interval(x, width):
    alpha = 0.5*(1 - width)
    lower = chi2.isf(1-alpha, 2*x)/2
    upper = chi2.isf(alpha, 2*(x+1))/2
    return lower, upper

#Simple numpy histogram calculation with PCI
def make_histogram(data, domain, bins, hist_kwargs = None):
    
    if hist_kwargs is None:
        counts, edges = np.histogram(data, range = domain, bins=bins)
    else:
        counts, edges = np.histogram(data, range = domain, bins=bins, **hist_kwargs)
    centers = edges[:-1] + np.diff(edges)/2
    error = poisson_central_interval(counts, .68)
    
    return counts, edges, centers, error

#Plot histogram
def histogram(data, domain, bins, normalize = False, fill_alpha=.35, 
              bin_weights = None,
              hist_kwargs = None, plot_kwargs = None, fill_kwargs = None): #kwargs should be dictionaries
    
    counts, edges, centers, error = make_histogram(data, domain, bins, hist_kwargs)
    
    if bin_weights is not None:
        counts = counts.astype(float) * bin_weights
        error = (error[0] * bin_weights, error[1] * bin_weights)
        
    if normalize == True:
        #normalize the histogram
        norm = np.sum(counts) * np.diff(edges)
        counts = counts.astype(float) / norm
        
        error = (error[0] / norm, error[1] / norm)
        
    
    #plot the histogram
    if plot_kwargs is None:
        plt.plot(centers, counts)
    else:
        plt.plot(centers, counts, **plot_kwargs)
    
    #plot the histogram error
    if fill_kwargs is None:
        plt.fill_between(centers, error[1], error[0], alpha=fill_alpha)
    else:
        plt.fill_between(centers, error[1], error[0], alpha=fill_alpha, **fill_kwargs)