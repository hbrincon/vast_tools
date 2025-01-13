import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from astropy.cosmology import FlatLambdaCDM

"""
Authors: Hernan Rincon

Some code has been adopted from the following individuals: Segev BenZvi
"""

def poisson_central_interval(x, width):
    """
    Calculates a Poisson central interval as given by Garwood (1936).
    See https://doi.org/10.2307/2333958
    
    params:
    ---------------------------------------------------------------------------------------------
    x (numpy array of floats): The sample data
    
    width (float): the width of the interval, given as a value between 0 and 1. A 68% interval
        would be given as 0.68.
        
        
    returns:
    ---------------------------------------------------------------------------------------------
    lower (numpy array of floats): the lower error bar for each data point.
    
    upper (numpy array of floats): the upper error bar for each data point.
    """
    
    alpha = 0.5*(1 - width)
    lower = chi2.isf(1-alpha, 2*x)/2
    upper = chi2.isf(alpha, 2*(x+1))/2
    return lower, upper

#Simple numpy histogram calculation with PCI
def make_histogram(data, domain, bins, hist_kwargs = None):
    """
    Creates a numpy histogram, together with bin errors determined by the 68% central interval of
    the data.
    
    params:
    ---------------------------------------------------------------------------------------------
    data (numpy array of floats): The histogram bin counts
    
    domain (tuple of two floats): the domain of the histogram.
    
    bins (int or array of floats): The bin arguemnt passed to the numpy histogram function. If an
        integer is passed, bins sets the number of bisn in the histogram. If an array is passed,
        bins sets the bin edges.
    
    hist_kwargs (dictionary): keyword arguments passed to numpy.histogram. 
    
    
    returns:
    ---------------------------------------------------------------------------------------------
    counts (numpy array of ints): The histogram bin counts
    
    edges (numpy array of floats): The histogram bin edges
    
    centers (numpy array of floats): The hsitogram bin centers
    
    error (numpy array of floats): The Poisson error in the histogram bin counts
    """
    
    # Call numpy.histogram with or without keyword arguments
    if hist_kwargs is None:
        counts, edges = np.histogram(data, range = domain, bins=bins)
    else:
        counts, edges = np.histogram(data, range = domain, bins=bins, **hist_kwargs)
    
    # The centers of the bins
    centers = edges[:-1] + np.diff(edges)/2
    
    # The 68% central interval around the data
    error = poisson_central_interval(counts, .68)
    
    # Return the histogram information
    return counts, edges, centers, error

#Plot histogram
def histogram(data, domain, bins, normalize = False, fill_alpha=.35, 
              bin_weights = None,
              hist_kwargs = None, plot_kwargs = None, fill_kwargs = None): #kwargs should be dictionaries
    """
    Plots a histogram with matplotlib, together with bin errors determined by the 68% central 
    interval of the data. The histogram is plotted with matplotlib.pyplot.plot rather than
    matplotlib.pyplot.hist.
    
    params:
    ---------------------------------------------------------------------------------------------
    data (numpy array of floats): The histogram bin counts
    
    domain (tuple of two floats): the domain of the histogram.
    
    bins (int or array of floats): The bin arguemnt passed to the numpy histogram function. If an
        integer is passed, bins sets the number of bisn in the histogram. If an array is passed,
        bins sets the bin edges.
        
    normalize (bool): If True, the histogram is normalized to integrate to unity. Defaults to 
        False.
    
    fill_alpha (float): The alpha value for the shaded region denoting the error in the bin 
        counts. Defaults to 0.35.
    
    bin_weights (numpy array of floats with a size of len(data)): Bin weights used to rescale the
        histogram bin counts. If set to None, no bin weights are used. Defaults to None.
    
    hist_kwargs (dictionary): keyword arguments passed to numpy.histogram.
    
    plot_kwargs (dictionary): keyword arguments passed to matplotlib.pyplot.plot.
    
    fill_kwargs (dictionary): keyword arguments passed to matplotlib.pyplot.fill_between.
    """
    
    # Get the histogram data
    counts, edges, centers, error = make_histogram(data, domain, bins, hist_kwargs)
    
    # Optionally apply weights to the bin counts
    if bin_weights is not None:
        counts = counts.astype(float) * bin_weights
        error = (error[0] * bin_weights, error[1] * bin_weights)
    
    # Optionally normalize the histogram
    if normalize == True:
        norm = np.sum(counts) * np.diff(edges)
        counts = counts.astype(float) / norm
        
        error = (error[0] / norm, error[1] / norm)
        
    # Plot the histogram
    if plot_kwargs is None:
        plt.plot(centers, counts)
    else:
        plt.plot(centers, counts, **plot_kwargs)
    
    # Plot the histogram error
    if fill_kwargs is None:
        plt.fill_between(centers, error[1], error[0], alpha=fill_alpha)
    else:
        plt.fill_between(centers, error[1], error[0], alpha=fill_alpha, **fill_kwargs)
        
        
def n_of_z(data, domain, bins, sky_fraction, is_redshift = True, kwargs=None):
    """
    Plots a histogram representing a radial n(z) selection function, where the data bins are 
    normalized by spherical shells of the survey volume.
    
    params:
    ---------------------------------------------------------------------------------------------
    data (numpy array of floats): The histogram bin counts
    
    domain (tuple of two floats): the domain of the histogram.
    
    bins (int or array of floats): The bin arguemnt passed to the numpy histogram function. If an
        integer is passed, bins sets the number of bisn in the histogram. If an array is passed,
        bins sets the bin edges.
        
    sky_fraction (float): A float betwee 0 and 1 representing the fraction of the sky that the 
        survey covers
    
    is_redshift (bool): A boolena denoting whether the bin edges are in units of redshift (True) 
        or comoving distance (False). Defaults to True.
    
    kwargs (dictionary): keyword arguments passed to histogram.
    """
    
    # Fiducial LambdaCDM model
    model = FlatLambdaCDM(H0=100, Om0=.315)
    
    # Convert integer number of bins to bin edges
    if type(bins) is int:
        bins = np.linspace(domain[0], domain[1], bins)
    
    # Convert bin edges to comoving units
    if is_redshift:
        comov_bins = model.comoving_distance(bins)
    else:
        comov_bins = bins
        
    # Calculate weights from bin edges
    weights = 1 / (sky_fraction * 4/3 * np.pi * np.diff(comov_bins**3))
    weights = np.array(weights)

    # Plot histogram
    if kwargs is None:
        histogram(data, domain, bins, bin_weights = weights)
    else:
        histogram(data, domain, bins, bin_weights = weights, **kwargs)
    