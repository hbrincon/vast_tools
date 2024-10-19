import numpy as np
import pickle
from camb import model, initialpower, CAMBError
import camb
from astropy.cosmology import w0waCDM
import os
from scipy.interpolate import interpn,interp1d,LinearNDInterpolator
from scipy.interpolate import InterpolatedUnivariateSpline as int_uni_spl
from scipy.special import factorial, loggamma
import emcee
import corner


def logposterior(theta, data, data_log_bin_widths, priors, grids, bgrid, cgrid, sgrid, smin, smax, volumes):
    #theta must be a list or array, not a tuple
    #cast to array for scipy interpolaton functions
    theta = np.array(theta)

    lp, s_8 = logprior(theta, priors, grids, sgrid, smin, smax) # get the prior

    if not np.isfinite(lp):
        return -np.inf, np.inf

    return lp + loglikelihood(theta, data, data_log_bin_widths, grids, bgrid, cgrid, volumes), s_8


def loglikelihood(theta, data, data_log_bin_widths, grids, bgrid, cgrid, volumes):

    # input: grid points xyz, data f(xyz), values to interpolate at x0y0z0
    # returns f(x0,y0z0)

    #the interpolated void size function for theta
    vsf = np.array([[ interpn(grids, bgrid[...,j,i], theta)[0] for i in range(bgrid.shape[-1])] for j in range (bgrid.shape[-2])])
    vsf = vsf.astype(float)
    #the interpolated volume for theta
    vvm = np.array([interpn(grids, cgrid[...,i], theta) for i in range (bgrid.shape[-2])])
    vvm = vvm.astype(float)
    # log poisson likelihood
    # N(theta) = number density * fiducial survey volume * volume correction for model
    # Sum_{r,z} ( N(data) * log (N(theta)) - N(theta) - factorial(N(data)) )

    rtrn = np.sum((data*np.log(vsf * volumes * vvm * data_log_bin_widths)) \
                  - (vsf * volumes * vvm * data_log_bin_widths) - loggamma(data+1))

    if np.isnan(rtrn):
        return -np.inf
    else:
        return rtrn


def logprior(theta, priors, grids, sgrid, smin, smax):

    lp=0.
    for param, prior in zip(theta,priors):
        #uniform prior
        lp += 0. if prior[0] <= param <= prior[1] else -np.inf
    s_8 = np.inf
    if np.isfinite(lp):  
        s_8 = interpn(grids, sgrid, theta) [0]
        #uniform prior
        lp += 0. if smin <= s_8 <= smax else -np.inf

    return lp, s_8
    
def vsf_fitting(name, 
                data_counts,
                data_bins,
                data_log_bin_widths,
                volumes,
                Nens = 100,   # number of ensemble points
                Nburnin = 200, #400   # number of burn-in samples
                Nsamples = 500, #1000#2000  # number of final posterior samples
                smin=0, #minimum allowed sigma 8 value
                smax=2, #maxiimum allowed sigma 8 value
               ):
    
    # set up MCMC for cosmological parameters
    
    #get the number of redshift bins 
    with open(f'{name}_info.pickle','rb') as fn:
        pre_bin_centers, z_centers, _, _, _, _, _, _, _, Om_grid, w0_grid, wa_grid, h_grid, tau_grid,\
        Obh_grid, ns_grid, as_grid = pickle.load(fn)
    
    bgrid = np.load(f"{name}_bgrid.npy")
    cgrid = np.squeeze(np.load(f"{name}_cgrid.npy"))
    sgrid = np.squeeze(np.load(f"{name}_sgrid.npy"))
    
        
    #"""#interpolate bgrid to the appropriate data bins
    tmp = np.zeros(bgrid.shape[:-1]+(len(data_bins),))
    for i, Om in enumerate(Om_grid):
        for j, w0 in enumerate(w0_grid):
            for k, wa in enumerate(wa_grid):
                for l, h in enumerate(h_grid):
                    for m, tau in enumerate(tau_grid):
                        for n, Obh in enumerate(Obh_grid):
                            for o, ns in enumerate(ns_grid):
                                for p, As in enumerate(as_grid):
                                    for z in range(len(z_centers)):
                                        spl = int_uni_spl(pre_bin_centers, bgrid[i][j][k][l][m][n][o][p][z])
                                        tmp[i][j][k][l][m][n][o][p][z] = spl(data_bins)
    
    bgrid = np.squeeze(tmp)
    del tmp#"""
    #bgrid = np.squeeze(bgrid)
    if len(z_centers) == 1:
        bgrid = bgrid[...,np.newaxis,:]
        cgrid = cgrid[...,np.newaxis]
        sgrid = sgrid[...,np.newaxis]
    
        
    full_grids = np.array([Om_grid, w0_grid, wa_grid, h_grid, tau_grid,
                      Obh_grid, ns_grid, as_grid],dtype=object)
    
    select = np.array([len(grid) for grid in full_grids]) > 1
    grids = full_grids[select]


    priors = np.array([[grid[0]+0.02*(grid[-1]-grid[0]), grid[-1]-0.02*(grid[-1]-grid[0])] for grid in grids])

    inisamples = np.array([np.random.uniform(prior[0], prior[1], Nens) for prior in priors]).T
    ndims = inisamples.shape[1] # number of parameters/dimensions
    
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=[data_counts, data_log_bin_widths, priors, grids, bgrid, cgrid, sgrid, smin, smax, volumes], blobs_dtype=[("s_8", float)])
    sampler.run_mcmc(inisamples, Nsamples+Nburnin);
    
    labels = get_labels(Om_grid, w0_grid, wa_grid)
    
    with open(f'{name}_sampler.pickle','wb') as fn:
        pickle.dump((labels,sampler),fn)
    return sampler

def get_labels(Om_grid, w0_grid, wa_grid):
        
    labels = []
    
    if len(Om_grid)>1:
        labels.append("$\Omega_m$")
        
    if len(w0_grid)>1:
        if len(wa_grid)>1:
            labels.append("$w_0$","$w_a$")
        else:
            labels.append("$w$")
            
    labels.append("$\sigma_8$")
    
    return labels

def plot_chains(name):
    
    with open(f'{name}_info.pickle','rb') as fn:
        pre_bin_centers, z_centers, _, _, _, _, _, _, _, Om_grid, w0_grid, wa_grid, h_grid, tau_grid,\
        Obh_grid, ns_grid, as_grid = pickle.load(fn)
    
    with open(f'{name}_sampler.pickle','rb') as fn:
        sampler = pickle.load(fn)

    
    fig, axes = plt.subplots(len(labels),1, figsize=(10, 7), sharex=True,
                         gridspec_kw={'hspace':0.1})
    
    good_walkers = sampler.acceptance_fraction > 0
    
    samples = sampler.get_chain()[:,good_walkers,:]
    blobs = sampler.get_blobs()[:,good_walkers,:]
    
    samples = np.concatenate([samples , np.expand_dims(blobs.astype(float), axis=2)],axis=2)