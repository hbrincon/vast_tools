#from mcmc import *
import os
import shutil
#import tqdm
from scipy.interpolate import LinearNDInterpolator, CubicSpline
#from scipy.stats import truncnorm
from multiprocessing import Pool
import numpy as np
import emcee
import pickle
import camb
from camb import model, initialpower, CAMBError
import multiprocessing
from itertools import repeat
import sys
sys.path.insert(1, '/global/homes/h/hrincon/python_tools')
import Histogram as hist
sys.path.insert(1, '/global/homes/h/hrincon/code/CosmoBolognaLib/Python')

import CosmoBolognaLib as cbl
from scipy.special import factorial, loggamma
from astropy.cosmology import w0waCDM, FlatLambdaCDM
import matplotlib.pyplot as plt
from vast.voidfinder.postprocessing import mknum

"""
Authors: Hernan Rincon

Some code has been adopted from the following individuals: Dahlia Veyrat
"""

#code still in development, errors may occur when used

#class for fitting the void size function to void spectrum data
class VoSS_Fit ():
    def __init__(self, s8_mode = 'free', B_mode='fixed', model = 'lambdaCDM'):
        """
        ---------------------------------------------------------------------------------------
        Creates a VoSS_Fit object.
        
        params:
        ---------------------------------------------------------------------------------------
        s8_mode (string): How sigma_8 is calculated for the VSF. One of 'fixed', 'free', 
            'derived', 'marginalized_free', or 'marginalized_derived'.
        
        model (string): The cosmological model used for the VSF. One of 'lambdaCDM','wCDM',
            'w0waCDM', or 'bias'. Of these options, 'bias' used lambdaCDM but callibrates the
            bias for each redshift bin and should generally be ran before using other models.
        """
        #identify free parameters for model
        self.model = model
        if model not in ('lambdaCDM','wCDM','w0waCDM', 'bias'):
            raise ValueError(f"Model parameter must be one of 'bias', 'lambdaCDM', 'wCDM', or 'w0waCDM'. The user input was '{model}'")
        om_mode = 'free'
        if model == 'bias':
            om_mode = 'fixed'
            s8_mode = 'fixed'
            if B_mode == 'fixed':
                self.bias = VoSS_Parameter ('bias', 2, 1.5, 3, mode='free')
        w0_mode = 'fixed'
        wa_mode = 'fixed'
        if model == 'wCDM':
            w0_mode = 'free'
        if model == 'w0waCDM':
            w0_mode = 'free'
            wa_mode = 'free'
        
        # set model parameters
        self.om = VoSS_Parameter ('om', 0.315, 0.1, 1-1e-3, mode=om_mode) #matter content
        self.omb = VoSS_Parameter ('omb', 0.0493, 1e-3, 0.1-1e-3) #baryon content
        self.h = VoSS_Parameter ('h', 0.6736, 0.5, 1) #reduced hubble parameter
        self.ns = VoSS_Parameter ('ns', 0.9649, 0.5, 1) #spectral index
        self.a_s = VoSS_Parameter ('as', 2.083e-09, 1e-09, 3e-09) #scalar amplitude
        self.s8 = VoSS_Parameter ('s8', 0.8111, .5, 1.5, mode=s8_mode) #sigma_8
        self.tau = VoSS_Parameter ('tau', 0.0544, 0.01, 0.09) #optical depth to reionization
        self.w0 = VoSS_Parameter ('w0', -1, -2, 0, mode=w0_mode) # DEEoS 0th order expansion constant
        self.wa = VoSS_Parameter ('wa', 0, -3, 3, mode=wa_mode) # DEEoS first order expansion constant
        self.slope = VoSS_Parameter('slope', 1, -5, 5, mode=B_mode)
        self.intercept = VoSS_Parameter('intercept', 0, -5, 5, mode=B_mode)
        
        #dictionary of parameters
        self.parameters = {'om':self.om, 'omb':self.omb, 'h':self.h, 'ns':self.ns, 'as':self.a_s, 's8':self.s8, 
                           'tau':self.tau, 'w0':self.w0, 'wa':self.wa, 'slope':self.slope, 'intercept':self.intercept}
        if model == 'bias' and B_mode == 'fixed':
            self.parameters['bias'] = self.bias 
        
    def update_vsfobj(self, vsfobj):
        """
        ---------------------------------------------------------------------------------------
        Updates the fiducial cosmology of a VoidSizeFunction object to match the fiducial 
        cosmology of a VoSS_Fit instance.
        
        params:
        ---------------------------------------------------------------------------------------
        vsfobj (VoidSizeFunction): The VoidSizeFunction to be updated
        """
        
        vsfobj.set_fid_cosmology( self.om.fiducial_value, self.w0.fiducial_value, self.wa.fiducial_value, 
                                      self.h.fiducial_value, self.tau.fiducial_value, self.omb.fiducial_value, 
                                      self.ns.fiducial_value, self.a_s.fiducial_value)
        
    def set_parameter(self, name, fiducial_value, min_value, max_value, num_values, mode):
        """
        ---------------------------------------------------------------------------------------
        Updates the fiducial value and prior range of a paramter for the void size spectrum 
        MCMC.
        
        params:
        ---------------------------------------------------------------------------------------
        name (string): The name of the paramter to be updated. Must match one of the keys in 
            the VoSS_Fit instance's parameter dictionary.
        
        fiducial_value (float): the fiducial value of the paramter.
        
        min_value (float): The minimum value for the paratmer to be considered in the MCMC
        
        max_value (float): The maximum value for the paratmer to be considered in the MCMC
        
        num_values (int): The number of points between min_value and max_value at which the 
        parameter is sampled for an emulator grid used for the MCMC. A higher value gives a more 
        accurate result at the cost of computation time.
        """
        
        #update the parameter
        if name in ('om','omb','h','ns','as','s8','tau','w0','wa', 'slope', 'intercept'):
            self.parameters[name] = VoSS_Parameter(name, fiducial_value, min_value, max_value, num_values, mode)
        else:
            raise ValueError(f"The cosmological parameter must be one of 'om','omb','h','ns','as','s8','tau','w0','wa', 'slope', or 'intercept'. The user input was '{name}'")
    
    def set_data (self, bin_counts, bin_edges, bin_volumes, z_centers, z_edges, bin_biases = None, bias_burnin = 500, bias_method = 'VoSS', bias_mcmc_file = None):
        if not isinstance(bin_counts, np.ndarray):
            bin_counts = np.array(bin_counts)
        if len(bin_counts.shape) == 1:
            bin_counts = np.array([bin_counts])
        self.bin_counts = bin_counts
        
        if not isinstance(bin_edges, np.ndarray):
            bin_edges = np.array(bin_edges)
        if len(bin_edges.shape) == 1:
            bin_edges = np.array([bin_edges])
        self.bin_edges = bin_edges

        self.bin_centers = np.array([bin_edge[:-1] + np.diff(bin_edge)/2 for bin_edge in bin_edges])
        
        if not isinstance(bin_volumes, np.ndarray):
            bin_volumes = np.array(bin_volumes)
        if bin_volumes.shape[0] > 1:
            bin_volumes = np.expand_dims(bin_volumes, 1)
        self.bin_volumes = bin_volumes
        
        if not isinstance(z_centers, np.ndarray):
            z_centers = np.array(z_centers)
        self.z_centers = z_centers
        
        if not isinstance(z_edges, np.ndarray):
            z_edges = np.array(z_edges)
        self.z_edges = z_edges
        
        if bin_biases is not None:
            if not isinstance(bin_biases, np.ndarray):
                bin_biases = np.array(bin_biases)
            self.bin_biases = bin_biases
            
        elif os.path.isfile(bias_mcmc_file):
            bin_biases = []
            with open (bias_mcmc_file,'rb') as temp_infile:
                bias_dict = pickle.load(temp_infile)
            """if interpolate_bias:
                z = []
                f = []
                for key in bias_dict.keys():
                    z.append(key)
                    f.append(np.mean(bias_dict[redshift][1][bias_burnin:,].flatten()))
                spline = CubicSpline(z, f)
                for redshift in z_centers:
                    bin_biases.append(spline(redshift))
            else:"""
            
            for redshift in z_centers:
                bin_biases.append(np.mean(bias_dict[redshift][1][bias_burnin:,].flatten()))
            self.bin_biases = np.array(bin_biases)
        
    def get_parameter_space (self):
        
        # identify parameters for mcmc
        axes = []
        point_names = []
        
        for key in self.parameters.keys():
            parameter = self.parameters[key]
            if parameter.mode in ('free', 'marginalized', 'marginalized_free'):
                axes.append(parameter.get_grid())
                point_names.append(parameter.name)
        
        #convert axes to list of points
        points = np.vstack(list(map(np.ravel, np.meshgrid(*axes)))).T 
        
        return points, point_names
    
    def get_interpolator (self, interpolator_name, method = 'VoSS', num_processes = 1, overwrite = False):
        
        print('Forming interpolation grid.')
        
        if os.path.isfile(interpolator_name) and not overwrite:
            with open (interpolator_name,'rb') as temp_infile:
                interp = pickle.load(temp_infile)
            print('The interpolator name was matched to an existing file. Exiting.')
            return
        
        if method not in ('VoSS', 'CBL'):
            raise ValueError(f"method parameter must be one of 'VoSS' or 'CBL'. The user input was '{method}'")
            
        derive_s8 = self.parameters['s8'].mode in ( 'derived', 'marginalized_derived')
        
        # list of points and the names of the cosmological parameters
        parameter_space_points, coord_names = self.get_parameter_space()
        
        cosmology = self.get_cosmology()
        
        num_points = len(parameter_space_points)
        
        #parallel version
        if num_processes > 1:
            pool = multiprocessing.Pool(num_processes)
            inputs = zip(parameter_space_points, repeat(cosmology), repeat(coord_names), repeat(derive_s8), repeat(method))
            values, s8_values = zip(*pool.starmap(self.evaluate_parameter_space_point, inputs))
        #serial version
        else:
            values = []
            s8_values = []
            #iterate over parameter space grid and calculate emulator grid values
            for i, parameter_space_point in enumerate(parameter_space_points):
                print(f'Evaluating point {i+1} of {num_points}', end='\r')
                vsf, s8 = self.evaluate_parameter_space_point(parameter_space_point, cosmology, coord_names, derive_s8, method)

                values.append(vsf)
                s8_values.append(s8)
        print(f'Evaluation of all points {num_points}/{num_points} are complete')
        
        #create interpolator for result
        if 'bias' in self.parameters.keys():
            
            #format 1D output
            values = np.array(values) #list to array
            parameter_space_points = parameter_space_points.flatten() #1D array for 1D parameter space
            
            #create a posterior probability spline for every redshift bin 
            #note: shape of values is (num param points, num redshift bins, num radial bins)
            interp = [CubicSpline(parameter_space_points, values[:,i,:]) for i in range(values.shape[1])]
            
            #derive s8 shouldn't be set to True for the bias callibration, but if it is, provide a None object
            if derive_s8:
                interp_s8 = None
        else:
            #1D interpolation
            if parameter_space_points.shape[1] == 1:
                
                #format 1D output
                parameter_space_points = parameter_space_points.flatten()
                
                #create posterior probability spline
                interp = CubicSpline(parameter_space_points, values)
                
                #create s8 spline
                if derive_s8:
                    interp_s8 = CubicSpline(parameter_space_points, s8_values)
            
            #ND interpolation
            else:
                
                #create posterior probability spline
                interp = LinearNDInterpolator(parameter_space_points, values)
                
                #create s8 spline
                if derive_s8:
                    interp_s8 = LinearNDInterpolator(parameter_space_points, s8_values)
        
        #save interpolator
        with open (interpolator_name,'wb') as temp_infile:
            if derive_s8:
                pickle.dump((interp, method, interp_s8), temp_infile)
            else:
                pickle.dump((interp, method), temp_infile)
            
    def get_cosmology(self):
        
        cosmology = {}
        for coord_name in self.parameters.keys():
            cosmology[coord_name] = self.parameters[coord_name].fiducial_value
        return cosmology
    
    def plot_parameter_space_point(self, bin_index, parameter_space_point, coord_names, derive_s8, include_cbl = True, 
                                   plot_data = True, plot_theory = True, data_kwargs = None, data_fill_kwargs = None, theory_kwargs = None, cbl_kwargs = None,
                                  normalize=False):
        
        cosmology = self.get_cosmology()

        if plot_theory:
            vsf, s8 = self.evaluate_parameter_space_point(parameter_space_point, cosmology, coord_names, derive_s8, 'VoSS')
            print('VoSS derived sigma_8 is ', s8)

        i = bin_index # rename for convenience
        
        z = self.z_centers[i]
        bias = self.bin_biases[i]
        
        data_scale = 1
        if normalize:
            bin_log_widths = np.diff(np.log(self.bin_edges[i]))
            volume = self.bin_volumes[i]
            data_scale = 1 / volume / bin_log_widths
        
        #plot data
        if plot_data:
            
            bin_counts = self.bin_counts[i]
            uncert_low, uncert_high = hist.poisson_central_interval(bin_counts,.68)
            
            if data_kwargs is not None:
                plt.plot(self.bin_centers[i], bin_counts * data_scale, **data_kwargs) 
            else:
                plt.plot(self.bin_centers[i], bin_counts * data_scale, color='purple') 
                
            if data_fill_kwargs is not None:
                plt.fill_between(self.bin_centers[i], uncert_high * data_scale, uncert_low * data_scale, **data_fill_kwargs)
            else:
                plt.fill_between(self.bin_centers[i], uncert_high * data_scale, uncert_low * data_scale, alpha=.35,color='purple', label = 'Size spectrum')

        #plot theory VoSS
        if plot_theory:
            if theory_kwargs is not None:
                plt.plot(self.bin_centers[i], vsf[i]*data_scale, **theory_kwargs)
            else:
                plt.plot(self.bin_centers[i], vsf[i]*data_scale, color='k', alpha=0.5, label = 'VoSS VSF')
        
        #report liklihood
        if plot_data and plot_theory:
            print('VoSS bin likelihood:',np.sum((bin_counts*np.log(vsf[i])) - vsf[i] - loggamma(bin_counts+1)))
        
        #plot theory CBL
        if include_cbl:
            
            vsf, s8 = self.evaluate_parameter_space_point(parameter_space_point, cosmology, coord_names, derive_s8, 'CBL')
            print('CBL derived sigma_8 is ', s8)
            
            if cbl_kwargs is not None:
                plt.plot(self.bin_centers[i], vsf[i]*data_scale, **cbl_kwargs)
            else:
                plt.plot(self.bin_centers[i], vsf[i]*data_scale, color='r', linestyle=":", alpha=0.5, label = 'CBL VSF')
                
            if plot_data and include_cbl:    
                print('CBL bin likelihood:',np.sum((bin_counts*np.log(vsf[i])) - vsf[i] - loggamma(bin_counts+1)))
            
        plt.legend(fontsize=12,title_fontsize=13)
        plt.xlabel('Void Radius $R_s$ [Mpc/h]',fontsize=15)
        plt.ylabel('Void Size Function',fontsize=15)
        plt.xticks(fontsize=25)#;
        plt.yticks(fontsize=13)
        plt.xscale('log');plt.yscale('log')
    
    def evaluate_parameter_space_point(self, parameter_space_point, cosmology, coord_names, derive_s8, method):
        
        for coord, coord_name in zip(parameter_space_point, coord_names):
            cosmology[coord_name] = coord

        if method == 'VoSS':
            vsf, s8 = self.evaluate_VoSS(cosmology, derive_s8)
        elif method == 'CBL':
            vsf, s8 = self.evaluate_CBL(cosmology, derive_s8)
        else:
            raise ValueError (f"Method must be one of 'VoSS' or 'CBL'. The user input was '{method}'.")

        return vsf, s8
    
    def evaluate_VoSS(self, parameters, derive_s8):
        
        #raise error if bin_biases is non
        
        try:
            s8 = None if derive_s8 else parameters['s8']
            
            fiducial_linear_bias = self.bin_biases if 'bias' not in parameters.keys() else parameters['bias'] * np.ones_like(self.bin_biases)
            #bias = parameters['slope'] * bias + parameters['intercept']
            
            vsfobj = VoidSizeFunction()
            self.update_vsfobj(vsfobj)
            
            #calculated_s8, spectrum, vol_correction = self.vsfobj.spectrum(
            calculated_s8, spectrum, vol_correction = vsfobj.spectrum(
                    self.bin_centers, self.z_centers, self.z_edges,#shell radii and redshift sampling
                    parameters['om'], parameters['w0'], parameters['wa'], parameters['h'], parameters['tau'] , 
                    parameters['omb'], parameters['ns'], parameters['as'],     
                    fiducial_linear_bias, parameters['slope'], parameters['intercept'],
                    delta_tr_v = -0.7, delta_c = 1.686, kmax=2, 
                    scale_to_bin = True, bin_edges = self.bin_edges,
                    get_sigma_8=derive_s8, sigma_8 = s8
                )
            
            if vol_correction.shape[0] > 1:
                vol_correction = np.expand_dims(vol_correction, 1)

            vsf = spectrum * self.bin_volumes * vol_correction
            
            if derive_s8:
                s8 = calculated_s8

            return vsf, s8

        except CAMBError:
            return np.full(self.bin_centers.shape, np.inf), np.inf
        
    def evaluate_CBL(self, parameters, derive_s8):
        
            
        biases = self.bin_biases if 'bias' not in parameters.keys() else parameters['bias'] * np.ones_like(self.bin_biases)
        biases = parameters['slope'] * biases + parameters['intercept']

        # catch unphysical bias (DM overdensity = tracer underdenity or negative DM density)
        if np.any(biases <= 0) or np.any(1 - .7 / biases <= 0):
            return np.full(self.bin_centers.shape, np.inf), np.inf
    
        # this info could be moved elsewhere so its not recalculated every funciton call
        cosm = cbl.Cosmology(cbl.CosmologicalModel__Planck18_)
        fiducial_global_shells = np.diff([cosm.D_C(redshift)**3 for redshift in self.z_edges])
        fiducial_void_shells = [np.power((cosm.D_A(redshift)**2) / cosm.HH(redshift), 1./3) for redshift in self.z_centers]
        
        
        cosm.set_Omega(parameters['om'])
        cosm.set_w0(parameters['w0'])
        cosm.set_wa(parameters['wa'])
        cosm.set_hh(parameters['h'])
        cosm.set_tau(parameters['tau'])
        cosm.set_OmegaB(parameters['omb'])
        cosm.set_n_spec(parameters['ns'])
        cosm.set_scalar_amp(parameters['as'])
        if derive_s8:
            s8 = cosm.sigma8_interpolated(0)
        else:
            s8 = parameters['s8']
            cosm.set_sigma8(s8)
            
        counts = self.bin_counts
        if len(counts.shape) == 1:
            counts =  np.array([counts])
            
        vsf = []
            
        for i, count in enumerate(counts):
            
            bin_centers = self.bin_centers[i]
            volume = self.bin_volumes[i]
            redshift = self.z_centers[i]
            low_z_edge = self.z_edges[i]
            high_z_edge = self.z_edges[i+1]
            bias = biases[i]
                   
            cosm_void_shells = np.power( (cosm.D_A( redshift ) ** 2 ) / cosm.HH( redshift ), 1./3)
            vol_correction = cosm_void_shells / fiducial_void_shells[i]

            rr = bin_centers * vol_correction
            
            r_edges = self.bin_edges[i]
            log_bin_widths = np.diff(np.log(r_edges * vol_correction))
            #note: if this crashes, it may indicate that the user input an integer array rather than a float array for rr or bias
            #TODO: type conversions
            spectrum = cosm.size_function(rr, redshift, "Vdn", bias, 1, 0, -0.7, 1.686,"CAMB",False,"output","Spline",2.)
            
            cosm_global_shell = cosm.D_C(high_z_edge)**3 - cosm.D_C(low_z_edge)**3 
            
            vol_correction = cosm_global_shell / fiducial_global_shells[i]
            
            spectrum *= volume * vol_correction * log_bin_widths
        
            
            vsf.append(spectrum)
        
        vsf = np.array(vsf)
        
        if len(vsf.shape) == 1:
            vsf = np.array([vsf])
            
        return vsf, s8
        
        
    def run_mcmc(self, interpolator_name, runID, Nens = 100, Nburnin = 500, Nsamples = 2000 ):
        
        derive_s8 = self.parameters['s8'].mode in ( 'derived', 'marginalized_derived')
        
        if os.path.isfile(interpolator_name):
            with open (interpolator_name,'rb') as temp_infile:
                if derive_s8:
                    interp, method, interp_s8 = pickle.load(temp_infile)
                    interp = (interp, interp_s8) #Note, setting model to bias and s8 to derived will break the code
                else:
                    interp, method = pickle.load(temp_infile)
        else:
            raise ValueError("The provided interpolator does not exist. Exiting.")
            
        # identify parameters for mcmc
        priors = []
        point_names = []
        
        for key in self.parameters.keys():
            parameter = self.parameters[key]
            if parameter.mode in ('free', 'marginalized', 'marginalized_free'):
                priors.append([parameter.min_value, parameter.max_value])
                point_names.append(param_to_latex(parameter.name, self.model))
            
        priors = np.array(priors)
        
        inisamples = np.array([np.random.uniform(prior[0], prior[1], Nens) for prior in priors]).T # initial samples

        ndims = inisamples.shape[1] # number of parameters/dimensions

        print(ndims, "parameters in mcmc.")

        
        # run mcmc
        if 'bias' in self.parameters.keys():
            
            #perform mcmc
            output = {}
            for i, interp_bin in enumerate(interp):
                
                print(f'Evaluating point {i+1} of {len(interp)}', end='\r')
                sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=(self.bin_counts[i], None, interp_bin, priors, derive_s8),
                                        #pool=pool 
                                       )
                sampler.run_mcmc(inisamples, Nsamples+Nburnin, skip_initial_state_check=True)
                
                
                if derive_s8:
                    output[self.z_centers[i]] = (point_names, sampler.get_chain(), sampler.acceptance_fraction, sampler.get_blobs())
                else:
                    output[self.z_centers[i]] = (point_names, sampler.get_chain(), sampler.acceptance_fraction)
            
            with open (f'mcmc_{method}_{self.model}_{runID}.pickle','wb') as temp_infile:
                pickle.dump(output, temp_infile)
            
            print(f'Completed evaluation of all points {i+1} of {len(interp)}')
            return
        
        
        
        sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args=(self.bin_counts, None, interp, priors, derive_s8), blobs_dtype=[("s_8", float)]
                                    #pool=pool 
                                   )
        sampler.run_mcmc(inisamples, Nsamples+Nburnin, skip_initial_state_check=True)

        

        with open (f'mcmc_{method}_{self.model}_{runID}.pickle','wb') as temp_infile:
            if derive_s8:
                pickle.dump((point_names, sampler.get_chain(), sampler.acceptance_fraction, sampler.get_blobs()), temp_infile)
            else:
                pickle.dump((point_names, sampler.get_chain(), sampler.acceptance_fraction), temp_infile)


            

class VoSS_Parameter ():
    
    def __init__(self, name, fiducial_value, min_value, max_value, num_values=20, mode='fixed'):
        self.name = name
        self.fiducial_value = fiducial_value
        self.min_value = min_value
        self.max_value = max_value
        self.num_values = num_values
        if mode not in ('fixed', 'free', 'derived', 'marginalized_free', 'marginalized_derived'):
            raise ValueError(f"mode parameter must be one of 'fixed', 'free', 'derived', 'marginalized_free', or 'marginalized_derived'. The user input was '{mode}'")
        self.mode = mode
        
    def get_grid(self):
        return np.linspace(self.min_value, self.max_value, self.num_values)
        

def logposterior(theta, data, sigma, interp, priors, derive_s8):
    #theta must be a list or array, not a tuple
    #cast to array for scipy interpolaton functions
    theta = np.array(theta)
    
    lp, s8 = logprior(theta, priors, interp, derive_s8) # get the prior
        
    if not np.isfinite(lp):
        return -np.inf, -1
    
    return lp + loglikelihood(theta, data, sigma, interp, derive_s8), s8


def loglikelihood(theta, data, sigma, interp, derive_s8): #nothing done with sigma atm
    
    if derive_s8:
        vsf = interp[0](*theta)
    else:
        vsf = interp(*theta)

    #the interpolated volume for theta
    # log poisson likelihood
    # N(theta) = number density * fiducial survey volume * volume correction for model
    # Sum_{r,z} ( N(data) * log (N(theta)) - N(theta) - factorial(N(data)) )
    rtrn = np.sum((data*np.log(vsf)) - vsf - loggamma(data+1))
    
    if np.isnan(rtrn):
        return -np.inf
    else:
        return rtrn
    
s8_min =0.5 #TODO: remove hardcoding
s8_max =1.5

def logprior(theta, priors, interp, derive_s8):
    
    lp=0.
    for param, prior in zip(theta, priors):
        #uniform prior
        lp += 0. if prior[0] <= param <= prior[1] else -np.inf
    
    if derive_s8:
        s8 = interp[1](*theta)
        lp += 0. if s8_min <= s8 <= s8_max else -np.inf
    else:
        s8 = 0
   
    return lp, s8

def param_to_latex(param, model):
    
    if model not in ('lambdaCDM','wCDM','w0waCDM', 'bias'):
            raise ValueError(f"Model parameter must be one of 'bias', 'lambdaCDM', 'wCDM', or 'w0waCDM'. The user input was '{model}'")
            
    parameters = {'om':'$\Omega_m$', 'omb':'$\Omega_b$', 'h':'$h$', 
                 'ns':'$n_s$', 'as':'$a_s$', 's8':'$\sigma_8$', 'tau':'$\tau$', 
                 'w0':'$w_0$', 'wa':'$w_a$',
                 'bias':'F', 'slope':'B_{\\text{slope}}', 'intercept':'B_{\\text{intercept}}'}
    
    if model == 'wCDM':
        parameters['w0'] = '$w$'
        
    return parameters[param]



#TODO: remove bin_edges/bin_width from input or make their use consistent

def get_linear_bins(r_low, r_high, r_bin_num):
    #half bin width
    hbw =(r_high-r_low)/(2*r_bin_num)
    #bin centers and widths
    bin_centers= np.linspace(r_low,r_high,r_bin_num+1)[:-1]+hbw
    bin_widths = hbw * np.ones_like(bin_centers)
    return bin_centers, bin_widths

def get_log_bins(r_low, r_high, r_bin_num):
    bin_edges = np.logspace(np.log10(r_low), np.log10(r_high), r_bin_num)
    return get_custom_bins(bin_edges)

def get_custom_bins(bin_edges):
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths/2
    return bin_centers, bin_edges

class VoidSizeFunction():
    def __init__(self):
        pass

    def _format_types(self, bin_centers, bin_edges, z_centers, z_edges, f_bias):
        if not isinstance(z_centers, np.ndarray):
            z_centers = np.array(z_centers)
        if not isinstance(z_edges, np.ndarray):
            z_edges = np.array(z_edges)
        # sort z_centers in descending order for CAMB compatability
        sort_order = z_centers.argsort()[::-1]
        z_centers = z_centers[sort_order]
        z_idx = np.arange(len(z_centers)) #location of user inputted redshifts in z_centers
        #sort z_edges in decreasing order
        z_edges = np.sort(z_edges)[::-1]
        #ensure that 0 is in z_centers for CAMB sigma8 calculation
        if not np.isin(0, z_centers):
            z_centers = np.concatenate((z_centers, [0]))
            
        if not isinstance(f_bias, np.ndarray):
            f_bias = np.array(f_bias)
        f_bias = f_bias[sort_order]
        
        if not isinstance(bin_centers, np.ndarray):
            bin_centers = np.array(bin_centers)
        #ensure one copy on bin_centers for every redshift bin
        if len(bin_centers.shape) == 1:
            bin_centers = np.repeat([bin_centers], len(z_centers[z_idx]), axis=0)
        else:
            #sort bin centers by decreasing redshift (only relevant if bin centers are non-identical)
            bin_centers = bin_centers[sort_order]
        #ensure one copy on bin_edges for every redshift bin   
        if len(bin_edges.shape) == 1:
                bin_edges = np.repeat([bin_edges], len(z_centers[z_idx]), axis=0)
        else:
            #sort bin edges by decreasing redshift (only relevant if bin edges are non-identical)
            bin_edges = bin_edges[sort_order]
        
        #sorting order for returning the user's input
        restore_sort_order = np.argsort(sort_order)
            
        return bin_centers, bin_edges, z_centers, z_idx, z_edges, f_bias, restore_sort_order
    
    def _set_fid_results(self, z_centers, static_z, kmax):
        
        #get results for calculating sigma_R
        
        if static_z >= 0:
            redshifts = [static_z] if static_z == 0 else [static_z, 0]
            self.fid_cosm.set_matter_power(redshifts=redshifts, kmax=kmax)
            
        else:
            self.fid_cosm.set_matter_power(redshifts=z_centers, kmax=kmax)
        
        self.fid_results = camb.get_results(self.fid_cosm)
    
    """#cumulative size spectrum
    def cumu_spectrum(self, shell_radii, z_centers, z_edges, #shell radii and redshift sampling
                 Om, w0, wa, h, tau, Omb, ns, As, #LCDM parameters
                 fiducial_linear_bias, B_slope, B_offset, #bias parameters
                 delta_tr_v = -0.7, delta_c = 1.686, #VSF parameters
                 static_z = -1, #static redshift for cutsky simulations (only used if >= 0)
                 vol_norm = 1, kmax = 2, scale_to_bin = False,#normalization and power spectrum options
                 bin_edges=None, #shell radii bin edges
                 w=1e-6, get_sigma_8 = True, sigma_8 = None
                ):
        
        sig8, spectrum, c_corr = self.spectrum(bin_centers, z_centers, 
                     Om, w0, wa, h, tau, Omb, ns, As,
                     f_bias, delta_tr_v, delta_c, static_z, kmax, w, get_sigma_8, sigma_8,
                     ...remaining params)

        return sig8, np.cumsum( spectrum [:,::-1], axis = 1 ) [:,::-1], c_corr
    """
    
    #size spectrum
    def spectrum(self, shell_radii, z_centers, z_edges, #shell radii and redshift sampling
                 Om, w0, wa, h, tau, Omb, ns, As, #LCDM parameters
                 fiducial_linear_bias, B_slope, B_offset, #bias parameters
                 delta_tr_v = -0.7, delta_c = 1.686, #VSF parameters
                 static_z = -1, #static redshift for cutsky simulations (only used if >= 0)
                 vol_norm = 1, kmax = 2, scale_to_bin = False,#normalization and power spectrum options
                 bin_edges=None, #shell radii bin edges
                 w=1e-6, get_sigma_8 = True, sigma_8 = None
                ):
        shell_radii, bin_edges, z_centers, z_idx, z_edges, fiducial_linear_bias, restore_sort_order = self._format_types(shell_radii, bin_edges, z_centers, z_edges, fiducial_linear_bias)
        if not hasattr(self, 'fid_cosm'):
            raise AttributeError("Fiducial cosmology must be set with set_fid_cosmology before caclulating VSF!")
        
        #get results for calculating sigma_R
        
        self._set_fid_results(z_centers, static_z, kmax)
        
        self.set_cosmology( Om, w0, wa, h, tau, Omb, ns, As)
        
        
        #get results for calculating sigma_R
        if static_z >= 0:
            redshifts = [static_z] if static_z == 0 else [static_z, 0]
            self.cosm.set_matter_power(redshifts=redshifts, kmax=kmax)
        else:
            self.cosm.set_matter_power(redshifts=z_centers, kmax=kmax)
        results = camb.get_results(self.cosm)
        self.results = results

        #calculate bias
        """_, _, pk_fid = self.fid_results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e-2, npoints = 50)
        _, _, pk_cosm = self.results.get_matter_power_spectrum(minkh=1e-4, maxkh=1e-2, npoints = 50)
        cosmo_linear_bias = fiducial_linear_bias * np.mean((pk_fid/pk_cosm)**(.5),axis=1)[z_idx]"""
        #cosmo_linear_bias = fiducial_linear_bias #temporary simplification for testing
        #f_bias = B_slope * cosmo_linear_bias + B_offset
        f_bias = fiducial_linear_bias
        delta_NL_v = delta_tr_v / np.expand_dims(f_bias, 1)
        # catch unphysical bias (DM overdensity = tracer underdenity or negative DM density)
        if np.any(f_bias <= 0) or np.any(1 + delta_NL_v <= 0):
            #sigma_8, void size function, volume correction
            return np.inf, np.full(shell_radii.shape, np.inf), np.full(z_edges[:-1].shape, np.inf)
        
        #account for change in survey volume
        ang = results.angular_diameter_distance(z_centers[z_idx])
        hz = results.h_of_z(z_centers[z_idx])
        fid_ang = self.fid_results.angular_diameter_distance(z_centers[z_idx])
        fid_hz = self.fid_results.h_of_z(z_centers[z_idx])
        
        #Alcock Paczynski effect on void radii
        ap_term = np.expand_dims( (np.power(fid_hz/hz, 1./3)) * np.power(ang/fid_ang, 2./3) , axis=1)
        #account for division by 0 errors
        zero = np.where (z_centers[z_idx]==0)
        ap_term[zero] = 1
        
        #void size function
        dN = dndln(shell_radii * ap_term, w,  delta_NL_v, results, static_z, z_idx, delta_c, sigma_8)
        if scale_to_bin: 
            dN *= np.diff(np.log(bin_edges * ap_term))
        
        #get sigma 8
        sig8 = results.get_sigma8_0() if get_sigma_8 else None

        #normalize void size function
        spectrum =  vol_norm * dN
        
        #calculate volume correction
        fid_comov_h  = self.fid_results.comoving_radial_distance(z_edges[:-1])
        comov_h = self.results.comoving_radial_distance(z_edges[:-1])
        fid_comov_l  = self.fid_results.comoving_radial_distance(z_edges[1:])
        comov_l = self.results.comoving_radial_distance(z_edges[1:])
        c_correction = (comov_h ** 3 - comov_l ** 3) / (fid_comov_h ** 3 - fid_comov_l ** 3)
        
        
        return sig8, spectrum[restore_sort_order], c_correction[restore_sort_order]
        
        
    def set_fid_cosmology(self, Om, w0, wa, h, tau, Omb, ns, As):
        
        self.fid_cosm = self._set_cosmology(Om, w0, wa, h, tau, Omb, ns, As)
        
    def set_cosmology(self, Om, w0, wa, h, tau, Omb, ns, As):
        
        self.cosm = self._set_cosmology(Om, w0, wa, h, tau, Omb, ns, As)
        
        
    def _set_cosmology(self, Om, w0, wa, h, tau, Omb, ns, As):
        
        cosm = camb.CAMBparams();
        # mnu and nnu are fixed for now. Consider making them input parameters
        cosm.set_cosmology(H0=h*100, mnu=0.06, nnu = 3.046, tau = tau) 
        
        # self.cosm.omnuh2 is derived from mnu and nnu inputs
        cosm.omch2 = (Om - Omb) * h * h - cosm.omnuh2
        cosm.ombh2 = Omb * h * h
        
        
        cosm.InitPower.set_params(ns = ns, As = As)
        # Dark energy model
        cosm.set_dark_energy(w=w0,wa=wa, dark_energy_model='ppf')
        #allow P(k) calculation
        cosm.WantTransfer = True
        #cosm.set_accuracy(AccuracyBoost=2.0)
               
        return cosm
        
        


        


# functions that go into the theoretical void size spectrum

#complete void size spectrum for corrected shell radii bin inputs
def dndln(r,w,dnlv,results, static_z, z_idx, delta_c, sigma_8):
    # r (float array): shell radii array
    # w (float): width of interval for slope caclulation
    # dnlv (float): linear thereshold of void formation
    # Plin (nbodykit power.linear object): the power spectrum
    
    #the linear void underdensity threshold
    dlv = 1.594*(1-((1+dnlv)**(-1./1.594)))
    #print("del_v",dlv)
    #conversion from nonlinear to linear shell radius
    r_rL = rL(r,dnlv) 
    
    rescale = 1
    if sigma_8 is not None:
        sig8 = results.get_sigma8_0()
        rescale = sigma_8 / sig8
    #rms fluctuations at each shell radius in each redshift bin (2D array)
    if static_z >= 0:
        sig_rL = np.vstack([rescale*results.get_sigmaR(r_i, z_indices=0) for r_i in r_rL])
    else:
        sig_rL = np.vstack([rescale*results.get_sigmaR(r_i, z_indices=i) for i,r_i in zip(z_idx, r_rL)])
    """print("fact",)
    print("RL",r_rL)
    print("sigmaR", )
    print("sigmaRz", sig_rL)
    print("SSSR",)
    print("Dln_SigmaR", dlns(r_rL, w, results,static_z, z_idx))"""
    #With our curent mapping r -> rL, the last term just evaluates to one and is commented out
    vsf = f(dlv, sig_rL, delta_c) / V(r) * dlns(r_rL, w, results,static_z, z_idx) # * dlnr(r, w, dnlv)
    #print("result",vsf)
    return vsf

#Multiplicity function
def f(dlv, sig_r, delta_c):
    # The multiplicity function of the void size spectrum
    # dlv (float): linear thereshold of void formation
    # sig_r (float array): the sigma_r values for the shell radii 
    
    abs_dlv = np.abs(dlv)
    
    # The D = |delta_v^L| / (delta_c^L + |delta_v^L|) aka "void-and-cloud" term in the void size spectrum
    D = abs_dlv/(delta_c+abs_dlv)
    
    # The x = D * sigma_r / delta_v^L term in the theoretical void size spectrum
    xx = D*sig_r/abs_dlv
        
    dlv = dlv.flatten()
    abs_dlv = abs_dlv.flatten()
    
    S =np.zeros_like(xx)
    for redshift_idx, _ in enumerate(S):
        for i,_ in enumerate(S[redshift_idx]):
            
            if xx[redshift_idx,i]<= 0.276:
                exp_term = np.exp(-dlv[redshift_idx]**2/(2*sig_r[redshift_idx,i]**2))
                S[redshift_idx,i] = np.sqrt(2/np.pi) * abs_dlv[redshift_idx] / sig_r[redshift_idx,i] * exp_term
            else:
                j = np.arange(4)+1
                exp_term = np.exp(-(j*np.pi*xx[redshift_idx,i])**2/2)
                sin_term = np.sin(j*np.pi*D[redshift_idx,0])
                S[redshift_idx,i] = 2*np.sum(exp_term*j*np.pi*xx[redshift_idx,i]**2 * sin_term)
            """j = 1
            sdd = np.inf
            while sdd>0.:
                sdd = np.exp(-.5 * (j*np.pi*xx[redshift_idx,i])**2.) * j * (xx[redshift_idx,i]**2)
                sd = sdd*np.sin(j*np.pi*D[redshift_idx,0])
                S[redshift_idx,i] += sd
                j += 1
    S = 2 * np.pi * S"""
    return S

# d ln sigma^-1 / d ln r_L
def dlns(r,w,results,static_z, z_idx):
    # The d ln sigma^-1 / d ln r_L term in the void size spectrum
    # rL gets passed in
    # returns change in log(sigma_rL) divided by change in log(rL)
    
    # range of rL of width d ln rL = w
    rl = np.exp(np.log(r)-w/2.)
    rh = np.exp(np.log(r)+w/2.)
    
    # corresponding range of sigma
    if static_z >= 0:
        s2l = np.vstack([results.get_sigmaR(r_i, z_indices=0) for r_i in rl])
        s2h = np.vstack([results.get_sigmaR(r_i, z_indices=0) for r_i in rh])
    else:
        s2l = np.vstack([results.get_sigmaR(r_i, z_indices=i) for i,r_i in zip(z_idx, rl)])
        s2h = np.vstack([results.get_sigmaR(r_i, z_indices=i) for i,r_i in zip(z_idx, rh)])
    
    # change in ln sigma^-1 over change in ln rL (recall d ln rL = w)
    # the -1 is factored out of the log 
    return (np.log(s2h)-np.log(s2l))/(-1.*w)

#conversion from nonlinear to linear shell radius
def rL(r,dnlv):
    return r*((1+dnlv)**(1./3))

#d ln rL / d ln r
def dlnr(r, w, dnlv):
    # The d ln rL / d ln r term in the void size spectrum
    # r gets passed in
    # returns change in log(rL) divided by change in log(r)
    
    # range of r of width d ln r = w
    rl = np.exp(np.log(r)-w/2.)
    rh = np.exp(np.log(r)+w/2.)
    
    #corresponding range of rL
    r2l = rL(rl,dnlv)
    r2h = rL(rh,dnlv)
    
    # change in ln rL over change in ln r (w could've been used here as in dlns)
    return (np.log(r2h)-np.log(r2l))/w

#Volume of sphere with radius r
def V(r):
    return 4*np.pi*(r**3)/3.

