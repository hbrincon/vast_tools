import numpy as np
import camb
from camb import model, initialpower, CAMBError
from astropy.cosmology import w0waCDM, FlatLambdaCDM

import pickle
import os


"""

# Fiducial Cosmology
# -----------------
# hubble constant
fid_h = 0.6736
# matter density
fid_Om = 0.3153
# CMB optical depth
fid_tau = 0.0544
# Baryon desity * h^2
fid_Obh = 0.0221 #0.02237
# power spectrum tilt
fid_ns = .9649 # same
# scalar amplitude
fid_as = 2.1e-9 #2.0830e-9
# dark energy
fid_w0, fid_wa = 0, -1

#CAMB Comology corresponding to Abacus c000 (detailed at https://abacussummit.readthedocs.io/en/latest/cosmologies.html)
#TODO: see if there is a way to load in lank as a default

Kos0 = camb.CAMBparams();
Kos0.set_cosmology(H0=fid_h*100, ombh2=fid_Obh, omch2=0.1200, mnu=0.06, nnu = 3.046, tau = fid_tau)
Kos0.InitPower.set_params(ns = fid_ns, As = fid_as)
"""


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
    """
    def set_grid(self, Om_range, Om_num, w0_range, w0_num, wa_range, wa_num,
                h_range, h_num, tau_range, tau_num, Obh_range, Obh_num, ns_range, ns_num,
                as_range, as_num):
        self.Om_grid = np.linspace(Om_range[0], Om_range[1], Om_num)
        self.w0_grid = np.linspace(w0_range[0], w0_range[1], w0_num)
        self.wa_grid = np.linspace(wa_range[0], wa_range[1], wa_num)
        self.h_grid = np.linspace(h_range[0], h_range[1], h_num)
        self.tau_grid = np.linspace(tau_range[0], tau_range[1], tau_num)
        self.Obh_grid = np.linspace(Obh_range[0], Obh_range[1], Obh_num)
        self.ns_grid = np.linspace(ns_range[0], ns_range[1], ns_num)
        self.as_grid = np.linspace(as_range[0], as_range[1], as_num)
     """
    def _format_types(self, bin_centers, z_centers, z_edges, f_bias):
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
            f_bias = np.array(f_bias)[sort_order]
        
        if not isinstance(bin_centers, np.ndarray):
            bin_centers = np.array(bin_centers)
        #ensure one copy on bin_cetners for every redshift bin
        if len(bin_centers.shape) == 1:
            bin_centers = np.repeat([bin_centers], len(z_centers[z_idx]), axis=0)
        else:
            #sort bin centers by decreasing redshift (only relevant if bin centers are non-identical)
            bin_centers = bin_centers[sort_order]
        
        #sorting order for returning the user's input
        restore_sort_order = np.argsort(sort_order)
            
        return bin_centers, z_centers, z_idx, z_edges, f_bias, restore_sort_order
    
    def _set_fid_results(self, z_centers, static_z, kmax):
        
        #get results for calculating sigma_R
        
        if static_z >= 0:
            redshifts = [static_z] if static_z == 0 else [static_z, 0]
            self.fid_cosm.set_matter_power(redshifts=redshifts, kmax=kmax)
            
        else:
            self.fid_cosm.set_matter_power(redshifts=z_centers, kmax=kmax)
        
        self.fid_results = camb.get_results(self.fid_cosm)
    
    """#cumulative size spectrum
    def cumu_spectrum(self, bin_centers, z_centers, #shell radii and redshift sampling
                 Om, w0, wa, h, tau, Obh, ns, As, #LCDM parameters
                 f_bias, delta_tr_v = -0.7, delta_c = 1.686, static_z = -1, kmax = 2, #VSF parameters
                w=1e-6, get_sigma_8=True, sigma_8 = None):
        
        sig8, spectrum = self.spectrum(bin_centers, z_centers, 
                     Om, w0, wa, h, tau, Obh, ns, As,
                     f_bias, delta_tr_v, delta_c, static_z, kmax, w, get_sigma_8, sigma_8)

        return sig8, np.cumsum( spectrum [:,::-1], axis = 1 ) [:,::-1]
    """
    
    #size spectrum
    def spectrum(self, bin_centers, z_centers, z_edges, #shell radii and redshift sampling
                 Om, w0, wa, h, tau, Omb, ns, As, #LCDM parameters
                 f_bias, delta_tr_v = -0.7, delta_c = 1.686, static_z = -1, #VSF parameters
                 kmax = 2, scale_to_bin = False,#normalization and power spectrum options
                 bin_edges=None, #shell radii bin edges
                 w=1e-6, get_sigma_8 = True, sigma_8 = None
                ):
        bin_centers, z_centers, z_idx, z_edges, f_bias, restore_sort_order = self._format_types(bin_centers, z_centers, z_edges, f_bias)

        if not hasattr(self, 'fid_cosm'):
            raise AttributeError("Fiducial cosmology must be set with set_fid_cosmology before caclulating VSF!")
        
        #get results for calculating sigma_R
        self._set_fid_results(z_centers, static_z, kmax)
        
        
        self.set_cosmology( Om, w0, wa, h, tau, Omb, ns, As)
        
        #caclulate void size spetrum
        sig8, spectrum = self._spectrum(bin_centers,
              delta_tr_v,
              w,z_centers,f_bias,
              z_idx,
              delta_c, static_z = static_z, 
              kmax = kmax, scale_to_bin = scale_to_bin, bin_edges=bin_edges,
              get_sigma_8 = get_sigma_8, sigma_8 = sigma_8)
        
        #calculate volume correction
        fid_comov_h  = self.fid_results.comoving_radial_distance(z_edges[:-1])
        comov_h = self.results.comoving_radial_distance(z_edges[:-1])
        fid_comov_l  = self.fid_results.comoving_radial_distance(z_edges[1:])
        comov_l = self.results.comoving_radial_distance(z_edges[1:])
        c_correction = (comov_h ** 3 - comov_l ** 3) / (fid_comov_h ** 3 - fid_comov_l ** 3)
        
        
        
        return sig8, spectrum[restore_sort_order], c_correction[restore_sort_order]
        

    """def spectrum_grid(self, name, bin_centers, z_centers, z_edges, f_bias, 
                      delta_tr_v = -0.7, delta_c = 1.686, static_z = -1, kmax = 2, w=1e-6, get_sigma_8=True):
        
        if os.path.isfile(f'{name}_bgrid.npy'):
            raise OSError("VSF grid file already exists. Delete the existing file before re-running.")
        if os.path.isfile(f'{name}_sgrid.npy'):
            raise OSError("Sigma_8 grid file already exists. Delete the existing file before re-running.")
        if os.path.isfile(f'{name}_cgrid.npy'):
            raise OSError("Comoving volume grid file already exists. Delete the existing file before re-running.")
        if os.path.isfile(f'{name}_info.pickle'):
            raise OSError(f"{name} information file already exists. Delete the existing file before re-running.")
            
        bgrid = np.zeros((len(self.Om_grid), len(self.w0_grid), len(self.wa_grid),
                          len(self.h_grid), len(self.tau_grid), len(self.Obh_grid), len(self.ns_grid), 
                          len(self.as_grid),
                          len(z_centers), len(bin_centers) ))
        sgrid = np.zeros((len(self.Om_grid), len(self.w0_grid), len(self.wa_grid),
                          len(self.h_grid), len(self.tau_grid), len(self.Obh_grid), len(self.ns_grid), 
                          len(self.as_grid)))
        cgrid = np.zeros((len(self.Om_grid), len(self.w0_grid), len(self.wa_grid),
                          len(self.h_grid), len(self.tau_grid), len(self.Obh_grid), len(self.ns_grid), 
                          len(self.as_grid), len(z_centers)))
        
        z_centers, z_idx, z_edges, f_bias = self._format_types(z_centers, z_edges, f_bias)
        
        if not hasattr(self, 'fid_cosm'):
            raise AttributeError("Fiducial cosmology must be set with set_fid_cosmology before caclulating VSF!")
            
        #get results for calculating sigma_R
        self._set_fid_results(z_centers, static_z, kmax)
        
        
        radii = np.repeat([bin_centers], len(z_centers[z_idx]), axis=0)
        
        error_count = 0

        for i, Om in enumerate(self.Om_grid):
            for j, w0 in enumerate(self.w0_grid):
                for k, wa in enumerate(self.wa_grid):
                    for l, h in enumerate(self.h_grid):
                        for m, tau in enumerate(self.tau_grid):
                            for n, Obh in enumerate(self.Obh_grid):
                                for o, ns in enumerate(self.ns_grid):
                                    for p, As in enumerate(self.as_grid):
                                        #set cosm
                                        try:
                                            self.set_cosmology( Om, w0, wa, h, tau, Obh, ns, As)
                                        except CAMBError:
                                            bgrid[i][j][k][l][m][n][o][p][:] =  np.inf
                                            sgrid[i][j][k][l][m][n][o][p] = np.inf
                                            cgrid[i][j][k][l][m][n][o][p][:] = np.inf
                                            error_count+=1  
                                            continue
                                            
                                        sig8, spectrum = self._spectrum(radii,
                                              delta_tr_v,
                                              1e-6,z_centers,f_bias,
                                              z_idx,
                                              delta_c, static_z = static_z, kmax = kmax, w=w, get_sigma_8 = get_sigma_8)

                                        bgrid[i][j][k][l][m][n][o][p] = spectrum
                                        sgrid[i][j][k][l][m][n][o][p] = sig8
                                

                                        for z in range(len(z_centers[z_idx])):
                                            fid_comov_h  = self.fid_results.comoving_radial_distance(z_edges[z])
                                            comov_h = self.results.comoving_radial_distance(z_edges[z])
                                            fid_comov_l  = self.fid_results.comoving_radial_distance(z_edges[z+1])
                                            comov_l = self.results.comoving_radial_distance(z_edges[z+1])
                                            cgrid[i][j][k][l][m][n][o][p][z] = (comov_h ** 3 - comov_l ** 3) / (fid_comov_h ** 3 - fid_comov_l ** 3)


                                        
        self.bgrid = bgrid
        self.sgrid = sgrid
        self.error_count = error_count
        
        np.save(f"{name}_bgrid.npy",bgrid)
        np.save(f"{name}_sgrid.npy",sgrid)
        np.save(f"{name}_cgrid.npy", cgrid)
        with open(f'{name}_info.pickle','wb') as fn:
            pickle.dump((bin_centers, 
                         z_centers[z_idx], z_edges, f_bias, 
                         delta_tr_v, delta_c, static_z, kmax,
                         error_count, 
                         self.Om_grid, self.w0_grid, self.wa_grid,
                         self.h_grid, self.tau_grid, self.Obh_grid,
                         self.ns_grid, self.as_grid)
                        , fn)
    """
        
        
        
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
        #allow P(k) calcualtion
        cosm.WantTransfer = True
        #cosm.set_accuracy(AccuracyBoost=2.0)
               
        return cosm
        
        
       
    """def calibrate_bias(self, bin_centers, z_centers, f_range, f_num, 
                       delta_tr_v = -0.7, delta_c = 1.686, static_z = -1, kmax = 2, #VSF parameters
                w=1e-6, get_sigma_8 = True, sigma_8=None):
        
        if os.path.isfile(f'{self.name}_bias_callibration.pickle'):
            raise OSError("Bias file already exists. Delete the existing file before re-running.")
        
        self.f_grid = np.linspace(f_range[0], f_range[1], f_num)
        fgrid = np.zeros((len(self.f_grid), len(z_centers), len(bin_centers) ))
        
        z_centers, z_idx, self.f_grid = self._format_types(z_centers, self.f_grid)

        
        if not hasattr(self, 'fid_cosm'):
            raise AttributeError("Fiducial cosmology must be set with set_fid_cosmology before caclulating VSF!")
        
        #get results for calculating sigma_R
        self._set_fid_results(z_centers, static_z, kmax)
        
        radii = np.repeat([bin_centers], len(z_centers[z_idx]), axis=0)
        
        self.cosm = self.fid_cosm
        
        for i, f in enumerate(self.f_grid):
            try:
                fbias = np.repeat(f, len(z_centers[z_idx]))
                
                sig8, spectrum = self._spectrum(radii,
                      delta_tr_v,
                      1e-6, z_centers, fbias,
                      z_idx,
                      delta_c, static_z = static_z, w=w, get_sigma_8 = get_sigma_8, sigma_8=sigma_8)
                fgrid[i] = spectrum
                
            except CAMBError:
                fgrid[i] = np.full((len(z_centers), len(bin_centers)), np.inf)
        self.fgrid = fgrid
        
        with open(f'{self.name}_bias_callibration.pickle','wb') as fn:
            pickle.dump((bin_centers, z_centers, self.f_grid, self.fgrid), fn)
    """
                
    def _spectrum(self, shell_radii,
                  delta_tr_v,w,z_centers,f_bias, #VSF theory parameters
                  z_idx,
                  delta_c = 1.686, #Default VSF theory parameters
                  static_z = -1, #static redshift for cutsky simulations (only used if >= 0)
                  vol_norm = 1, kmax=2, scale_to_bin = False,#normalization and power spectrum options
                  bin_edges=None, #shell radii bin edges
                  get_sigma_8 = True, sigma_8 = None): 
        

        delta_NL_v = delta_tr_v / np.expand_dims(f_bias, 1)
        
        #get results for calculating sigma_R
        if static_z >= 0:
            redshifts = [static_z] if static_z == 0 else [static_z, 0]
            self.cosm.set_matter_power(redshifts=redshifts, kmax=kmax)
        else:
            self.cosm.set_matter_power(redshifts=z_centers, kmax=kmax)
        results = camb.get_results(self.cosm)
        self.results = results
        
        #account for change in survey volume
        ang = results.angular_diameter_distance(z_centers[z_idx])
        hz = results.h_of_z(z_centers[z_idx])
        fid_ang = self.fid_results.angular_diameter_distance(z_centers[z_idx])
        fid_hz = self.fid_results.h_of_z(z_centers[z_idx])

        agr = np.expand_dims( (np.power(fid_hz/hz, 1./3)) * np.power(ang/fid_ang, 2./3) , axis=1)
        #account for division by 0 errors
        zero = np.where (z_centers[z_idx]==0)
        agr[zero] = 1
        
        #void size function
        dN = dndln(shell_radii * agr, w,  delta_NL_v, results, static_z, z_idx, delta_c, sigma_8)
        if scale_to_bin: 
            if len(bin_edges.shape) == 1:
                bin_edges = np.repeat([bin_edges], len(z_centers[z_idx]), axis=0)
            dN *= np.diff(np.log(bin_edges * agr))
        
        #get sigma 8
        sig8 = results.get_sigma8_0() if get_sigma_8 else None

        return sig8, vol_norm * dN

        


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



#Test contarini functions
