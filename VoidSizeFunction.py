import numpy as np
import camb
from camb import model, initialpower, CAMBError
from astropy.cosmology import w0waCDM, FlatLambdaCDM

# Fiducial Cosmology
# -----------------
# hubble constant
h = 0.6736
# matter density
omega_M = 0.3153
# CMB optical depth
tau = 0.0544
# Baryon desity * h^2
omega_Bh2 =  0.0221 #0.02237 #
# power spectrum tilt
n_s = .9649 # same
# scalar amplitude
a_s =  2.1e-9 #2.0830e-9 #

fid_w0, fid_wa = 0, -1

#CAMB Comology corresponding to Abacus c000 (detailed at https://abacussummit.readthedocs.io/en/latest/cosmologies.html)
Kos0 = camb.CAMBparams();
#TODO: see if there is a way to load in lank as a default
Kos0.set_cosmology(H0=h*100, ombh2=omega_Bh2, omch2=0.1200, mnu=0.06, nnu = 3.046, tau = tau)
Kos0.InitPower.set_params(ns = n_s, As = a_s)
#Kos0.set_matter_power(redshifts=[0], kmax=2.0)
#results = camb.get_results(Kos0)

Kos2 = w0waCDM(H0=100*h, Om0=omega_M, Ode0=1-omega_M, w0=-1., wa=0., Neff = 3.046)


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

# void size spectrum function
def cumu_theory_size_spectrum( delta,Om_m,w0,wa,w,z,f_bias, #VSF theory parameters
                              delta_c = 1.686, #Default VSF theory parameters
                              bin_centers=None, bin_edges=None, #bin centers and bin edges
                              static_z = -1, #static redshift for cutsky simulations (only used if >= 0)
                              vol_norm = 1, kmax=1.2, scale_to_bin = False): #normalization and power spectrum options
    s8, dN = theory_size_spectrum_NC(delta,Om_m,w0,wa,w,z,f_bias,delta_c, vol_norm = vol_norm, kmax = kmax)
    return s8, np.cumsum( dN [:,::-1], axis = 1 ) [:,::-1]

def theory_size_spectrum( delta,Om_m,w0,wa,w,z,f_bias, #VSF theory parameters
                              delta_c = 1.686, #Default VSF theory parameters
                              bin_centers=None, bin_edges=None, #bin centers and bin edges
                              static_z = -1, #static redshift for cutsky simulations (only used if >= 0)
                              vol_norm = 1, kmax=1.2, scale_to_bin = False): #normalization and power spectrum options
    if not isinstance(z, np.ndarray):
        z = np.array(z)
    if not isinstance(f_bias, np.ndarray):
        f_bias = np.array(f_bias)
    dnlv = delta/np.expand_dims(f_bias,1)
    #create cosmology
    Kos = camb.CAMBparams();
    #set Om and DEoS in cosmology
    Kos.set_cosmology(H0=h*100, mnu=0.06, nnu = 3.046, tau = tau) 
    Kos.omch2 = Kos0.omch2*Om_m/Kos0.omegam
    Kos.ombh2 = Kos0.ombh2*Om_m/Kos0.omegam
    Kos.omnuh2 = Kos0.omnuh2*Om_m/Kos0.omegam
        
    Kos.InitPower.set_params(ns = n_s, As = a_s)
    #Dark energy model
    Kos.set_dark_energy(w=w0,wa=wa, dark_energy_model='ppf')
    #get results for calculating sigma_R
    if static_z >= 0:
        Kos.set_matter_power(redshifts=static_z, kmax=kmax)
    else:
        Kos.set_matter_power(redshifts=z, kmax=kmax)
    results = camb.get_results(Kos)
    # fiducial cosmology
    h2   = Kos2.H(z).value
    a2   = Kos2.angular_diameter_distance(z).value
    # same as asumed cosmology Kos but with astropy
    Kos3 = w0waCDM(H0=100*h,Om0=Om_m,Ode0=1.-Om_m,w0=w0,wa=wa, Neff = 3.046)
    h3   = Kos3.H(z).value
    a3   = Kos3.angular_diameter_distance(z).value
    #accounts for change in survey volume
    agr = np.expand_dims( ((h2/h3)**(1./3))*((a3/a2)**(2./3)) , axis=1)
    #account for division by 0 errors
    zero = np.where (z==0)
    agr[zero] = 1
    #non-cumulative vss
    radii = np.repeat(bin_centers, len(z), axis=0)
    dN = dndln((radii)/agr,w,dnlv,results, static_z, delta_c)
    if scale_to_bin: dN *= np.diff(np.log(bin_edges))
    #get sigma 8
    Kos.set_matter_power(redshifts=[0], kmax=kmax)
    results = camb.get_results(Kos)
    
    return results.get_sigma8_0(), vol_norm * dN


# functions that go into the theoretical void size spectrum

#complete void size spectrum for corrected shell radii bin inputs
def dndln(r,w,dnlv,results, static_z, delta_c):
    # r (float array): shell radii array
    # w (float): width of interval for slope caclulation
    # dnlv (float): linear thereshold of void formation
    # Plin (nbodykit power.linear object): the power spectrum
    
    #the linear void underdensity threshold
    dlv = 1.594*(1-((1+dnlv)**(-1./1.594)))
    #conversion from nonlinear to linear shell radius
    r_rL = rL(r,dnlv)  
    #rms fluctuations at each shell radius in each redshift bin (2D array)
    if static_z >= 0:
        sig_rL = np.vstack([results.get_sigmaR(r_i, z_indices=0) for  i,r_i in enumerate(r_rL)])
    else:
        sig_rL = np.vstack([results.get_sigmaR(r_i, z_indices=i) for  i,r_i in enumerate(r_rL)])
    #With our curent mapping r -> rL, the last term just evaluates to one and is commented out
    return f(dlv, sig_rL, delta_c) / V(r) * dlns(r_rL, w, results,static_z) # * dlnr(r, w, dnlv)

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
    
    S =np.zeros_like(xx)
    for redshift_idx, _ in enumerate(S):
        for i,_ in enumerate(S[redshift_idx]):
            j = 1
            sdd = np.inf
            while sdd>0.:
                sdd = np.exp(-.5 * (j*np.pi*xx[redshift_idx,i])**2.) * j * (xx[redshift_idx,i]**2)
                sd = sdd*np.sin(j*np.pi*D[redshift_idx,0])
                S[redshift_idx,i] += sd
                j += 1
    S = 2 * np.pi * S
    return S

# d ln sigma^-1 / d ln r_L
def dlns(r,w,results,static_z):
    # The d ln sigma^-1 / d ln r_L term in the void size spectrum
    # rL gets passed in
    # returns change in log(sigma_rL) divided by change in log(rL)
    
    # range of rL of width d ln rL = w
    rl = np.exp(np.log(r)-w/2.)
    rh = np.exp(np.log(r)+w/2.)
    
    # corresponding range of sigma
    if static_z >= 0:
        s2l = np.vstack([results.get_sigmaR(r_i, z_indices=0) for  i,r_i in enumerate(rl)])
        s2h = np.vstack([results.get_sigmaR(r_i, z_indices=0) for  i,r_i in enumerate(rh)])
    else:
        s2l = np.vstack([results.get_sigmaR(r_i, z_indices=i) for  i,r_i in enumerate(rl)])
        s2h = np.vstack([results.get_sigmaR(r_i, z_indices=i) for  i,r_i in enumerate(rh)])
    
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

