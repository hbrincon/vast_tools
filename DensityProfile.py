import numpy as np
from astropy.table import Table, vstack #can remove these imports
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import ShellVolumeMaskedPython as svm

  

def profile(voids, 
            galaxies, 
            survey_density, 
            nbins = 30, 
            extent = 120, 
            norm_extent = 3, 
            radius_column_name="radius", 
            prof=True, 
            prof_norm=True,
            mask=None, 
            mask_resolution=None, 
            dist_limits=None, 
            xyz_limits=None, 
            mask_type="ra_dec_z", 
            workers = 1
           ):
    """
    Makes a stacked void density profile for voids found in galaxy-redshift surveys or cubic box
    simulations. Currently does not account for periodic boundary conditions.
    
    params:
    ---------------------------------------------------------------------------------------------
    voids (astropy table): A table containing the void information. Required coluns include the 
        cartesian coordinates of the void centers ('x', 'y', 'z') and the void radii (see 
        radius_column_name).
        
    galaxies (astropy table): A table containing the galaxy information. Required coluns include 
        the cartesian coordinates of the galaxies ('x', 'y', 'z').
        
    survey_density (float or numpy array of floats): The average tracer density of the 
        galaxy-reshift survey. The void density profile is overdense when the density exceeds the 
        survey_density value. If a numpy array of len(voids) is provided, each array value is 
        matched to its corresponding void, allowing for a tracer density that evolves with
        redshift, n(z).
        
    nbins (int): The number of bins used to histogram the void dnesity profile
    
    extent (float): The maximum radius that the void density profile extends to in Mpc/h
    
    norm_extent (float): The maximum radius that the normalized void density profile extends to 
        in units of R_eff, where R_eff is the void radius provided by voids[radius_column_name]. See
        prof_norm for furhter informationa about the normalized void density profile.
        
    radius_column_name (string): The name of the column in voids that stroes the void radii. Defaults to
        "radius".
    
    prof (bool): Determines if a stacked density profile is returned where the void radii have a
        common scaling in units of Mpc/h. Defaults to True
        
    prof_norm (bool): Determines if a normalized stacked density profile is returned where the
        void radii are rescaled to units of R_eff, where R_eff is the radius of each void 
        provided by voids[radius_column_name]. When the stacked density profile is normalized, the
        characterisic transition from low density to high density at the void edges occur at 
        the same radius, creating a sharper bucket shape in the void density profile. Defaults 
        to True.
        
    mask, mask_resolution, dist_limits, xyz_limits, mask_type: The survey mask information. 
        Currently unused in the code.
        
    workers (int): The number of parallel processes to use when calculating the void density
        profile. Defaults to 1.
        
        
    returns:
    ---------------------------------------------------------------------------------------------
    dens_prof (numpy array of floats): The stacked void density profile in units of Mpc/h
    
    dens_prof_norm (numpy array of floats): The stacked void density profile normalized in units 
        of the effective void radii.
    """
    
    # Ensure that a least one profile is being returned
    assert prof or prof_norm
    
    # Collect void information (last two columns will be used for void radii bins)
    void_centers = np.array([voids['x'],  voids['y'], voids['z'], np.zeros_like(voids['z']), np.zeros_like(voids['z'])]).T 
    
    void_radii = voids[radius_column_name]

    # Collect galaxy information and make a KDTree
    galaxy_centers = np.array([galaxies['x'], galaxies['y'], galaxies['z']]).T
    kdt = cKDTree(galaxy_centers)

    # Set up arrays for the void density profiles
    if prof:
        dens_prof = np.zeros(nbins, dtype=float)
        num_in_prev_sphere = 0 # number of galaxies in privious radial bin
    if prof_norm:
        dens_prof_norm = np.zeros(nbins, dtype=float)
        num_in_prev_sphere_norm = 0 # number of galaxies in privious radial bin
    
    # Fill in each bin of the void density profile
    for i in range(nbins):
        if prof:
            # The edges of the current radial bin
            r_out = extent*(i+1)/nbins
            r_in = extent*(i)/nbins
            
            # The number of points that fall within r_out of the void centers
            num_in_sphere = kdt.query_ball_point(void_centers[:,:3], r_out, return_length = True, workers = workers)
            
            #Unused mask code
            """
            # Add the bin edges to the void information
            void_centers[:,3] = r_out
            void_centers[:,4] = r_in
            
            # Calculate the fraction of the voids located within the mask
            vol_frac = svm.shell_fraction(
               void_centers,
               mask=mask, 
               mask_resolution=mask_resolution,
               dist_limits=dist_limits,
               xyz_limits=xyz_limits,
               pts_per_unit_volume=.1,
               mask_type=mask_type
               )"""
            vol_frac = 1
            
            
            #num_in_sphere = np.array([len(s) for s in q])
            
            # The number of galaxies in the current radial bin
            num = num_in_sphere - num_in_prev_sphere
            
            # The volume of the voids
            vol = vol_frac * 4/3 * np.pi * (r_out**3 - r_in**3)
            
            # The density of the current radial bin, normalized by the survey density
            den = np.average(num / vol / survey_density)
            
            # Write to the density profile array
            dens_prof[i] = den
            
            # Record the number of galalxies in the previous bin
            num_in_prev_sphere = num_in_sphere
        
        # again for normalized profile
        if prof_norm:
            # The edges of the current radial bin
            r_out = norm_extent*void_radii*(i+1)/nbins
            r_in = norm_extent*void_radii*(i)/nbins
            
            # The number of points that fall within r_out of the void centers
            num_in_sphere = kdt.query_ball_point(void_centers[:,:3], r_out, return_length = True, workers = workers)

            #make sure that pts_per_unit_volume ishigh enough that there are no divide by 0 errors
            # (0.1 has worked for past applications)
            """
            void_centers[:,3] = r_out
            void_centers[:,4] = r_in
            
            vol_frac = svm.shell_fraction(
               void_centers,
               mask=mask, 
               mask_resolution=mask_resolution,
               dist_limits=dist_limits,
               xyz_limits=xyz_limits,
               pts_per_unit_volume=0.1,
               mask_type=mask_type
               )"""
            vol_frac = 1
            #num_in_sphere = np.array([len(s) for s in q])
            
            # The number of galaxies in the current radial bin
            num = num_in_sphere - num_in_prev_sphere_norm
            
            #multiply void_radii by vol_frac once other bus are worked out
            
            # The volume of the voids
            vol = vol_frac * 4/3 * np.pi * (r_out**3 - r_in**3)
            
            # The density of the current radial bin, normalized by the survey density
            den = np.average(num / vol / survey_density)
            
            # Write to the density profile array
            dens_prof_norm[i] = den
            
            # Record the number of galalxies in the previous bin
            num_in_prev_sphere_norm = num_in_sphere
            
    
    # Return the void radii
    
    if prof and prof_norm:
        return dens_prof, dens_prof_norm
    
    elif prof:
        return dens_prof
    
    elif prof_norm:
        return dens_prof_norm


