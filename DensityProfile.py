import numpy as np
from astropy.table import Table, vstack #can remove these imports
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import ShellVolumeMaskedPython as svm

"""
def profile(voidfile, galfile, survey_density, nbins = 30, extent = 120, 
            norm_extent = 3,r_col_name="radius",prof=True,prof_norm=True,
            mask=None, mask_resolution=None, dist_limits=None,xyz_limits=None,
            mask_type="ra_dec_z"):
    
    #voidfile = Table.read("iron_NGC_maximals.txt",format='ascii.commented_header')
    #galfile = vstack([Table.read("iron_NGC_wall_gal_file.txt",format='ascii.commented_header'),
    #                 Table.read("iron_NGC_field_gal_file.txt",format='ascii.commented_header')])

    assert prof or prof_norm
    vx = voidfile['x']
    vy = voidfile['y']
    vz = voidfile['z']
    vc = np.array([vx,vy,vz,vz,vz]).T
    vr = voidfile[r_col_name]

    # convert from sky coordinates if needed
    gx = galfile['x']
    gy = galfile['y']
    gz = galfile['z']
    gc = np.array([gx,gy,gz]).T
    kdt = cKDTree(gc)

    # shell volume normalization
    norm = np.linspace(0,1,nbins+1) #bins of radius (scaled from 0 to 1)
    norm = 4/3*np.pi*(norm[1:]**3 - norm[:-1]**3) #differences in shell volume (scaled form 0 to 1)
    
    if prof:
        dens_prof = np.zeros(nbins, dtype=float)
        n_temp = 0
        en_temp = 0
    if prof_norm:
        dens_prof_norm = np.zeros(nbins, dtype=float)
        n_temp_norm = 0
        en_temp_norm = 0
    
    for i in range(nbins):
        if prof:
            # set bin equal to number of galaxies in spherical shell
            q = kdt.query_ball_point(vc[:,:3], extent*(i+1)/nbins)
            vc[:,3] = extent*(i+1)/nbins
            vc[:,4] = extent*(i)/nbins
            vol_frac = svm.shell_fraction(
               vc,
               mask=mask, 
               mask_resolution=mask_resolution,
               dist_limits=dist_limits,
               xyz_limits=xyz_limits,
               pts_per_unit_volume=0.1,
               mask_type=mask_type
               )
            en_new = np.array([len(s) for s in q])
            num = en_new - en_temp
            den = np.average(num/vol_frac)
            dens_prof[i] = den
            en_temp = en_new
        if prof_norm:
            # again for normalized profile
            q = kdt.query_ball_point(vc[:,:3],norm_extent*vr*(i+1)/nbins)
            vc[:,3] = norm_extent*vr*(i+1)/nbins
            vc[:,4] = norm_extent*vr*(i)/nbins
            #make sure that pts_per_unit_volume ishigh enough that there are no divide by 0 errors
            # (0.1 has worked for past applications)
            vol_frac = svm.shell_fraction(
               vc,
               mask=mask, 
               mask_resolution=mask_resolution,
               dist_limits=dist_limits,
               xyz_limits=xyz_limits,
               pts_per_unit_volume=0.1,
               mask_type=mask_type
               )
            #print("Vol frac range:")
            #print(np.min(vol_frac), np.max(vol_frac))
            en_new = np.array([len(s) for s in q])
            num = en_new - en_temp_norm
            #multiply vr by vol_frac once other bus are worked out
            den = np.average(num/(vol_frac*vr**3))
            #n_new = np.average([len(s)/(r**3) for s,r in zip(q,vr)])
            dens_prof_norm[i] = den
            en_temp_norm = en_new
            
    
    # normalize profiles by shell volume and against background
    result=[]
    if prof:
        print("Density results:")
        print(survey_density, dens_prof[-1],norm[-1]*extent**3)
        dens_prof = dens_prof/(norm*extent**3) #shell volume (n = dens_prof/volume f shell)
        #dens_prof = dens_prof/dens_prof[-1] #background (delta + 1 = n / n_bar)
        #(why not use survey galaxy density?)
        dens_prof = dens_prof/survey_density
        result.append(dens_prof)
    if prof_norm:
        print(survey_density, dens_prof_norm[-1])
        dens_prof_norm = dens_prof_norm/(norm_extent**3 * norm) 
        #dens_prof_norm = dens_prof_norm/dens_prof_norm[-1]
        dens_prof_norm = dens_prof_norm/survey_density
        result.append(dens_prof_norm)
    if len(result) == 1:
        return result[0]
    return result

"""
    

def profile(voidfile, galfile, survey_density, nbins = 30, extent = 120, 
            norm_extent = 3,r_col_name="radius",prof=True,prof_norm=True,
            mask=None, mask_resolution=None, dist_limits=None,xyz_limits=None,
            mask_type="ra_dec_z", workers = 1):
    """
    voidfile = Table.read("iron_NGC_maximals.txt",format='ascii.commented_header')
    galfile = vstack([Table.read("iron_NGC_wall_gal_file.txt",format='ascii.commented_header'),
                     Table.read("iron_NGC_field_gal_file.txt",format='ascii.commented_header')])
    """
    assert prof or prof_norm
    vx = voidfile['x']
    vy = voidfile['y']
    vz = voidfile['z']
    vc = np.array([vx,vy,vz,vz,vz]).T
    vr = voidfile[r_col_name]

    # convert from sky coordinates if needed
    gx = galfile['x']
    gy = galfile['y']
    gz = galfile['z']
    gc = np.array([gx,gy,gz]).T
    kdt = cKDTree(gc)

    if prof:
        dens_prof = np.zeros(nbins, dtype=float)
        n_temp = 0
        num_in_prev_sphere = 0
    if prof_norm:
        dens_prof_norm = np.zeros(nbins, dtype=float)
        n_temp_norm = 0
        num_in_prev_sphere_norm = 0
    
    for i in range(nbins):
        if prof:
            # set bin equal to number of galaxies in spherical shell
            r_out = extent*(i+1)/nbins
            r_in = extent*(i)/nbins
            q = kdt.query_ball_point(vc[:,:3], r_out, workers = workers)
            vc[:,3] = r_out
            vc[:,4] = r_in
            """vol_frac = svm.shell_fraction(
               vc,
               mask=mask, 
               mask_resolution=mask_resolution,
               dist_limits=dist_limits,
               xyz_limits=xyz_limits,
               pts_per_unit_volume=.1,
               mask_type=mask_type
               )"""
            vol_frac = 1
            num_in_sphere = np.array([len(s) for s in q])
            num = num_in_sphere - num_in_prev_sphere
            vol = vol_frac * 4/3 * np.pi * (r_out**3 - r_in**3)
            den = np.average(num/vol) / survey_density
            dens_prof[i] = den
            num_in_prev_sphere = num_in_sphere
        if prof_norm:
            # again for normalized profile
            r_out = norm_extent*vr*(i+1)/nbins
            r_in = norm_extent*vr*(i)/nbins
            q = kdt.query_ball_point(vc[:,:3],r_out, workers = workers)
            vc[:,3] = r_out
            vc[:,4] = r_in
            #make sure that pts_per_unit_volume ishigh enough that there are no divide by 0 errors
            # (0.1 has worked for past applications)
            """vol_frac = svm.shell_fraction(
               vc,
               mask=mask, 
               mask_resolution=mask_resolution,
               dist_limits=dist_limits,
               xyz_limits=xyz_limits,
               pts_per_unit_volume=0.1,
               mask_type=mask_type
               )"""
            vol_frac = 1
            num_in_sphere = np.array([len(s) for s in q])
            num = num_in_sphere - num_in_prev_sphere_norm
            #multiply vr by vol_frac once other bus are worked out
            vol = vol_frac * 4/3 * np.pi * (r_out**3 - r_in**3)
            den = np.average(num/vol) / survey_density
            dens_prof_norm[i] = den
            num_in_prev_sphere_norm = num_in_sphere
            
    
    # normalize profiles by shell volume and against background
    result=[]
    if prof:
        result.append(dens_prof)
    if prof_norm:
        result.append(dens_prof_norm)
    if len(result) == 1:
        return result[0]
    return result


