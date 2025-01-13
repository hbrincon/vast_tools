import numpy as np
import os
import mmap
import struct
import socket
import select
import tempfile
import multiprocessing
from psutil import cpu_count
from astropy.table import Table


from multiprocessing import Process, Value
from ctypes import c_int64


from vast.voidfinder.hole_combine import spherical_cap_volume
from ShellVolumeMasked import _check_holes_mask_overlap
from vast.voidfinder._voidfinder import process_message_buffer
from vast.voidfinder._voidfinder_cython_find_next import MaskChecker
from vast.voidfinder.volume_cut import build_unit_sphere_points, generate_mesh

import time

"""
Authors: Hernan Rincon

Some code has been adopted from the following individuals: Steve O'Neil
"""

# This code calculates the percentage of a sphereical shell's volume that is located within 
# a survey mask (such as an angular mask + redshift limits)


def shell_fraction(
    x_y_z_r_r0_array,
    mask_type='ra_dec_z',
    mask=None, 
    mask_resolution=None,
    dist_limits=None,
    xyz_limits=None,
    pts_per_unit_volume=0.01,
    ):
    """
    Calculates the percentage of a spherical shell's volume that is located within a survey 
    or simulation mask. This is done individually for a set of N spherical shells given in the
    input.
    
    params:
    ---------------------------------------------------------------------------------------------
    x_y_z_r_r0_array (numpy array of floats of shape (N, 5)): A set of N spherical shells. Each 
        row contains the following information: (x coordinate, y coordinate, z coordiante, 
        outer shell radius, inner shell radius)
    
    mask_type (string): One of 'ra_dec_z', 'xyz', or 'periodic' and is used to determine the mask
        mode of the MaskChecker object. Defaults to 'ra_dec_z'
        
    mask (array of ints of shape (M*360, M*180)): the angular survey mask used by the MaskChecker 
        object, where values of 0 are exluded from the mask and values of 1 are included. The 
        shape is set by the mask resolution M. Used only when mask_type='ra_dec_z'. Defaults to 
        None.
    
    mask_resolution (int): The mask resolution used by the MaskChecker object, which acts to 
        convert the mask from units of array indices to RA, Dec. Used only when 
        mask_type='ra_dec_z'. Defaults to None
        
    dist_limits (2-element list of floats) The distance limits of the survey. Used only when 
        mask_type='ra_dec_z'. Defaults to None
        
    xyz_limits (numpy array of floats of shape (2, 3)): The xyz limits of the simulation. Used 
        only when mask_type='xyz'. Defaults to None.
    
    pts_per_unit_volume (float): The number of points per unit volume to sample when calculating 
        the shell filling fraction. A larger number of points leads to a more accurate 
        calculation at the cost of increased computation. Defaults to 0.01 (intended for volume 
        units of [Mpc/h]^3)
    
    returns:
    ---------------------------------------------------------------------------------------------
    vol_frac (numpy array of floats of shape N): The shell filling fraction for each of N shells
    """  
    
    # Set the mask mode
    if mask_type == "ra_dec_z":
        mask_mode = 0
    elif mask_type == "xyz":
        mask_mode = 1
    elif mask_type == "periodic":
        mask_mode = 2
    else:
        raise ValueError(f"mask_type must be 'ra_dec_z', 'xyz', or 'periodic.' The provided value was '{mask_type}'")
    
    
    # Set the distance limits
    if dist_limits is None:
        min_dist = None
        max_dist = None
    else:
        min_dist = dist_limits[0]
        max_dist = dist_limits[1]
    
    # Initialize the mask object
    if mask_mode == 0: # sky mask
        mask_checker = MaskChecker(mask_mode,
                                   survey_mask_ra_dec=mask.astype(np.uint8),
                                   n=mask_resolution,
                                   rmin=min_dist,
                                   rmax=max_dist)
        
    elif mask_mode in [1,2]: # Cartesian mask
        mask_checker = MaskChecker(mask_mode,
                                   xyz_limits=xyz_limits)


    # Calculate the volume of the shells within the survey mask    
    vol_frac = oob_cut_single(x_y_z_r_r0_array, 
                   mask_checker,
                   cut_pct=0.1,
                   pts_per_unit_volume=pts_per_unit_volume,
                   num_surf_pts=20)

    return vol_frac

        

def oob_cut_single(x_y_z_r_r0_array, 
                   mask_checker,
                   cut_pct=0.1,
                   pts_per_unit_volume=0.01,
                   num_surf_pts=20):
    """
    Out-Of-Bounds cut single threaded version.
    
    params:
    ---------------------------------------------------------------------------------------------
    x_y_z_r_r0_array (numpy array of floats of shape (N, 5)): A set of N spherical shells. Each 
        row contains the following information: (x coordinate, y coordinate, z coordiante, 
        outer shell radius, inner shell radius)
    
    mask_checker (MaskChecker): The MaskChecker object.
    
    cut_ptc (float): unused variable for _check_holes_mask_overlap
    
    pts_per_unit_volume (float): The number of points per unit volume to sample when calculating 
        the shell filling fraction. A larger number of points leads to a more accurate 
        calculation at the cost of increased computation. Defaults to 0.01 (intended for volume 
        units of [Mpc/h]^3)
        
    num_surf_pts (int): The number of surface points to place on the outer surface of the 
        spherical shell. These points are used to determine if the shell is fully embedded in the 
        survey volume
    
    returns:
    ---------------------------------------------------------------------------------------------
    vol_frac (numpy array of floats of shape N): The shell filling fraction for each of N shells
    """
    
    # Initalize the shell filling fraction array
    vol_frac = np.zeros(x_y_z_r_r0_array.shape[0], dtype=np.float64)
    
    # Array used for Monte Carlo
    monte_index = np.zeros(x_y_z_r_r0_array.shape[0], dtype=np.uint8)
    
    # Distrubute N points on a unit sphere
    unit_sphere_pts = build_unit_sphere_points(num_surf_pts)

    # Find the largest radius shell in the results, and generate a mesh of
    # constant density such that the largest shell will fit in this mesh
    mesh_points, mesh_points_radii = generate_mesh(x_y_z_r_r0_array[:,3].max(), 
                                                   pts_per_unit_volume)

    # Iterate through the shells and calculate their filling fractions    
    _check_holes_mask_overlap(x_y_z_r_r0_array,
                              mask_checker,
                              unit_sphere_pts,
                              mesh_points,
                              mesh_points_radii,
                              cut_pct,
                              vol_frac,
                              monte_index)
    
    return vol_frac