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
  
    
    if mask_type == "ra_dec_z":
        mask_mode = 0
    elif mask_type == "xyz":
        mask_mode = 1
    elif mask_type == "periodic":
        mask_mode = 2
    else:
        raise ValueError(f"mask_type must be 'ra_dec_z', 'xyz', or 'periodic.' The provided value was '{mask_type}'")
    
    
    if dist_limits is None:
        min_dist = None
        max_dist = None
    else:
        min_dist = dist_limits[0]
        max_dist = dist_limits[1]
    
    
    

    ############################################################################
    # Initialize mask object
    #---------------------------------------------------------------------------
    if mask_mode == 0: # sky mask
        mask_checker = MaskChecker(mask_mode,
                                   survey_mask_ra_dec=mask.astype(np.uint8),
                                   n=mask_resolution,
                                   rmin=min_dist,
                                   rmax=max_dist)
        
    elif mask_mode in [1,2]: # Cartesian mask
        mask_checker = MaskChecker(mask_mode,
                                   xyz_limits=xyz_limits)
    ############################################################################




    
    ############################################################################
    # Calculate volume of shells within the survey mask
    #---------------------------------------------------------------------------
    
    vol_frac = oob_cut_single(x_y_z_r_r0_array, 
                   mask_checker,
                   cut_pct=0.1,
                   pts_per_unit_volume=pts_per_unit_volume,
                   num_surf_pts=20)

    return vol_frac

        

def oob_cut_single(x_y_z_r_r0_array, 
                   mask_checker,
                   cut_pct=0.1,
                   pts_per_unit_volume=3,
                   num_surf_pts=20):
    """
    Out-Of-Bounds cut single threaded version.
    """

    vol_frac = np.zeros(x_y_z_r_r0_array.shape[0], dtype=np.float64)
    
    monte_index = np.zeros(x_y_z_r_r0_array.shape[0], dtype=np.uint8)
    
    ############################################################################
    # Distrubute N points on a unit sphere
    #---------------------------------------------------------------------------
    unit_sphere_pts = build_unit_sphere_points(num_surf_pts)
    ############################################################################

    
    
    ############################################################################
    # Find the largest radius hole in the results, and generate a mesh of
    # constant density such that the largest hole will fit in this mesh
    #---------------------------------------------------------------------------
    mesh_points, mesh_points_radii = generate_mesh(x_y_z_r_r0_array[:,3].max(), 
                                                   pts_per_unit_volume)
    ############################################################################

    
    ############################################################################
    # Iterate through our holes
    #---------------------------------------------------------------------------
    
    _check_holes_mask_overlap(x_y_z_r_r0_array,
                              mask_checker,
                              unit_sphere_pts,
                              mesh_points,
                              mesh_points_radii,
                              cut_pct,
                              vol_frac,
                              monte_index)
    ############################################################################
    

    return vol_frac