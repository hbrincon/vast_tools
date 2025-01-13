#cython: language_level=3


from __future__ import print_function

cimport cython

import numpy as np

cimport numpy as np

np.import_array()  # required in order to use C-API

#from libc.stdio cimport printf


from vast.voidfinder.typedefs cimport DTYPE_CP128_t, \
                      DTYPE_CP64_t, \
                      DTYPE_F64_t, \
                      DTYPE_F32_t, \
                      DTYPE_B_t, \
                      ITYPE_t, \
                      DTYPE_INT32_t, \
                      DTYPE_INT64_t, \
                      DTYPE_INT8_t

from numpy.math cimport NAN, INFINITY

from libc.math cimport fabs, sqrt, asin, atan, ceil#, exp, pow, cos, sin, asin


from vast.voidfinder._voidfinder_cython_find_next cimport not_in_mask as Not_In_Mask
from vast.voidfinder._voidfinder_cython_find_next cimport MaskChecker



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline ITYPE_t binary_searchsorted(DTYPE_F64_t[:] sorted_array,
                                        DTYPE_F64_t search_val,
                                        ):
                                      
    cdef ITYPE_t array_len = sorted_array.shape[0]  
    cdef ITYPE_t left_sub_array_idx = 0
    cdef ITYPE_t right_sub_array_idx = array_len - 1
    cdef ITYPE_t sub_array_len = array_len
    cdef ITYPE_t compare_index
    cdef ITYPE_t idx
    #cdef DTYPE_F64_t search_array_val

    if search_val > sorted_array[right_sub_array_idx]:
        return array_len
    elif search_val < sorted_array[0]:
        return 0
    
    while True:
        
        if sub_array_len < 8:
            
            for idx in range(sub_array_len):
                
                if search_val < sorted_array[left_sub_array_idx+idx]:
                    
                    return left_sub_array_idx+idx
                
        else:
            
            compare_index = (left_sub_array_idx + right_sub_array_idx)/2
            
            if search_val < sorted_array[compare_index]:
                
                right_sub_array_idx = compare_index
                
            else:
                
                left_sub_array_idx = compare_index + 1
                
            sub_array_len = right_sub_array_idx - left_sub_array_idx + 1






@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void _check_holes_mask_overlap(DTYPE_F64_t[:,:] x_y_z_r_r0_array, 
                                     MaskChecker mask_checker,
                                     DTYPE_F64_t[:,:] unit_sphere_pts,
                                     DTYPE_F64_t[:,:] mesh_points,
                                     DTYPE_F64_t[:] mesh_points_radii,
                                     DTYPE_F64_t cut_pct,
                                     DTYPE_F64_t[:] vol_frac,
                                     DTYPE_B_t[:] monte_index
                                     ):
    """
    Given N points on the boundary of each spherical shell, check them against the mask
    if any of the N fall outside the mask, do a monte-carlo volume calculation
    on the sphere
    """
                                           
                                           
    cdef ITYPE_t idx, jdx, kdx, num_holes, num_shell_pts, num_pts_check
    
    cdef DTYPE_F64_t[:] curr_hole_position = np.empty((3,), dtype=np.float64)
    
    cdef DTYPE_F64_t[:] curr_pt = np.empty(3, dtype=np.float64)
        
    cdef DTYPE_F64_t curr_hole_radius
    
    cdef DTYPE_B_t require_monte_carlo, not_in_mask
    
    cdef DTYPE_INT64_t total_checked_pts, total_outside_mask, 
    
    num_holes = x_y_z_r_r0_array.shape[0]
    
    num_shell_pts = unit_sphere_pts.shape[0]
        
    
                        
    for idx in range(num_holes):
        
        curr_hole_position[0] = x_y_z_r_r0_array[idx,0]
        curr_hole_position[1] = x_y_z_r_r0_array[idx,1]
        curr_hole_position[2] = x_y_z_r_r0_array[idx,2]
        
        curr_hole_radius = x_y_z_r_r0_array[idx,3]
        
        curr_hole_radius0 = x_y_z_r_r0_array[idx,4]
        
        ########################################################################
        # First check shell points
        #-----------------------------------------------------------------------
        require_monte_carlo = False
        
        for jdx in range(num_shell_pts):
            
            curr_pt[0] = curr_hole_radius*unit_sphere_pts[jdx,0] + curr_hole_position[0]
            curr_pt[1] = curr_hole_radius*unit_sphere_pts[jdx,1] + curr_hole_position[1]
            curr_pt[2] = curr_hole_radius*unit_sphere_pts[jdx,2] + curr_hole_position[2]
            
            
            not_in_mask = mask_checker.not_in_mask(curr_pt)
            #not_in_mask = Not_In_Mask(curr_pt, mask, mask_resolution, min_dist, max_dist)
            
            if not_in_mask:
                
                require_monte_carlo = True
                
                break
        ########################################################################
        
            
        ################################################################################
        # Check the monte carlo points if any of the shell points failed
        ################################################################################
        if require_monte_carlo:
            
            monte_index[idx] = True
            
            total_checked_pts = 0
            
            total_outside_mask = 0
                               
            num_pts_check = binary_searchsorted(mesh_points_radii, curr_hole_radius)
            
            num_pts_check0 = binary_searchsorted(mesh_points_radii, curr_hole_radius0)
                        
            #check all points within the spherical shell
            for kdx in range(num_pts_check0, num_pts_check):
                
                curr_pt[0] = curr_hole_position[0] + mesh_points[kdx,0]
                curr_pt[1] = curr_hole_position[1] + mesh_points[kdx,1]
                curr_pt[2] = curr_hole_position[2] + mesh_points[kdx,2]
                                                      
                not_in_mask = mask_checker.not_in_mask(curr_pt)
                #not_in_mask = Not_In_Mask(curr_pt, mask, mask_resolution, min_dist, max_dist)
            
                if not_in_mask:
                    
                    total_outside_mask += 1
                                        
            
            vol_frac[idx] = 1.0 - total_outside_mask/(num_pts_check-num_pts_check0)  #fraction of shell volume inside mask   
        else:
            vol_frac[idx] = 1.0
                                           


