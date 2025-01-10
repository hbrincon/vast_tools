import numpy as np

from vast.voidfinder._voidfinder_cython_find_next import MaskChecker
from vast.voidfinder.voidfinder_functions import not_in_mask

#import ShellVolumeMaskedPython as svm

"""
Authors: Hernan Rincon

"""

# This code calculates the volume of a VAST survey mask by tiling it with cubes

#TODO: parallelize code

def calculate_survey_volume(x,y,z,
                            grid_spacing,
                            mask, 
                            mask_resolution,
                            dist_limits,
                            trimmed_dist_limits,
                            mask_trim=30,
                            verbose=True):
    
    #full caclulation process for survey volume
    
    if verbose: print('Generating Grid')
    grid = generate_grid(x,y,z,grid_spacing)
    
    if verbose: print('Applying Survey Mask')
    grid = apply_survey_mask(grid, mask, mask_resolution, dist_limits)
    
    if verbose: print('Applying Edge Cuts')
    
    if mask_trim > 0:
        grid = apply_survey_edge_cut(grid, mask, mask_resolution, dist_limits, mask_trim)
        
    grid = apply_dist_limits(grid, trimmed_dist_limits)
    
    #Record final volume
    vol = get_grid_volume(grid, grid_spacing)
    
    if verbose: print('Survey Volume:',vol)
    
    return vol, grid

def calculate_binned_survey_volume(
                            bin_edges,
                            x,y,z,
                            grid_spacing,
                            mask, 
                            mask_resolution,
                            dist_limits,
                            trimmed_dist_limits,
                            mask_trim=30,
                            verbose=True):
    
    #full caclulation process for survey volume, with distance bins
    
    vol, grid =  calculate_survey_volume(
                            x,y,z,
                            grid_spacing,
                            mask, 
                            mask_resolution,
                            dist_limits,
                            trimmed_dist_limits,
                            mask_trim=mask_trim,
                            verbose=verbose)
    
    if verbose: print('Binning Volumes')
    
    vols = get_binned_grid_volume(bin_edges, grid, grid_spacing)
    
    return vol, vols, grid



def generate_grid(x,y,z, grid_spacing = 1):
    #creates a grid over a survey region with shape (N, 3)
    # x, y, z: galaxy locations in cartesian coordinates 
    # grid_spacing: The default grid spacing is 1 (xyz units)
    x_min = np.floor(np.min(x)/grid_spacing)*grid_spacing
    x_max = np.ceil(np.max(x)/grid_spacing)*grid_spacing
    y_min = np.floor(np.min(y)/grid_spacing)*grid_spacing
    y_max = np.ceil(np.max(y)/grid_spacing)*grid_spacing
    z_min = np.floor(np.min(z)/grid_spacing)*grid_spacing
    z_max = np.ceil(np.max(z)/grid_spacing)*grid_spacing
        
    x_range = np.arange(x_min, x_max+grid_spacing, grid_spacing)
    y_range = np.arange(y_min, y_max+grid_spacing, grid_spacing)
    z_range = np.arange(z_min, z_max+grid_spacing, grid_spacing)

    # Creating a meshgrid from the input ranges 
    X,Y,Z = np.meshgrid(x_range,y_range,z_range)

    x_points = np.ravel(X)
    y_points = np.ravel(Y)
    z_points = np.ravel(Z)
    
    grid = np.array([x_points, y_points, z_points]).T
    
    return grid

def apply_survey_mask(
    grid,
    mask, 
    mask_resolution,
    dist_limits,
    return_indices = False
    ):
    #cut down a grid into points within a survey mask
  
    num_points = grid.shape[0]
    
    
    if dist_limits is None:
        min_dist = None
        max_dist = None
    else:
        min_dist = dist_limits[0]
        max_dist = dist_limits[1]
    
    
    

    
    ############################################################################
    # Calculate which points are in the mask
    #---------------------------------------------------------------------------
    outside_mask = np.full(num_points, 0)
    
    for i in range(num_points):
        #handle origin
        if np.sum(grid[i])==0:
            outside_mask[i] = 1 if min_dist > 0 else 0
        #handle points on z axis
        elif np.sum(grid[i][:-1])==0:
            if grid[i][-1]<0:
                outside_mask[i] = 0 if np.any(mask[:,0]) else 1
            else:
                outside_mask[i] = 0 if np.any(mask[:,-1]) else 1
        #handle points in yz plane
        #TODO: divide by zero encountered in double_scalars ra = np.arctan(coords[1]/coords[0])
        elif grid[i][0] == 0:
            #Temporary solution, exclude these points
            outside_mask[i] = 1
        else:
            outside_mask[i] = not_in_mask([grid[i]], mask, mask_resolution, min_dist, max_dist)
    
    if return_indices:
        return ~outside_mask.astype(bool)
    
    return grid[~outside_mask.astype(bool)]


def apply_survey_edge_cut(
    grid,
    mask, 
    mask_resolution,
    dist_limits,
    mask_trim,
    ):
    #removes objects near the mask borders
    
    
    num_points = grid.shape[0]

    
    #find points that are fully in mask within a distance of mask_trim
    
    in_mask = np.full(num_points, True)
    
    # Create a set of points on the surface of a sphere to test for mask membership
    coords = np.array([
        [0, 2**.5, 2**-.25],
        [0, -2**.5, 2**-.25],
        [2**.5, 0, 2**-.25],
        [-2**.5, 0, 2**-.25],
        [1, 1, -2**-.25],
        [1, -1, -2**-.25],
        [-1, 1, -2**-.25],
        [-1, -1, -2**-.25],
    ])


    for x,y,z in coords:
    
        grid_copy = grid * 1
        # defined so that 0<theta<pi and 0<phi<2pi
        grid_copy[:,0] = grid_copy[:,0] + x * mask_trim
        grid_copy[:,1] = grid_copy[:,1] + y * mask_trim
        grid_copy[:,2] = grid_copy[:,2] + z * mask_trim
        print(grid_copy[0])

        in_mask = in_mask * apply_survey_mask(grid_copy, mask, mask_resolution, dist_limits, return_indices = True)
    
   
    
    return grid[in_mask], in_mask

"""
Old version:

def apply_survey_edge_cut(
    grid,
    mask, 
    mask_resolution,
    dist_limits,
    grid_spacing,
    mask_trim,
    ):
    #removes objects near the mask borders
    
    
    num_points = grid.shape[0]

    near_edge = np.full(num_points, False)
    
    #find points that are fully in mask within a distance of mask_trim
    
    in_mask = np.full(num_points, True)
    
    # We will check eight cooridnates in a cube around each point
    x_coords = [-1, -1, -1,  1,  1,  1, -1, 1]
    y_coords = [-1, -1,  1, -1,  1, -1,  1, 1]
    z_coords = [-1,  1, -1, -1, -1,  1,  1, 1]
    extreme_coords = np.array([x_coords, y_coords, z_coords]).T
    
    for coord in extreme_coord:
    
        grid_copy = grid * 1

        grid_copy[:,0] = grid_copy[:,0] + coord[0] * mask_trim
        grid_copy[:,1] = grid_copy[:,1] + coord[1] * mask_trim
        grid_copy[:,2] = grid_copy[:,2] + coord[2] * mask_trim

        in_mask = in_mask * apply_survey_mask(grid_copy, mask, mask_resolution, dist_limits, return_indices = True)
    
    # mark points that are fully in mask within a distance of mask_trim
    near_edge[in_mask] = True
    
    
    for i in range(num_points):
        
        #skip over points that we know are fully in mask within a distance of mask_trim
        if near_edge[i] == True:
            continue
        
        vol_frac = svm.shell_fraction(
               np.array([np.concatenate((grid[i], [mask_trim, 0]))]),
               mask=mask, 
               mask_resolution=mask_resolution,
               dist_limits=dist_limits,
               pts_per_unit_volume=0.001,
               mask_type='ra_dec_z'
               )[0]
               
        
        #not_in_mask([grid[i]], mask, mask_resolution, min_dist, max_dist)
        near_edge[i] = vol_frac < 1
    
    return grid[~near_edge]
"""

def apply_dist_limits(
    grid,
    trimmed_dist_limits,
    ):
    #removes objects near the redshift limits
    r = np.linalg.norm(grid, axis=1)
    in_dist_bound = (r > trimmed_dist_limits[0]) * (r < trimmed_dist_limits[1])
    return grid[in_dist_bound]

def get_grid_volume(grid, grid_spacing):
    #the volume of a grid
    #returns the survey volume one appropiate cuts are made on the grid
    
    num_points = grid.shape[0]
    
    return num_points * grid_spacing ** 3



def get_binned_grid_volume(bin_edges, grid, grid_spacing):
    #volume of a grid, broken up into LOS distance bins
    vols = []
    r = np.linalg.norm(grid, axis=1)
    for i, bin_low in enumerate(bin_edges[:-1]):
        in_bin = (r >= bin_low) * (r < bin_edges[i+1])
        vols.append(get_grid_volume(grid[in_bin], grid_spacing))
    
    return np.array(vols)

# Extra functions

def calculate_number_density(
                            binned_volumes,
                            bin_edges,
                            x,y,z,
                            grid_spacing,
                            mask, 
                            mask_resolution,
                            dist_limits,
                            trimmed_dist_limits,
                            mask_trim=0,
                            verbose=True):
    
    if binned_volumes is None:
        vol, binned_volumes, grid = calculate_binned_survey_volume(
                                    bin_edges,
                                    x,y,z,
                                    grid_spacing,
                                    mask, 
                                    mask_resolution,
                                    dist_limits,
                                    trimmed_dist_limits,
                                    mask_trim=mask_trim,
                                    verbose=verbose)
    
    comov_dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    number = []
    number_density = []
    
    for bin_l, bin_h, V in zip (bin_edges[:-1], bin_edges[1:], binned_volumes):
        
        N = len(comov_dist[(comov_dist > bin_l)*(comov_dist < bin_h)])
        
        number.append(N)
        number_density.append(N/V)

    return np.array(number_density), np.array(number)
