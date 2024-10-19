'''Identify galaxies as being in a void, a wall, or an alternate classification.'''


################################################################################
# IMPORT LIBRARIES
#-------------------------------------------------------------------------------
import os

import numpy as np

from astropy.table import Table

import pickle

import matplotlib.pyplot as plt


#import sys
#sys.path.insert(1, '/local/path/VAST/VoidFinder/vast/voidfinder/')
from vast.voidfinder.vflag import determine_vflag
from vast.voidfinder.distance import z_to_comoving_dist
################################################################################






#-------------------------------------------------------------------------------
################################################################################





################################################################################
# CONSTANTS
#-------------------------------------------------------------------------------
c = 3e5 # km/s

h = 1
H = 100*h

Omega_M = 0.26

DtoR = np.pi/180
################################################################################




class GalaxyEnvClassifier():
    
    def __init__(self, galaxies, voids, mask_filename=None, mask_tuple=None, dist_metric = 'comoving'):
        
        #Perform the following as input to this class
        
        # galaxy_file_format = 'commented_header'
        # galaxies = Table.read( galaxy_filename, format='ascii.' + galaxy_file_format)
        # Don't use escv files for galaxy_filename (or use the original ClassifyEnvironment.py if you do)
        
        # Read in list of void holes
        # voids = Table.read(void_filename, format='ascii.commented_header')
        '''
        voids['x'] == x-coordinate of center of void (in Mpc/h)
        voids['y'] == y-coordinate of center of void (in Mpc/h)
        voids['z'] == z-coordinate of center of void (in Mpc/h)
        voids['R'] == radius of void (in Mpc/h)
        voids['voidID'] == index number identifying to which void the sphere belongs
        '''
        

        
        self.galaxies = galaxies
        self.voids = voids
        self.mask_filename = mask_filename
        self.mask_tuple = mask_tuple
        self.dist_metric = dist_metric
    def classify(self):
        
        galaxies = self.galaxies
        voids = self.voids

        # Read in survey mask
        if self.mask_tuple is None:
            mask_infile = open(self.mask_filename, 'rb')
            mask, mask_resolution, dist_limits = pickle.load(mask_infile)
            mask_infile.close()
        else:
            mask, mask_resolution, dist_limits = self.mask_tuple
        self.mask=mask
        self.mask_resolution = mask_resolution
        self.dist_limits = dist_limits

        
        ################################################################################




        ################################################################################
        # CONVERT GALAXY ra,dec,z TO x,y,z
        #
        # Conversions are from http://www.physics.drexel.edu/~pan/VoidCatalog/README
        #-------------------------------------------------------------------------------
        print('Converting coordinate system')

        # Convert redshift to distance
        if self.dist_metric == 'comoving':
            if 'Rgal' not in galaxies.columns:
                galaxies['Rgal'] = z_to_comoving_dist(galaxies['redshift'].data.astype(np.float32), Omega_M, h)
            galaxies_r = galaxies['Rgal']

        else:
            galaxies_r = c*galaxies['redshift']/H


        # Calculate x-coordinates
        galaxies_x = galaxies_r*np.cos(galaxies['dec']*DtoR)*np.cos(galaxies['ra']*DtoR)

        # Calculate y-coordinates
        galaxies_y = galaxies_r*np.cos(galaxies['dec']*DtoR)*np.sin(galaxies['ra']*DtoR)

        # Calculate z-coordinates
        galaxies_z = galaxies_r*np.sin(galaxies['dec']*DtoR)

        print('Coordinates converted')
        ################################################################################





        ################################################################################
        # IDENTIFY LARGE-SCALE ENVIRONMENT
        #-------------------------------------------------------------------------------
        print('Identifying environment')

        galaxies['vflag'] = -9

        for i in range(len(galaxies)):

            #vflag : integer
            #0 = wall galaxy
            #1 = void galaxy
            #2 = edge galaxy (too close to survey boundary to determine)
            #9 = outside survey footprint

            galaxies['vflag'][i] = determine_vflag(galaxies_x[i], 
                                                   galaxies_y[i], 
                                                   galaxies_z[i], 
                                                   voids, 
                                                   mask, 
                                                   mask_resolution, 
                                                   dist_limits[0], 
                                                   dist_limits[1])

        print('Environments identified')
        ################################################################################





        ################################################################################
        # OUTPUT RESULTS
        #-------------------------------------------------------------------------------
        return galaxies

    def createImages(self, title, save_image = False):
        """
        Creates an output plot of (1) the mask and (2) the galaxies partitioned into 
        void/wall/other types in ra-dec coordinates

        Parameters:
        title: string
            A name that is attached to the output png files to identify them
        """
        galaxies = self.galaxies
        mask=self.mask
        #Save graphical information

        #mask
        plt.imshow(np.rot90(mask.astype(int)))
        plt.savefig(title + "_classify_env_mask.png",dpi=100,bbox_inches="tight")

        #galaxy catagories
        walls=np.where(galaxies['vflag']==0)
        voids=np.where(galaxies['vflag']==1)
        wall_gals=np.array(galaxies)[walls]
        void_gals=np.array(galaxies)[voids]
        plt.figure(dpi=100)
        plt.scatter(galaxies['ra'],galaxies['dec'],s=.5,label="excluded")
        plt.scatter(void_gals['ra'],void_gals['dec'],color='r',s=.5,label="voids")
        plt.scatter(wall_gals['ra'],wall_gals['dec'],color='k',s=.5,label="walls")
        plt.legend(loc="upper right")
        plt.xlabel("RA")
        plt.ylabel("DEC")
        plt.title("Galaxy Distribution")
        
        if save_image:
            plt.savefig(title + "_classify_env_gals.png",dpi=100,bbox_inches="tight")

        return