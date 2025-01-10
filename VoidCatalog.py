from vast.voidfinder.postprocessing import open_fits_file
from vast.voidfinder.preprocessing import load_data_to_Table
from vast.voidfinder.vflag import determine_vflag
from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder.voidfinder import ra_dec_to_xyz
from vast.voidfinder._voidfinder_cython_find_next import MaskChecker
from vast.vsquared.util import open_fits_file_V2


import numpy as np
import copy
from astropy.table import Table, vstack
from astropy.io import fits

import VoidVolume as vol
import VoidOverlap as vo
import VoidSWRadii as vswr
import SurveyVolume as sv
import ShellVolumeMaskedPython as svm

"""
Authors: Hernan Rincon

"""

# Void catalog classes designed for loading VAST void catalogs and performing analysis
   
c = 3e5 # km/s

DtoR = np.pi/180

class VoidCatalog():
    # base class for void catalogs
    
    def __init__(self, edge_buffer):
        self.edge_buffer = edge_buffer
    
    def __getitem__(self, table):
        return self.tables[str.upper(table)]
    
    def lower_col_names(self):
        for table in self.tables.values():
            for name in table.colnames:
                table[name].name = name.lower()
        
    def upper_col_names(self):
        for table in self.tables.values():
            for name in table.colnames:
                table[name].name = name.upper()
                
    def read_catalog(self, file_name):
        raise NotImplementedError("Please Implement this method")
        
    def clear_catalog(self):
        del self._catalog
                
    def save_catalog(self, output_file_name = None):
        
        #overwrite catalog if no output file is specified
        if output_file_name is None:
            output_file_name = self.file_name
            
        #read catalog
        self.read_catalog()
        
        #select formatting for column names
        if self.capitalize_colnames:
            self.upper_col_names()
        
        #update each catalog table
        for table_name, table in self.tables.items():
            
            try:
                self._catalog[table_name].data = fits.BinTableHDU(table).data
                
            except:
                
                hdu = fits.BinTableHDU(table)
                hdu.name = table_name
                self._catalog.append(hdu)
        
        #update each catalog header
        for header_name, header in self.headers.items():
        
            self._catalog[header_name].header = header
            
        #save catalog
        self._catalog.writeto(output_file_name, overwrite=True)
        
        #undo column name formatting
        if self.capitalize_colnames:
            self.lower_col_names()
            
        #free up memory    
        self.clear_catalog()
                
    def add_galaxies(self, galaxies_path, load_vflag = False, redshift_name='redshift', ra_name = 'ra', dec_name='dec', cartesian = False, x_name = 'x', y_name = 'y', z_name = 'z'):
            
        self.galaxies = load_data_to_Table(galaxies_path)
        self.galaxies['gal'] = np.arange(len(self.galaxies)) 
        
        if cartesian:
            if x_name != 'x':
                self.galaxies[x_name].name = 'x'
            if y_name != 'y':
                self.galaxies[y_name].name = 'y'
            if z_name != 'z':
                self.galaxies[z_name].name = 'z'
            
            #temporary fix for any caclulation requireing Rgal (there shouldreally be no such caclualtions for the cartesian case)   
            #self.galaxies['Rgal'] = np.sqrt(self.galaxies['x']**2 + self.galaxies['y']**2 + self.galaxies['z']**2)
        else: 
            if redshift_name != 'redshift':
                self.galaxies[redshift_name].name = 'redshift'
            if ra_name != 'ra':
                self.galaxies[ra_name].name = 'ra'
            if dec_name != 'dec':
                self.galaxies[dec_name].name = 'dec'
                

            if np.sum(~np.isin(['x','y','z','Rgal'], self.galaxies.colnames)) > 0:

                self.galaxies['Rgal']=z_to_comoving_dist(self.galaxies['redshift'].astype(np.float32),
                                                        self.info['OMEGAM'],self.info['HP'])
                tmp = ra_dec_to_xyz(self.galaxies)
                self.galaxies['x']=tmp[:,0]
                self.galaxies['y']=tmp[:,1]
                self.galaxies['z']=tmp[:,2]

        # For compatability with old vflag model where info was stored in 
        # galaxies file instead of void catalog
        if load_vflag:
            
            hdu = fits.BinTableHDU(
                Table(self.galaxies['gal','vflag'],
                 names = ['gal', 'vflag'])
            )
            hdu.name = 'VFLAG'
            try:
                self._catalog['VFLAG'].data = hdu.data
            except:
                self._catalog.append(hdu)
                self.vflag = Table(self._catalog['VFLAG'].data)
                self.tables['VFLAG'] = self.vflag
                

            
    def get_single_overlap(self, mask_hdu=None):
        
        if mask_hdu is None and hasattr(self, 'info') and hasattr(self, 'mask'):
            mask = self.mask
            mask_res = self.info['MSKRES']
        else:
            mask = mask_hdu.data.astype(bool) # convert from into to bool to avoid endian compiler error
            mask_res = mask_hdu.header['MSKRES']

        if isinstance(self, VoidFinderCatalog):
            cat_type='VF'
            zone1 = None
            void1 = self.holes 
        elif isinstance(self, V2Catalog):
            cat_type='V2'
            void1 = self.galzone
            zone1 = self.zonevoid

        rmin = self.info['DLIML']
        rmax = self.info['DLIMU']
        vosc = vo.SingleCalculator(void1,  "Cat 1", 
                                          None,
                                          rmin, rmax,
                                          zone_table_V1 = zone1,
                                          V1_algorithm=cat_type, 
                                          mask_tuple=(mask.astype(bool), mask_res)
                                         )
        vosc.find_overlap(self.edge_buffer)

        return vosc.report(do_print=False, do_return=True)
    
                

class VoidFinderCatalog (VoidCatalog):
    # Class for VoidFinder catalogs
    
    def __init__ (self, file_name, survey_name=None, directory = './', edge_buffer=30):
        
        super().__init__(edge_buffer)
        
        # format input file name
        if file_name is None:
            file_name = directory + survey_name + '_VoidFinder_Output.fits'
            
        # read input file
        self.read_catalog(file_name)
        self.file_name = file_name
        #Gather all column names that appear in the catalog
        hdu_names = [self._catalog[i].name for i in range(len(self._catalog))]
        
        #format column names
        col_names = []
          
        if 'WALL' in hdu_names:
            col_names = col_names + self._catalog['WALL'].data.names
        if 'FIELD' in hdu_names:
            col_names = col_names + self._catalog['FIELD'].data.names
        if 'MAXIMALS' in hdu_names:
            col_names = col_names + self._catalog['MAXIMALS'].data.names
        if 'HOLES' in hdu_names:
            col_names = col_names + self._catalog['HOLES'].data.names
        
        #lowercase version of column names (possibly identical to col_names)
        col_names_lower = [string.lower() for string in col_names]
        
        # create dictionary that maps each column name used by VoidFinderCatalogBase to the correct
        # column name in the void data file
        self.capitalize_colnames = False
        
        for key, item in zip(col_names_lower, col_names):

            if col_names_lower != col_names:
                self.capitalize_colnames = True
            
        
        # Format data tables        
        self.tables = {}
        self.headers = {}
        
        if 'PRIMARY' in hdu_names:
            self.info = self._catalog['PRIMARY'].header
            self.headers['PRIMARY'] = self.info
        if 'MASK' in hdu_names:
            self.mask_info = self._catalog['MASK'].header
            self.mask = self._catalog['MASK'].data
        if 'WALL' in hdu_names:
            self.wall_info = self._catalog['WALL'].header
            self.wall = Table(self._catalog['WALL'].data)
            self.tables['WALL'] = self.wall
            self.headers['WALL'] = self.wall_info
        if 'FIELD' in hdu_names:
            self.field_info = self._catalog['FIELD'].header
            self.field = Table(self._catalog['FIELD'].data)
            self.tables['FIELD'] = self.field
            self.headers['FIELD'] = self.field_info
        if 'MAXIMALS' in hdu_names:
            self.maximals_info = self._catalog['MAXIMALS'].header
            self.maximals = Table(self._catalog['MAXIMALS'].data)
            self.tables['MAXIMALS'] = self.maximals
            self.headers['MAXIMALS'] = self.maximals_info
        if 'HOLES' in hdu_names:
            self.holes_info = self._catalog['HOLES'].header
            self.holes = Table(self._catalog['HOLES'].data)
            self.tables['HOLES'] = self.holes
            self.headers['HOLES'] = self.holes_info
        if 'VFLAG' in hdu_names:
            self.vflag = Table(self._catalog['VFLAG'].data)
            self.tables['VFLAG'] = self.vflag
            
        self.lower_col_names()
        
    def read_catalog(self, file_name):
        self._catalog = open_fits_file(file_name)
     
    def void_stats(self):
        
        num_voids = len(self.maximals)
        print(num_voids, 'voids')
        edge = self.maximals['edge']
        print(len(self.maximals[edge==1]),'edge voids')
        print(len(self.maximals[edge==2]),'near-edge voids')
        print(len(self.maximals[edge==0]),'interior voids')
        
        if np.product(np.isin(['r_eff','r_eff_uncert'], self.maximals.colnames)) > 0:
            
            points_boolean = np.zeros(len(self.maximals), dtype = bool)
            
            mask = self.mask
            mask_res = self.mask_info['MSKRES']
            rmin = self.info['DLIML']
            rmax = self.info['DLIMU']

            #Remove voids near the survey edges
            for i in range(len(self.maximals)):
                # The current point
                curr_pt = self.maximals[i]

                is_edge = vo.is_edge_point(curr_pt['x'], curr_pt['y'], curr_pt['z'],
                                           mask, mask_res, rmin, rmax, self.edge_buffer)
                points_boolean[i] = not is_edge

            maximals = self.maximals[points_boolean]
            
            edge = maximals['edge']
            print(len(maximals[edge==1]),'edge voids (V. Fid)')
            print(len(maximals[edge==2]),'near-edge voids (V. Fid)')
            print(len(maximals[edge==0]),'interior voids (V. Fid)')
            
            reff = maximals['r_eff']
            uncert_mean = np.std(reff) / np.sqrt(num_voids)
            uncert_median = np.sqrt(np.pi / 2) * uncert_mean

            print('Mean Reff (V. Fid):', mknum(np.mean(reff)), '+/-',mknum(uncert_mean),'Mpc/h')
            print('Median Reff (V. Fid):', mknum(np.median(reff)), '+/-',mknum(uncert_median),'Mpc/h')
            print('Maximum Reff (V. Fid):', mknum(np.max(reff)),'Mpc/h')
            
    def calculate_r_eff(self, overwrite = False, save_every = None):
        
        def save_r_eff():
            #format and save output
            if self.capitalize_colnames:
                self.upper_col_names()

            self._catalog['MAXIMALS'].data = fits.BinTableHDU(self.maximals).data
            self._catalog.writeto(self.file_name, overwrite=True)

            if self.capitalize_colnames:
                self.lower_col_names()
        
        save_every_applied = save_every is not None
        
        
        if not save_every_applied:
            #ensure that reff wasn't previously calculated
            if not overwrite and np.sum(np.isin(['r_eff','r_eff_uncert'],self.maximals.colnames))>0:
                print ('R_eff already calculated. Run with overwrite=True to overwrite effective radii')
                return
            self.maximals['r_eff'] = -1.
            self.maximals['r_eff_uncert'] = -1.
                    
            
        else:
            if not np.sum(np.isin(['r_eff','r_eff_uncert'],self.maximals.colnames))>0:
                self.maximals['r_eff'] = -1.
                self.maximals['r_eff_uncert'] = -1.
                    
        # calculate reff
        flags = self.maximals['void'][self.maximals['r_eff']==-1]
        for i, flag in enumerate(flags):
            holes = self.holes[self.holes['void']==flag]
            positions = np.array([holes['x'], holes['y'],holes['z']]).T
            radius = holes['radius'].data
            vol_info = vol.volume_of_spheres(positions, radius)
            self.maximals['r_eff'][flag] = ((3/4) * vol_info[2] / np.pi) ** (1/3) 
            self.maximals['r_eff_uncert'][flag] = vol_info[3] * ((3 * vol_info[2]) ** -2 / (4 * np.pi)) ** (1/3) 
            if save_every_applied and i%save_every == 0:
                save_r_eff()
        
        save_r_eff()
 
    def calculate_vflag(self, overwrite = False, dist_metric = 'comoving', cartesian = False):
        # warning: no mask feature is used for cubic box simulations (cartesian = True). 
        # All galaxies are assumed to be inside the mask. There is no option for a 
        # periodic mode.
        
        galaxies = self.galaxies
        
        #ensure that vflag wasn't previously calculated
        hdu_names = [self._catalog[i].name for i in range(len(self._catalog))]
        if not overwrite and 'VFLAG' in hdu_names:
            print ('vflags already calculated. Run with overwrite=True to overwrite vflags')
            return
        

        # Calculate xyz-coordinates
        galaxies_x = galaxies['x']
        galaxies_y = galaxies['y']
        galaxies_z = galaxies['z']
        
        #set up mask
        if cartesian:
            mask = np.ones ((360, 180))
            mask_res = 1
            rmin = -np.inf
            rmax = np.inf
        else:
            mask = self.mask
            mask_res = self.mask_info['MSKRES']
            rmin = self.info['DLIML']
            rmax = self.info['DLIMU']

        ################################################################################
        # IDENTIFY LARGE-SCALE ENVIRONMENT
        #-------------------------------------------------------------------------------
        print('Identifying environment')

        galaxies['vflag'] = -9
        
        voids = Table(self.holes)

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
                                                   mask_res, 
                                                   rmin,
                                                   rmax)
            
            
        ################################################################################
        # SAVE OUTPUT
        #-------------------------------------------------------------------------------
        
        hdu = fits.BinTableHDU(
            Table(self.galaxies['gal','vflag'],
             names = ['gal', 'vflag'])
        )
        hdu.name = 'VFLAG'
        try:
            self._catalog['VFLAG'].data = hdu.data
        except:
            self._catalog.append(hdu)
            self.vflag = Table(self._catalog['VFLAG'].data)
            self.tables['VFLAG'] = self.vflag
            
        #format and save output
        if self.capitalize_colnames:
            self.upper_col_names()
            
        self._catalog.writeto(self.file_name, overwrite=True)
        
        if self.capitalize_colnames:
            self.lower_col_names()
            
    def plot_vflag(self, mask_title='Survey Mask', galaxies_title='Galaxy Distribution', file_prefix='vast', save_image = False):
        """
        Creates an output plot of (1) the mask and (2) the galaxies partitioned into 
        void/wall/other types in ra-dec coordinates

        Parameters:
        mask_title: string
            The mask plot title
        galaxies_title: string
            The galaxies plot title
        file_prefix: string
            A name that is attached to the output png files to identify them
        """
        
        print('WARNING: ensure that the calcualted vflags match the currently loaded galaxy file, as vflags are saved to the void file and not the galaxy file.')
        
        if not hasattr(self, 'vflag'):
            raise ValueError('vflags not calculated for galaxies.')
            
        galaxies = self.galaxies
        mask=self.mask
        
        #Save graphical information

        #mask
        plt.imshow(np.rot90(mask.astype(int)))
        plt.xlabel("RA [pixels]")
        plt.ylabel("Dec. [pixels]")
        plt.title(mask_title)
        if save_image:
            plt.savefig(file_prefix + "_classify_env_mask.png",dpi=100,bbox_inches="tight")

        #galaxy catagories
        walls=np.where(self.vflag['vflag']==0)
        voids=np.where(self.vflag['vflag']==1)
        wall_gals=np.array(galaxies)[walls]
        void_gals=np.array(galaxies)[voids]
        plt.figure(dpi=100)
        plt.scatter(galaxies['ra'],galaxies['dec'],s=.5,label="excluded")
        plt.scatter(void_gals['ra'],void_gals['dec'],color='r',s=.5,label="voids")
        plt.scatter(wall_gals['ra'],wall_gals['dec'],color='k',s=.5,label="walls")
        plt.legend(loc="upper right")
        plt.xlabel("RA")
        plt.ylabel("Dec.")
        plt.title(galaxies_title)
        
        if save_image:
            plt.savefig(file_prefix + "_classify_env_gals.png",dpi=100,bbox_inches="tight")      
        
    def galaxy_membership(self, custom_mask_hdu=None, return_selector=False,
                         rmin = None, rmax = None, mag_lim = None):
        
        print('WARNING: ensure that the calcualted vflags match the currently loaded galaxy file, as vflags are saved to the void file and not the galaxy file.')
        
        if rmin is None:
            rmin = self.info['DLIML']
        if rmax is None:
            rmax = self.info['DLIMU']
        if mag_lim is None:
            mag_lim = self.info['MAGLIM']
        
        self.galaxies['vflag'] = self.vflag['vflag']
        
        if custom_mask_hdu is not None:
            mask = custom_mask_hdu.data
            mask_res = custom_mask_hdu.header['MSKRES']
        else:
            mask = self.mask
            mask_res = self.mask_info['MSKRES']
        
        galaxies = select_mask(self.galaxies, mask, mask_res, rmin, rmax)

        galaxies = galaxies[galaxies['rabsmag'] < mag_lim]
        
        # Cut galaxies down to those within 10 mpc/h of survey border
        # Note: we use the main survey mask rather than the custom mask option, 
        # because were ony worried about galaxies near edge voids
        points_boolean = np.zeros(len(galaxies), dtype = bool)

        #Flag points that fall outside the main survey mask
        for i in range(len(galaxies)):
            # The current point
            curr_pt = galaxies[i]

            is_edge = vo.is_edge_point(curr_pt['x'], curr_pt['y'], curr_pt['z'],
                                       mask, mask_res, rmin, rmax, self.edge_buffer)
            points_boolean[i] = not is_edge
        
        galaxies = galaxies[points_boolean]
        
        num_in_void = np.sum(galaxies['vflag']==1) 
        
        # This is redundant now, points_boolean already makes this selection 
        # while correctly handling custom masks
        # num_tot = np.sum((galaxies['vflag']==1) + (galaxies['vflag']==0)) 
        # Instead we can simply do
        num_tot = len(galaxies)
        
        if return_selector:
            selector = np.isin(self.galaxies['gal'], galaxies['gal'][galaxies['vflag']==1])
            return ( selector, num_tot )

        return ( num_in_void, num_tot )



    # return a new void catalog containing only voids in a masked region
    def subsample_catalog(self, mask_hdu, catalog_file_path, rmin = None, rmax = None):
        
        if rmin is None:
            rmin = self.info['DLIML']
            
        if rmax is None:
            rmax = self.info['DLIMU']   
            
        #deep copy the current catalog
        new_catalog = copy.deepcopy(self)
        
        #change the file output name
        new_catalog.file_name = catalog_file_path
        
        # mask the maximals
        
        new_catalog.maximals = select_mask(self.maximals, mask_hdu.data, 
                                           mask_hdu.header['MSKRES'], rmin, rmax, r_name = 'r')
        new_catalog._catalog['MAXIMALS'] = fits.BinTableHDU(new_catalog.maximals)
        
            
        #TODO: update the maximal HDU header with new void count
        
        #grab holes that are in maximals
        
        holes_select = np.isin(self.holes['void'], new_catalog.maximals['void'])
        
        new_catalog.holes = self.holes[holes_select]
        new_catalog._catalog['HOLES'] = fits.BinTableHDU(new_catalog.holes)
        
        #TODO: update the holes HDU header with new void count
        
        #if galaxies exist, mask the galaxies
        
        if hasattr(self, 'galaxies'):
            new_catalog.galaxies = select_mask(self.galaxies, mask_hdu.data, 
                                               mask_hdu.header['MSKRES'], rmin, rmax)

            gal_select = np.isin(self.galaxies['gal'], new_catalog.galaxies['gal'])

            if hasattr(self, 'vflag'):

                new_catalog.vflag = self.vflag[gal_select]

                self._catalog['VFLAG'].data = fits.BinTableHDU(new_catalog.vflag).data

        
        #replace the mask HDU
        if hasattr(self, 'mask'):
            self.mask = mask_hdu.data
            new_catalog._catalog['MASK'] = fits.ImageHDU(mask_hdu.data)
        
        #TODO update the mask HDU header with new mask info
        
        #TODO: update he primary HDU with new mask info
        
        return new_catalog
        
    
        

class V2Catalog(VoidCatalog):
    # Class for V2 catalogs
    
    def __init__(self, file_name, survey_name=None, pruning = 'VIDE', directory = './', edge_buffer=30):
        
        super().__init__(edge_buffer)
        
        if file_name is None:
            file_name = directory + survey_name + f'_V2_{pruning}_Output.fits'
        
        self.read_catalog(file_name)
        self.file_name = file_name
        hdu_names = [self._catalog[i].name for i in range(len(self._catalog))]
        
        #define column names
        
        #Gather all column names that appear in the catalog
        col_names = []
        
        
        if 'VOIDS' in hdu_names:
            col_names = col_names + self._catalog['VOIDS'].data.names
        if 'ZONEVOID' in hdu_names:
            col_names = col_names + self._catalog['ZONEVOID'].data.names
        if 'GALZONE' in hdu_names:
            col_names = col_names + self._catalog['GALZONE'].data.names
        if 'TRIANGLE' in hdu_names:
            col_names = col_names + self._catalog['TRIANGLE'].data.names
        if 'GALVIZ' in hdu_names:
            col_names = col_names + self._catalog['GALVIZ'].data.names
        
        #lowercase version of column names (possibly identical to col_names)
        col_names_lower = [string.lower() for string in col_names]
        
        # create dictionary that maps each column name used by VoidFinderCatalogBase to the correct
        # column name in the void data file
        self.capitalize_colnames = False
        
        for key, item in zip(col_names_lower, col_names):
            
            if col_names_lower != col_names:
                self.capitalize_colnames = True
        
        
        self.tables = {}
        self.headers = {}
        
        if 'PRIMARY' in hdu_names:
            self.info = self._catalog['PRIMARY'].header
            self.headers['PRIMARY'] = self.info
        if 'VOIDS' in hdu_names:
            self.voids_info = self._catalog['VOIDS'].header
            self.voids = Table(self._catalog['VOIDS'].data)
            self.tables['VOIDS'] = self.voids
            self.headers['VOIDS'] = self.voids_info
        if 'ZONEVOID' in hdu_names:
            self.zonevoid_info = self._catalog['ZONEVOID'].header
            self.zonevoid = Table(self._catalog['ZONEVOID'].data)
            self.tables['ZONEVOID'] = self.zonevoid
            self.headers['ZONEVOID'] = self.zonevoid_info
        if 'GALZONE' in hdu_names:
            self.galzone_info = self._catalog['GALZONE'].header
            self.galzone = Table(self._catalog['GALZONE'].data)   
            self.tables['GALZONE'] = self.galzone
            self.headers['GALZONE'] = self.galzone_info
        if 'TRIANGLE' in hdu_names:
            self.triangle_info = self._catalog['TRIANGLE'].header
            self.triangle = Table(self._catalog['TRIANGLE'].data)
            self.tables['TRIANGLE'] = self.triangle
            self.headers['TRIANGLE'] = self.triangle_info
        if 'GALVIZ' in hdu_names:
            self.galviz_info = self._catalog['GALVIZ'].header
            self.galviz = Table(self._catalog['GALVIZ'].data)
            self.tables['GALVIZ'] = self.galviz
            self.headers['GALVIZ'] = self.galviz_info

        self.lower_col_names()
        
    """def add_mask(self, voidfinder_cat):
        #copy over ask info from a voidfinder catalog
        # This function exists because V2 doen'st save masks. In future work, V2 should
        # just create a mask when it runs
        self.mask_info = voidfinder_cat.mask_info
        self.mask = voidfinder_cat.mask
        # This is a workaround for me mistakenly running VF and V2 with different redshift limits
        # In future work, this should be removed, and the catalogs should have the same redshift limits
        self.mask_info['DLIML'] = voidfinder_cat.info['DLIML']
        self.mask_info['DLIMU'] = voidfinder_cat.info['DLIMU']"""
    
    def read_catalog(self, file_name):
        self._catalog = open_fits_file_V2(file_name,None)        
        
    def void_stats(self):
        
        num_voids = len(self.voids)
        print(num_voids, 'voids')
        if np.product(np.isin(['tot_area','edge_area'], self.voids.colnames))>0:
            edge_area = self.voids['edge_area']
            tot_area = self.voids['tot_area']
            edge = edge_area/tot_area > 0.1
            print(len(self.voids[edge]),'edge voids')
            print(len(self.voids[~edge]),'interior voids')
            
        points_boolean = np.zeros(len(self.voids), dtype = bool)

        mask = self.mask
        mask_res = self.mask_info['MSKRES']
        rmin = self.info['DLIML']
        rmax = self.info['DLIMU']

        #Remove voids near the survey edges
        for i in range(len(self.voids)):
            # The current point
            curr_pt = self.voids[i]

            is_edge = vo.is_edge_point(curr_pt['x'], curr_pt['y'], curr_pt['z'],
                                       mask, mask_res, rmin, rmax, self.edge_buffer)
            points_boolean[i] = not is_edge

        voids = self.voids[points_boolean]
        
        if np.product(np.isin(['tot_area','edge_area'], self.voids.colnames))>0:
            edge_area = voids['edge_area']
            tot_area = voids['tot_area']
            edge = edge_area/tot_area > 0.1
            print(len(voids[edge]),'edge voids (V. Fid)')
            print(len(voids[~edge]),'interior voids (V. Fid)')
        
        reff = voids['radius']
        uncert_mean = np.std(reff) / np.sqrt(num_voids)
        uncert_median = np.sqrt(np.pi / 2) * uncert_mean
        
        print('Mean Reff (V. Fid):', mknum(np.mean(reff)), '+/-',mknum(uncert_mean),'Mpc/h')
        print('Median Reff (V. Fid):', mknum(np.median(reff)), '+/-',mknum(uncert_median),'Mpc/h')
        print('Maximum Reff (V. Fid):', mknum(np.max(reff)), 'Mpc/h')
        
    def galaxy_membership(self, custom_mask_hdu=None, return_selector=False,
                         rmin = None, rmax = None, mag_lim = None):
        
        if rmin is None:
            rmin = self.info['DLIML']
        if rmax is None:
            rmax = self.info['DLIMU']
        if mag_lim is None:
            mag_lim = self.info['MAGLIM']
        
        #select galaxies withing magnitude limit
        galaxies = self.galaxies
        galaxies = galaxies[galaxies['rabsmag'] < mag_lim]
        
        #select galaxies within survey mask
        if custom_mask_hdu is not None:
            mask = custom_mask_hdu.data
            mask_res = custom_mask_hdu.header['MSKRES']
            galaxies = select_mask(galaxies, mask, mask_res, rmin, rmax)
            
            # Cut galaxies down to those within 10 mpc/h of survey border
            # Note: we use the main survey mask rather than the custom mask option, 
            # because were ony worried about galaxies near edge voids
            points_boolean = np.zeros(len(galaxies), dtype = bool)

            #Flag points that fall outside the main survey mask
            for i in range(len(galaxies)):
                # The current point
                curr_pt = galaxies[i]

                is_edge = vo.is_edge_point(curr_pt['x'], curr_pt['y'], curr_pt['z'],
                                           mask, mask_res, rmin, rmax, self.edge_buffer)
                points_boolean[i] = not is_edge


            galaxies = galaxies[points_boolean]
        else:
            #This should never be the case (in current draft of code)
            print('V2 galaxy membership should have a custom mask to accurately exclude edge galaxies')
            assert 1==2
            galaxies = galaxies[(galaxies['Rgal'] >rmin)*(galaxies['Rgal'] < rmax)]
        
        
        #get indexes in full galaxy list of galaxies that make survey cuts
        in_sample = np.isin(self.galaxies['gal'], galaxies['gal'])
        # map the zonevoid void0 column onto every galaxy via the galaxies' zone membership
        # then cut down to the galaxies that make the survey cuts
        void0 = self.zonevoid[self.galzone['zone']]['void0'][in_sample]
        void0[self.galzone[in_sample]['zone']==-1] = -1 #remove galaxies that are not in zones (otherwise erroneously maped to self.zonevoid['void0'][-1])

        num_in_void = np.sum(void0!=-1)
        num_tot = len(void0)
        
        if return_selector:
            #cut the galzone 'gal' column down to the survey masked galaxies and the galaxies in voids
            selected_IDs = self.galzone['gal'][in_sample][void0!=-1]
            #get boolean mask of galaxies in final cut
            selector = np.isin(self.galaxies['gal'], selected_IDs)
            return ( selector, num_tot )

        return ( num_in_void, num_tot )


class VoidCatalogStacked ():
    
    # base class for loading multiple void catalogs at once 
    # (such as for surveys with mutliple contiguous footprints)

    def __init__ (self, edge_buffer):
        self.edge_buffer=edge_buffer
    
    def __getitem__(self, cat):
        return self._catalogs[cat]
    
    def lower_col_names(self):
        
        for cat in self._catalogs:
            self._catalogs[cat].lower_col_names()
        
    def upper_col_names(self):
        
        for cat in self._catalogs:
            self._catalogs[cat].upper_col_names()
                
    def add_galaxies(self, galaxies_paths, **kwargs):
        
        for cat, path in zip(self._catalogs, galaxies_paths):
            self._catalogs[cat].add_galaxies(path, **kwargs)
            
    def get_single_overlap(self, mask_hdu=None): 
        
        res = []
        
        for cat in self._catalogs:
            res.append(self._catalogs[cat].get_single_overlap(mask_hdu, self.edge_buffer))
            
        return res
            
    def void_stats(self):
        
        for cat in self._catalogs:
            print(cat)
            self._catalogs[cat].void_stats()
            print("")
        print("Combined")
                    
    def galaxy_membership(self, custom_mask_hdu=None):
        
        res = []
        for cat in self._catalogs:
            res.append(self._catalogs[cat].galaxy_membership(custom_mask_hdu, return_selector=False))
            
        return res
    
class VoidFinderCatalogStacked (VoidCatalogStacked):
    
    # Class for loading multiple VoidFinder catalogs at once 
    # (such as for surveys with mutliple contiguous footprints)

    def __init__ (self, cat_names, file_names, survey_names=None, directory = './', capitalize_colnames = False, edge_buffer=30):
    
        super().__init__(edge_buffer)
         
        if file_names is None:
            file_names = [directory + name + '_VoidFinder_Output.fits' for name in survey_names]
                    
        self._catalogs = {}
        
        for cat_name, file_name in zip(cat_names, file_names):
            self._catalogs[cat_name] = VoidFinderCatalog(file_name)
    
            
    def void_stats(self, report_individual=True):
        
        def filter_maximals(catalog):
            
            points_boolean = np.zeros(len(catalog.maximals), dtype = bool)
            
            mask = catalog.mask
            mask_res = catalog.mask_info['MSKRES']
            rmin = catalog.info['DLIML']
            rmax = catalog.info['DLIMU']

            #Remove voids near the survey edges
            for i in range(len(catalog.maximals)):
                # The current point
                curr_pt = catalog.maximals[i]

                is_edge = vo.is_edge_point(curr_pt['x'], curr_pt['y'], curr_pt['z'],
                                           mask, mask_res, rmin, rmax, self.edge_buffer)
                points_boolean[i] = not is_edge

            return catalog.maximals[points_boolean]
        
        if report_individual:
            super().void_stats()
        
        maximals = vstack([self._catalogs[cat].maximals for cat in self._catalogs])
        
        num_voids = len(maximals)
        print(num_voids, 'voids')
        edge = maximals['edge']
        print(len(maximals[edge==1]),'edge voids')
        print(len(maximals[edge==2]),'near-edge voids')
        print(len(maximals[edge==0]),'interior voids')
        
        maximals = vstack([filter_maximals(self._catalogs[cat]) for cat in self._catalogs])
        

        if np.product(np.isin(['r_eff','r_eff_uncert'], maximals.colnames)) > 0:
            
            """edge = maximals['edge']
            print(len(maximals[edge==1]),'edge voids (V. Fid)')
            print(len(maximals[edge==2]),'near-edge voids (V. Fid)')
            print(len(maximals[edge==0]),'interior voids (V. Fid)')"""

            reff = maximals['r_eff']
            uncert_mean = np.std(reff) / np.sqrt(num_voids)
            uncert_median = np.sqrt(np.pi / 2) * uncert_mean

            print('Mean Reff (V. Fid):', mknum(np.mean(reff)), '+/-',mknum(uncert_mean),'Mpc/h')
            print('Median Reff (V. Fid):', mknum(np.median(reff)), '+/-',mknum(uncert_median),'Mpc/h')
            print('Maximum Reff (V. Fid):', mknum(np.max(reff)),'Mpc/h')
        
    def calculate_r_eff(self, overwrite = False):
        
        for cat in self._catalogs:
            self._catalogs[cat].calculate_r_eff(overwrite)
            
    def calculate_vflag(self, overwrite = False, dist_metric = 'comoving'):
        
        for cat in self._catalogs:
            self._catalogs[cat].calculate_vflag(overwrite, dist_metric)
            
        
class V2CatalogStacked (VoidCatalogStacked):
    # Class for loading multiple V2 catalogs at once 
    # (such as for surveys with mutliple contiguous footprints)

    def __init__ (self, cat_names, file_names, survey_names=None,  pruning = 'VIDE', directory = './', edge_buffer=30):
        
        super().__init__(edge_buffer)
        
        #format file names
        if file_names is None:
            file_names = [directory + name + f'_V2_{pruning}_Output.fits' for name in survey_names]
             
        self._catalogs = {}
        
        for cat_name, file_name in zip(cat_names, file_names):
            self._catalogs[cat_name] = V2Catalog(file_name)
        
            
    """def add_mask(self, voidfinder_cat_stacked):
        
        for cat in self._catalogs:
            self._catalogs[cat].add_mask(voidfinder_cat_stacked[cat])"""
        
    def void_stats(self, report_individual=True):
        
        if report_individual:
            super().void_stats()
            
        def filter_voids(catalog):
            
            points_boolean = np.zeros(len(catalog.voids), dtype = bool)

            mask = catalog.mask
            mask_res = catalog.mask_info['MSKRES']
            rmin = catalog.info['DLIML']
            rmax = catalog.info['DLIMU']

            #Remove voids near the survey edges
            for i in range(len(catalog.voids)):
                # The current point
                curr_pt = catalog.voids[i]

                is_edge = vo.is_edge_point(curr_pt['x'], curr_pt['y'], curr_pt['z'],
                                           mask, mask_res, rmin, rmax, self.edge_buffer)
                points_boolean[i] = not is_edge

            return catalog.voids[points_boolean]
        
        voids = vstack([self._catalogs[cat].voids for cat in self._catalogs])
        num_voids = len(voids)
        print(num_voids, 'voids')
        
        if np.product(np.isin(['tot_area','edge_area'], voids.colnames))>0:
            edge_area = voids['edge_area']
            tot_area = voids['tot_area']
            edge = edge_area/tot_area > 0.1
            print(len(voids[edge]),'edge voids')
            print(len(voids[~edge]),'interior voids')
            
        voids = vstack([filter_voids(self._catalogs[cat]) for cat in self._catalogs])
        
        reff = voids['radius']
        uncert_mean = np.std(reff) / np.sqrt(num_voids)
        uncert_median = np.sqrt(np.pi / 2) * uncert_mean
        
        print('Mean Reff (V. Fid):', mknum(np.mean(reff)), '+/-',mknum(uncert_mean),'Mpc/h')
        print('Median Reff (V. Fid):', mknum(np.median(reff)), '+/-',mknum(uncert_median),'Mpc/h')
        print('Maximum Reff (V. Fid):', mknum(np.max(reff)),'Mpc/h')
        
    def calculate_r_eff(self, overwrite = False):
        
        for cat in self._catalogs:
            self._catalogs[cat].calculate_r_eff(overwrite)
            
    def calculate_vflag(self, overwrite = False, dist_metric = 'comoving'):
        
        for cat in self._catalogs:
            self._catalogs[cat].calculate_vflag(overwrite, dist_metric)
            
                
        
def mknum (flt):

    if flt is None:
        return None

    #preserve 3 sig figs for numbers starting with "0."
    if abs(flt) < 1:
        return float(f"{flt:.3g}")
    #otherwise round to two decimal places
    else:
        return float(f"{flt:.2f}")


def select_mask(gals, mask, mask_resolution, rmin, rmax, r_name = 'Rgal'):

    mask_checker = MaskChecker(0,
                            mask.astype(bool),
                            mask_resolution,
                            rmin,
                            rmax)
    
    points_boolean = np.zeros(len(gals), dtype = bool)
    
    #Flag points that fall outside the mask
    for i in range(len(gals)):
        # The current point
        curr_pt = np.array([gals['x'][i],gals['y'][i],gals['z'][i]])
        if gals[r_name][i] > 0:
            # Declare if point is not in mask
            not_in_mask = mask_checker.not_in_mask(curr_pt)
            # Invert not_in_mask to tag points in the mask
            points_boolean[i] = not bool(not_in_mask)
    return gals[points_boolean]

def combined_galaxy_membership(catalog1, catalog2, custom_mask_hdu=None):
    
    assert len(catalog1.galaxies)==len(catalog2.galaxies)
    
    rmin = max(catalog1.info['DLIML'], catalog2.info['DLIML'])
    rmax = min(catalog1.info['DLIMU'], catalog2.info['DLIMU'])
    mag_lim = min(catalog1.info['MAGLIM'], catalog2.info['MAGLIM'])

    
    selector1, num_tot1  = catalog1.galaxy_membership(custom_mask_hdu, return_selector=True,
                                                     rmin=rmin, rmax=rmax, mag_lim=mag_lim)
    selector2, num_tot2  = catalog2.galaxy_membership(custom_mask_hdu, return_selector=True,
                                                     rmin=rmin, rmax=rmax, mag_lim=mag_lim)
    
    
    assert num_tot1 == num_tot2
    
    num_void = np.sum(selector1*selector2)
    
    return num_void, num_tot1

def get_overlap(cat1, cat2, mask_hdu, edge_buffer):
    
    mask = mask_hdu.data.astype(bool) # convert from into to bool to avoid endian compiler error
    mask_res = mask_hdu.header['MSKRES']
    
    rmin = max(cat1.info['DLIML'], cat2.info['DLIML'])
    rmax = min(cat1.info['DLIMU'], cat2.info['DLIMU'])
    
    if isinstance(cat1, VoidFinderCatalog):
        cat1_type='VF'
        void1 = cat1.holes
        zone1 = None
    elif isinstance(cat1, V2Catalog):
        cat1_type='V2'
        void1 = cat1.galzone
        zone1 = cat1.zonevoid
    if isinstance(cat2, VoidFinderCatalog):
        cat2_type='VF'
        void2 = cat2.holes
        zone2 = None
    elif isinstance(cat2, V2Catalog):
        cat2_type='V2'
        void2 = cat2.galzone
        zone2 = cat2.zonevoid

    vooc = vo.OverlapCalculator(void1, void2, "Cat 1", "Cat 2",  
                                None,rmin, rmax, 
                                zone_table_V1 = zone1, zone_table_V2 = zone2, 
                                V1_algorithm=cat1_type, V2_algorithm=cat2_type,
                                mask_tuple=(mask.astype(bool), mask_res)
                                )
    vooc.find_overlap(edge_buffer)
    return vooc.report(do_print=False, do_return=True)

def combine_overlaps(overlaps, do_print=True, do_return=True):
    
    n_points = np.sum([overlap[0] for overlap in overlaps])
    n_V1_V2 = np.sum([overlap[1] for overlap in overlaps])
    n_not_V1_V2 = np.sum([overlap[2] for overlap in overlaps])
    n_V2_not_V1 = np.sum([overlap[3] for overlap in overlaps])
    n_V1_not_V2 = np.sum([overlap[4] for overlap in overlaps])
    
    if do_print:
    
        print('Shared volume:',n_V1_V2/n_points)
        print('Cat 1 volume:',(n_V1_V2+n_V1_not_V2)/n_points)
        print('Cat 2 volume:',(n_V1_V2+n_V2_not_V1)/n_points)
    
    if do_return:
        return (n_V1_V2/n_points, (n_V1_V2+n_V1_not_V2)/n_points, (n_V1_V2+n_V2_not_V1)/n_points, n_points )
   
#function still in development
def void_size_spectrum(catalog, save_every=None):
    
    gals = catalog.galaxies
    
    if isinstance(catalog, VoidFinderCatalog):
        voids = catalog.maximals
        mask = catalog.mask
        mask_resolution = catalog.info['MSKRES']
        dist_limits = [catalog.info['DLIML'], catalog.info['DLIMU']]
    else:
        assert 1==2 # not set up for V2 yet
    
    # get survey volume
    """volume, _ = sv.calculate_survey_volume(
                               gals['x'], gals['y'], gals['z'], 10,
                               mask, mask_resolution,
                               dist_limits, dist_limits, mask_trim=0, verbose = False)"""
    volume = 313637000.

    svdw = vswr.SWRadii(gals, voids, volume,
                    mask, mask_resolution, dist_limits,
                    )
    
    # test if voids has r_svdw
    if not np.sum(np.isin(['r_svdw','cen_den','thirty_fill'], voids.colnames))>0:
        voids['r_svdw'] = -1.
        voids['cen_den'] = -1.
        voids['thirty_fill'] = -1
                
    flags = np.arange(len(voids))[catalog.maximals['r_svdw']==-1]
    if save_every is not None:
        flags = [ flags[i*save_every:(i+1)*save_every] for i in range(int(np.ceil(len(flags)/save_every)))]
    else:
        flags = [flags]

    for flag_set in flags:
        
        r_svdw = svdw.run_SVdW(min_r = None, max_r = 50.,indexes=flag_set)
        cen_den = svdw.get_central_density(indexes=flag_set)
        
        
        vc = np.array([voids[flag_set]['x'],voids[flag_set]['y'],voids[flag_set]['z']]).T
        vfill = []
        for i in range(0, len(flag_set)):
            #Find radius of each void
            c = vc[i]
            vfill.append(svm.shell_fraction(
                            np.array([np.concatenate((c, [30, 0]))]),
                            mask = mask,
                            mask_resolution = mask_resolution,
                            dist_limits = dist_limits,
                            pts_per_unit_volume=0.1,
                           ))
        vfill = np.array(vfill).T[0] 
        
        for i, flag in enumerate(flag_set):
            voids[flag]['r_svdw'] = r_svdw[i]
            voids[flag]['cen_den'] = cen_den[i]
            voids[flag]['thirty_fill'] = vfill[i]
        
        if catalog.capitalize_colnames:
            catalog.upper_col_names()

        catalog._catalog['MAXIMALS'].data = fits.BinTableHDU(catalog.maximals).data
        catalog._catalog.writeto(catalog.file_name, overwrite=True)
        if catalog.capitalize_colnames:
            catalog.lower_col_names()