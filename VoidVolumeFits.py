import numpy as np

from vast.voidfinder.postprocessing import open_fits_file
from astropy.table import Table
from astropy.io import fits
import VoidVolume as vol

def volume_from_fits(out_directory, survey_name):
    
    catalog, log_filename = open_fits_file(out_directory, survey_name)
    maxs = Table(catalog['MAXIMALS'].data)
    holes=Table(catalog['HOLES'].data)
    maxs['EFFECTIVE_RADIUS']=-1.
    maxs['EFFECTIVE_RADIUS_UNCERT']=-1.
    maxs['EFFECTIVE_RADIUS'].unit = 'Mpc/h'
    maxs['EFFECTIVE_RADIUS_UNCERT'].unit = 'Mpc/h'

    for flag in maxs['FLAG']:
        hole_in_max = holes[holess['FLAG']==flag]
        x = np.array([holes['X'], holes['Y'],holes['Z']]).T
        R = holes['RADIUS'].data
        vol_info = vol.volume_of_spheres(x, R)
        maxs['EFFECTIVE_RADIUS'][flag] = ((3/4) * vol_info[2] / np.pi) ** (1/3) 
        maxs['EFFECTIVE_RADIUS_UNCERT'][flag] = vol_info[3] * ((3 * vol_info[2]) ** -2 / (4 * np.pi)) ** (1/3) 

    catalog['MAXIMALS'].data = fits.BinTableHDU(maxs).data
    hdul.writeto(log_filename, overwrite=True)

