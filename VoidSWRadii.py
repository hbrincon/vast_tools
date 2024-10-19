# Authors: Hernan Rincon, Dahlia Veyrat

import numpy as np
from scipy.spatial import cKDTree
import healpy as hp
import ShellVolumeMaskedPython as svm
from scipy.interpolate import InterpolatedUnivariateSpline as int_uni_spl


def sky_fraction(gals, csm0, z_limit, min_maximal_radius=10.0, smooth_mask=True):
    #Calculates the fraction of the sky occupied by a survey
    
    D2R = np.pi/180.0
    maskra = 360
    maskdec = 180
    
    ra  = gals['ra'].data % 360.0
    
    dec = gals['dec'].data
    
    r_max = csm0.comoving_distance([z_limit]).value[0]


    mask_resolution = 1 + int(D2R*r_max/min_maximal_radius) # scalar value despite value of r_max
    ############################################################################

    
    ############################################################################

    num_px = maskra * maskdec * mask_resolution ** 2
    nside = 2**np.math.floor(np.math.log2(np.sqrt(num_px / 12)))
    healpix_mask = np.zeros(hp.nside2npix(nside), dtype = bool)
    galaxy_pixels = hp.ang2pix(nside, ra, dec, lonlat = True)
    healpix_mask[galaxy_pixels] = 1
    
    if smooth_mask:
        
        #Note: the [::2] selects only neighbors that share edges with the cell in question
        neighbors = hp.get_all_neighbours(nside,np.arange(len(healpix_mask)))[::2]
        correct_idxs = np.sum(healpix_mask[neighbors], axis=0)
        healpix_mask[np.where(correct_idxs >= 3)] = 1

    #return fractional sky coverage and the mask used to caclulate it
    return (len(np.where(healpix_mask==True)[0])/len(healpix_mask), healpix_mask)

class SWRadii():

    def __init__ (self, galdata, vdata, sv,
                  mask=None, mask_resolution=None, dist_limits=None,xyz_limits=None,
                  mask_type="ra_dec_z", number_density = None, number_density_comov = None):
        # galdata (Table object with named columns): galaxy coordinates
        # vdata (Table object with named columns): void center coordinates
        # sv (float): total survey volume in (Mpc/h)^3 - computing this depends on shape, etc

        # load in catalogs
        self.galdata = galdata
        gx = galdata['x']
        gy = galdata['y']
        gz = galdata['z']
        gc = np.array([gx,gy,gz]).T
        self.vdata = vdata
        vx = vdata['x']
        vy = vdata['y']
        vz = vdata['z']
        self.vc = np.array([vx,vy,vz]).T

        self.num_voids = len(vdata['x'])

        # mean density of galaxies/tracers
        if number_density is not None and number_density_comov is not None:
            spline = int_uni_spl(number_density_comov, number_density)
            comov = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            self.num_density = spline(comov)
        else:
            self.num_density = None
        self.md = len(gc) / sv
        
        # set minimum radius for valid VdW radius to 
        # 2 times the mean galaxy separation
        self.min_r = 2 * self.md ** (-1/3)

        # K-D tree of galaxies/tracers
        self.kdt = cKDTree(gc)
        
        # mask info
        self.mask = mask
        self.mask_resolution = mask_resolution
        self.dist_limits = dist_limits
        self.xyz_limits = xyz_limits
        self.mask_type = mask_type
    
    def run_SVdW(self, dr = -0.7, min_r = 10, max_r = 45., indexes = None):
        # Sheth-van de Weygaert radius function (for all voids in catalog)
        # dr : desired threshold density relative to mean; commonly 0.7
        # returns radii, the radius
        
        #The starting sphere radius
        #TODO: make this vary with redshift
        #print (2*self.md ** (-1/3))
        #calculate radii for all voids if indexs not specified
        if indexes is None:
            indexes = range(0, self.num_voids)
        #Array of void radii, initialized to 0
        radii = []
        #Loop over void catalog
        if min_r is None:
                if self.num_density is not None:
                    for i in indexes:
                        min_r = 2 * self.num_density[i] ** (-1/3)
                        #Find radius of each void
                        radii.append(self.grow_SVdW_sphere(self.vc[i], min_r = min_r, max_r = max_r))       
                else:
                    min_r = 2*self.md ** (-1/3)
                    for i in indexes:
                        #Find radius of each void
                        radii.append(self.grow_SVdW_sphere(self.vc[i], min_r = min_r, max_r = max_r))
        else:
            for i in indexes:
                #Find radius of each void
                radii.append(self.grow_SVdW_sphere(self.vc[i], min_r = min_r, max_r = max_r))

        #return void radii
        return radii
    
    def grow_SVdW_sphere(self, c, dr = -0.7, min_r = 10, max_r = 45.):
        
        dr = 1 + dr #write the underdensity threshold as an addition to the mean
        
        #check that SVdW sphere is larger than the minimum allowed size
        sphere_radius = min_r
        N = len(self.kdt.query_ball_point(c, sphere_radius))
        V = 4/3 * np.pi * sphere_radius**3
        vol_frac = self.get_volume_fraction(c, sphere_radius, ppuv = 100/V)
        if vol_frac == 0:
            return -3
        V *= vol_frac
        density = N/V
        #if SVdW sphere is smaller than minimum allowed size return -1
        if density >= self.md*dr:
            return -1
        
        #grow the sphere until the SVdW radius is reached
        
        N+=1
        sphere_radius, idx = self.kdt.query(c,k=[N])
        sphere_radius = sphere_radius[0]
        V = 4/3 * np.pi * sphere_radius**3
        vol_frac = self.get_volume_fraction(c, sphere_radius, ppuv = 100/V)
        if vol_frac == 0:
            return -3
        V *= vol_frac
        density = N/V        
        
        while sphere_radius < max_r:
            
            if density >= self.md*dr:
                return sphere_radius
            
            N+=1
            sphere_radius, idx = self.kdt.query(c,k=[N])
            sphere_radius = sphere_radius[0]
            V = 4/3 * np.pi * sphere_radius**3
            vol_frac = self.get_volume_fraction(c, sphere_radius, ppuv = 100/V)
            if vol_frac == 0:
                return -3
            V *= vol_frac
            density = N/V
            
        #return -2 for spheres exceeding maximum allowed size
        return -2
        
            
    def getIndividualR(self,c, dr = -0.7, dd = 0.01, R = 25.,Rmin=10, Rmax=45):
        # Sheth-van de Weygaert radius function (for individual void)
        # c  : coordinates of a void center
        # dr : desired threshold density relative to mean; commonly -0.7
        # dd : desired precision of the radius, ideally less than 0.01
        # R  : starting radius, could be higher than 100
        # returns R, the radius

        dr = 1 + dr #write the underdensity threshold as an addition to the mean

        dR = R/2. #The sphere-size incrementation severity
        
            
        # grow sphere  
        while dR>dd:
            
            # reject voids outside of size range of interest
            if R+2*dR < Rmin:
                return -1
            if R-2*dR > Rmax:
                return -2
            
            N = len(self.kdt.query_ball_point(c,R)) #the number of tracers in the spehere
            V = 4 / 3 * np.pi * R ** 3 #the sphere's volume
            
            vol_frac = self.get_volume_fraction(c, R, ppuv = 100/V)
            if vol_frac == 0:
                return -3
            V=V*vol_frac
            d = N/V #tracer number density within sphere
            
            
            # if the tracer density exceeds the mean density times the threshold fraction
            if d>self.md*dr:
                R = R-dR #shrink the sphere
            else:
                R = R+dR #grow the sphere
            dR = dR/2.
        return R

    def getR(self, dr = -0.7,dd = 0.01, R=25.,Rmin=10, Rmax=45):
        # Sheth-van de Weygaert radius function (foll all voids in catalog)
        # dr : desired threshold density relative to mean; commonly 0.7
        # dd : desired precision of the radius, ideally less than 0.01
        # returns radii, the radius
        
        #The starting sphere radius
        print (2*self.md ** (-1/3))
        #Array of void radii, initialized to 0
        radii = np.zeros(self.num_voids)
        #Loop over void catalog
        if Rmin is None:
                if self.num_density is not None:
                    for i in range(0, self.num_voids):
                        Rmin = 2 * self.num_density[i] ** (-1/3)
                        #Find radius of each void
                        radii[i] = self.getIndividualR(self.vc[i], dr, dd, R, Rmin = Rmin, Rmax = Rmax)        
                else:
                    Rmin = 2*self.md ** (-1/3)
                    for i in range(0, self.num_voids):
                        #Find radius of each void
                        radii[i] = self.getIndividualR(self.vc[i], dr, dd, R, Rmin = Rmin, Rmax = Rmax)   
        else:
            for i in range(0, self.num_voids):
                #Find radius of each void
                radii[i] = self.getIndividualR(self.vc[i], dr, dd, R, Rmin = Rmin, Rmax = Rmax)   

        #return void radii
        return radii

    
    def getC(self,  num = 5):
        # central density in volume enclosing specified amount of nearest neighbors
 
        distances, idx = self.kdt.query(self.vc, k=num) #the number of tracers in the sphere
        distances = distances[:,-1]
        cenden = np.zeros(self.num_voids)
        
        for i in range(0, self.num_voids):
            cenden[i] = self.getIndividualC(self.vc[i], distances[i], num)
            
        return cenden
            
    def getIndividualC(self, c, R, N):
        
        if R < 0:
            return R
        
        V = 4/3 * np.pi * R**3
        vol_frac = self.get_volume_fraction(c, R, ppuv = 100/V)
        if vol_frac == 0:
            return -3
        V*=vol_frac
        
        return N/V
    
    def getDaR(self, R):
        # "Get Density at Radius"
        #density in given volume
        
        cenden = np.zeros(self.num_voids)
        #Loop over void catalog
        for i in range(0, self.num_voids):
            #Find radius of each void
            cenden[i] = self.getIndividualDaR(self.vc[i], R)
        return cenden
    
    def get_central_density(self, indexes = None):
        # "Get Density at Radius"
        #density in given volume
        
        min_r = self.min_r
        
        #Loop over void catalog
        if indexes is None:
            indexes = range(self.num_voids)
        cenden = []
        for i in indexes:
            #Find radius of each void
            c = self.vc[i]
            if self.num_density is not None:
                min_r = 2 * self.num_density[i] ** (-1/3)
                
            N = self.kdt.query_ball_point(c, min_r, return_length = True) #the number of tracers in the sphere
            
            if N == 0:
                N = 1
            R = self.kdt.query(c, k=[N])[0][0] #adjust the sphere size to break discretization

            V = 4/3 * np.pi * R ** 3 #the sphere's volume

            vol_frac = self.get_volume_fraction(c, R, ppuv = 100/V)

            if vol_frac == 0:
                cenden.append(-3)
            else:
                V=V*vol_frac

                cenden.append(N/V) #tracer number density within sphere
              
        return cenden
    
    def getDaRarray(self, radii, ratio ):
        # "Get Density at Radius<array>"
        # density in unique given volume for each void
        # This is the central density defined in CBL when
        #     radii = the SvDW radii
        #     ratio < 1
        
        scaled_radii = radii * ratio
        
        cenden = np.zeros(self.num_voids)
        #Loop over void catalog
        for i in range(0, self.num_voids):
            #Find radius of each void
            cenden[i] = self.getIndividualDaR(self.vc[i], scaled_radii[i])
        return cenden
    
    def getIndividualDaR(self,c, R):
        
        if R < 0:
            return R
        
        N = len(self.kdt.query_ball_point(c,R)) #the number of tracers in the sphere
        V = 4/3 * np.pi * R ** 3 #the sphere's volume

        vol_frac = self.get_volume_fraction(c, R, ppuv = 100/V)
        if vol_frac == 0:
            return -3
        V=V*vol_frac
        
        return N/V #tracer number density within sphere
    
    def get_volume_fraction(self, c, R, Rmin=0, ppuv = 0.1):
        
        vol_frac = svm.shell_fraction(
           np.array([np.concatenate((c, [R, Rmin]))]),
           mask=self.mask, 
           mask_resolution=self.mask_resolution,
           dist_limits=self.dist_limits,
           xyz_limits=self.xyz_limits,
           pts_per_unit_volume=ppuv,
           mask_type=self.mask_type
           )[0]
        
        return vol_frac