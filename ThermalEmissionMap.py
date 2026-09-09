import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import matplotlib.patheffects as pe

figsize_2column_wide = (7.1,5)
figsize_2column_long = (7.1,9)
figsize_1column = (3.35,5)

class ThermalEmissionMap:
    """
    A Python class to work with 2D thermal emission data in HEALPix format. 
    This class provides methods to load, manipulate, and visualize thermal emission data from planetary bodies.
    """
    def __init__(self, radiance_map, wavelengths, nside, planet_name="Unknown"):
        self.planet_name = planet_name # For labeling purposes in plots
        sorted_indices = np.argsort(wavelengths)
        self.radiance =  radiance_map[sorted_indices,:]*u.W/(u.m**2 * u.micron * u.sr)  # shape (n_lambda, npix)
        self.wavelengths = wavelengths[sorted_indices] * u.micron  # shape (n_lambda,)
        self.nside = nside # HEALPix NSIDE parameter
        self.npix = hp.nside2npix(nside) 
        self.dOmega = np.ones(self.npix) * 4 * np.pi / self.npix  * u.steradian # sr per pixel
        self.longitude,self.co_lat = hp.pix2ang(self.nside,np.arange(0,self.npix,1),lonlat=True)

    @classmethod
    def from_lonlat(cls, longitude, latitude, wavelengths, radiance, nside=64, planet_name="Unknown"):
        """
        Converts longitude and latitude data into a HEALPix map representation.

        Parameters:
        ----------
        longitude : array-like
            An array of longitude values in degrees.
        latitude : array-like
            An array of latitude values in degrees.
        wavelengths : array-like
            An array of wavelength values corresponding to the radiance measurements.
        radiance : array-like
            An array of radiance values corresponding to the given wavelengths.
        nside : int, optional
            The HEALPix nside parameter, which determines the resolution of the map (default is 64).
        planet_name : str, optional
            The name of the planet for which the data is being processed (default is "Unknown").

        Returns:
        -------
        cls
            An instance of the class containing the HEALPix map and associated metadata.
        """
        hp_map, wl = cls.load_to_healpix(longitude, latitude, wavelengths, radiance, nside)
        return cls(hp_map, wl, nside, planet_name=planet_name)

    @classmethod
    def from_healpix(cls, radiance_map, wavelengths, nside, planet_name="Unknown"):
        """
        Create a ThermalEmission object directly from HEALPix-formatted data.

        Parameters:
        -----------
        radiance_map : ndarray
            Radiance in shape (n_lambda, npix), in W/m²/µm/sr or similar units.
        wavelengths : ndarray
            Wavelengths in meters (shape n_lambda).
        nside : int
            HEALPix NSIDE resolution (used to define npix).
        planet_name : str
            Name of the object (e.g., "Earth", "Moon").

        Returns:
        --------
        ThermalEmission
            Initialized object.
        """
        npix = hp.nside2npix(nside)
        if radiance_map.shape[1] != npix:
            raise ValueError(f"radiance_map has {radiance_map.shape[1]} pixels, expected {npix} for nside={nside}")
        
        return cls(radiance_map, wavelengths, nside, planet_name=planet_name)

    def save(self, filepath):
        """
        Saves the thermal emission data to a pickle file.
        
        Parameters:
        - filepath: str
            Path to the file where the data will be saved.
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Thermal emission data saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Loads the thermal emission data from a pickle file.
        
        Parameters:
        - filepath: str
            Path to the file from which the data will be loaded.
        
        Returns:
        - ThermalEmission
            The loaded ThermalEmission object.
        """
        import pickle
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Thermal emission data loaded from {filepath}")
        return obj
    

    @staticmethod
    def load_to_healpix(longitude, latitude, wavelengths, radiance, nside=64):
        """
        Convert (lon, lat, radiance) data to a HEALPix map.

        Parameters:
        -----------
        longitude : array-like
            Longitudes in degrees
        latitude : array-like
            Latitudes in degrees
        wavelengths : array-like
            Wavelengths 
        radiance : array-like
            Radiance (n_lambda, npix)
        nside : int
            HEALPix NSIDE

        Returns:
        --------
        hp_map : ndarray
            Radiance in HEALPix format, shape (n_lambda, npix)
        wavelengths : ndarray
            Wavelength array (unchanged)
        """
        n_lambda, N = radiance.shape
        npix = hp.nside2npix(nside)

        # Convert to radians
        lon_rad = np.radians(longitude)
        lat_rad = np.radians(latitude)
        theta = 0.5 * np.pi - lat_rad  # co-latitude
        phi = lon_rad

        # Get pixel indices
        pix = hp.ang2pix(nside, theta, phi)

        # Initialize map
        hp_map = np.zeros((n_lambda, npix))
        counts = np.zeros(npix)

        for i in range(N):
            hp_map[:, pix[i]] += radiance[:, i]
            counts[pix[i]] += 1

        # Avoid division by zero
        counts[counts == 0] = 1
        hp_map /= counts

        return hp_map, wavelengths
    def B_fitting(self,lamb,T):
        """
        Planck function.

        Parameters:
        -----------
        lamb : float
            Wavelength in meters.
        T : float
            Temperature in Kelvin.
        
        Returns:
        --------
        B : float
            Spectral radiance in W/m2/um/sr.
        """
        import astropy.constants as c
        B = (2*c.h.value*c.c.value**2/((lamb)**5))/(np.exp(c.h.value*c.c.value/((lamb)*c.k_B.value*T))-1)
        return B

    def disk_integrated_flux(self, phi_interest=0.0,theta= np.pi/2,limb_darkening=True):
        """
        Calculate the disk-integrated flux at a given sub-observer longitude (phi_interest).

        Parameters:
        -----------
        phi_interest : float or str
            Sub-observer longitude in radians. If 'sphere', computes flux over the entire sphere.
            For example, 
                phi_interest = 0.0   for dayside 
                phi_interest = np.pi for nightside.
        theta : float
            Sub-observer co-latitude in radians (default is equator, np.pi/2).
        limb_darkening : bool
            If True, apply limb-darkening correction based on the planet type.
            For planet_name = 'Earth',     use (1 + 0.09 * log(mu)),
            For planet_name = 'Moon_day'   use mu^0.06
            For Planet_name = 'Moon_night' use mu^0.14.
            else no limb-darkening correction is applied.
        Returns:
        --------
        flux_lambda : ndarray
            Disk-integrated flux at each wavelength in W/m2/um.
        """
        if isinstance(phi_interest, float):
            theta = theta  # equator
            phi = phi_interest
            observer_vec = hp.ang2vec(theta, phi)
            pixel_vecs = hp.pix2vec(self.nside, np.arange(self.npix))
            mu = np.dot(observer_vec, pixel_vecs)  # cosine of angle

            visible = mu > 0
            mu = np.where(visible, mu, 0.0)
            self.mu_disk = mu  # Store mu for later use
            if limb_darkening:
                if self.planet_name == "Earth":
                    # Use limb-darkening correction for Earth
                    limb_darkening = (1 + 0.09 * np.log(mu))
                elif self.planet_name == "Moon_day" or self.planet_name == "Moon":
                    limb_darkening = (mu)**(0.06)
                elif self.planet_name == "Moon_night":
                    limb_darkening = (mu)**(0.14)
            else:
                limb_darkening = 1.0
            flux_lambda = np.nansum(self.radiance * mu * self.dOmega * limb_darkening, axis=1)  # shape: (n_lambda,)

        elif phi_interest == 'sphere':
            # For full sphere, use all pixels
            # no limb darkening correction for full sphere
            flux_lambda = np.mean(self.radiance, axis=1)*np.pi*u.sr
        else:
            raise ValueError("phi_interest must be a float or 'sphere'")
        return flux_lambda 

    def bolometric_flux(self, phi_interest=0.0):
        """
        Compute the bolometric flux by integrating the disk-integrated flux over all avialable wavelengths.

        Parameters:
        -----------
        phi_interest : float or str
            Sub-observer longitude in radians. If 'sphere', computes flux over the entire sphere.
            Same as in disk_integrated_flux method.

        Returns:
        -------
        flux_total : float
            Bolometric flux in W/m2.
        """
        if isinstance(phi_interest, float) or phi_interest == 'sphere':
            ### check units of self.wavelengths and self.radiance

            flux_lambda = self.disk_integrated_flux(phi_interest)
            wavelengths = self.wavelengths.to('um')
            # Integrate using trapezoidal rule
            flux_total = np.trapezoid(flux_lambda, wavelengths)
        else:
            raise ValueError("phi_interest must be a float or 'sphere'")
        return flux_total
    
    def e_factor(self,method='Morris'):
        """
        Calculate the e-factor (heat redistribution efficiency) using the method of Morris et al. (2021) or Cowan & Agol (2011).

        Parameters:
        -----------
        method : str
            Method to use for calculating e-factor. Options are 'Morris' or 'Cowan'.

        Returns:
        -------
        e : float
            Heat redistribution efficiency.
        """
        nightside_flux = self.disk_integrated_flux(phi_interest=np.pi)
        dayside_flux = self.disk_integrated_flux(phi_interest=0.0)
        C = np.trapezoid(nightside_flux,self.wavelengths)/np.trapezoid(dayside_flux,self.wavelengths)
        if method=='Morris':
            e = C
        elif method=='Cowan':
            e = 8*C/(5*C+3)
        else:
            print ("method must be 'Morris' (Morris et al. 2021) or 'Cowan' (Cowan & Agol 2011)")
        return e
    
    def f_factor(self):
        """
        Calculate the f-factor (heat redistribution factor) based on the e-factor.
        f = ((8/5) - e) * (5/12)
        Returns:
        -------
        f : float
            Heat redistribution factor.
        """
        e_factor = self.e_factor()
        return ((8./5.)-e_factor)*(5./12.)
    
    def find_bightness_temperature(self, wl_wanted, rad_wanted):
        """
        Find the brightness temperature at a specific wavelength of interest.

        Parameters:
        -----------
        wl_wanted : float
            Wavelength of interest in microns.
        rad_wanted : float
            Radiance at the specified wavelength in W/m2/um/sr.

        Returns:
        -------
        Tb : float
            Brightness temperature at the specified wavelength in Kelvin.
        """
        rad = rad_wanted.to('W/(m2 m sr)')*u.sr 
        wavelengths_m = wl_wanted.to('m') 

        # Invert the Planck function:
        a = 2 * c.h * c.c**2
        b = c.h * c.c / c.k_B

        with np.errstate(divide='ignore', invalid='ignore'):
            Tb = (b / (wavelengths_m * np.log((a / (wavelengths_m**5 * rad)) + 1))).T

        Tb = np.real(Tb)
        Tb[np.isnan(Tb)] = 0  # Optional: zero out unphysical values
        return Tb
    
    def radiance_to_brightness_temperature(self):
        """
        Convert the radiance map to brightness temperature using the inverse Planck function for each pixel and wavelength.

        Returns:
        -------
        Tb_raw : ndarray
            Brightness temperature map in Kelvin, shape (n_lambda, npix).
        """
        import astropy.constants as c
        import numpy as np
        Tbs = []
        for i in range (self.npix):
            Tb = self.find_bightness_temperature(self.wavelengths, self.radiance[:, i])
            Tbs.append(Tb.decompose())
        self.Tb_raw = np.array(Tbs).T  
        return self.Tb_raw

    def disk_integrated_brightness_temperature(self, phi_interest=0.0,theta= np.pi/2,limb_darkening=True):
        """
        Calculate the disk-integrated brightness temperature at a given sub-observer longitude (phi_interest).

        Parameters:
        -----------
        phi_interest : float or str
            Sub-observer longitude in radians. If 'sphere', computes flux over the entire sphere.
        theta : float
            Sub-observer co-latitude in radians (default is equator, np.pi/2).
        limb_darkening : bool
            If True, apply limb-darkening correction based on the planet type.
            For planet_name = 'Earth',     use (1 + 0.09 * log(mu)),
            For planet_name = 'Moon_day'   use mu^0.06
            For Planet_name = 'Moon_night' use mu^0.14.
            else no limb-darkening correction is applied.

        Returns:
        -------
        Tb_disk : ndarray
            Disk-integrated brightness temperature at each wavelength in Kelvin, shape (n_lambda,).
        """
        flux = self.disk_integrated_flux(phi_interest,theta,limb_darkening)
        wavelengths_m = self.wavelengths.to('m') 
        Tb_disk = []
        for i in range(len(wavelengths_m)):
            lam = wavelengths_m[i]
            
            a = 2 * c.h * c.c**2
            b = c.h * c.c / c.k_B
            with np.errstate(divide='ignore', invalid='ignore'):
                Tb = (b / (lam * np.log((a / (lam**5 * flux[i]/np.pi)) + 1)))
            Tb = np.real(Tb)
            Tb_disk.append(Tb.to('K').value)
        self.Tb_disk = np.array(Tb_disk) 
        return self.Tb_disk

    def fit_bolometric_brightness_temperature(self):
        """
        Fit the planck spectrum to the spectrum of each pixel to find the bolometric brightness temperature. Usually used for temperature map of the planet.

        Returns:
        -------
        Tb_bolometric : ndarray
            Bolometric brightness temperature map in Kelvin, shape (npix,).
        """
        if not hasattr(self, 'Tb_map'):

            from scipy.optimize import curve_fit
            wavelengths_m = self.wavelengths.to('m').value  
            Tb_map = np.zeros(self.npix)
            rad_m = self.radiance.to('W / (m2 m sr)').value 

            for i in range(self.npix):
                spectrum = rad_m[:, i]  
                if np.all(spectrum == 0) or np.any(spectrum < 0):
                    Tb_map[i] = 0.0
                    continue
                try:
                    popt, _ = curve_fit(
                        lambda wl, T: self.B_fitting(wl, T),
                        wavelengths_m, spectrum, p0=[300], bounds=(10, 2000)
                    )
                    Tb_map[i] = popt[0]
                except RuntimeError:
                    Tb_map[i] = 0.0

            self.Tb_map = Tb_map
            return self.Tb_map
        else:
            print ("Bolometric brightness temperature map already computed. Use self.Tb_map to access it.")
            return self.Tb_map
    
    def plot_brightness_temperature_map(self, projection="mollweide",Tmax = 1,**kwargs):
        """
        Plot the brightness temperature map using HEALPix projections.

        Parameters:
        -----------
        projection : str
            Type of projection to use for the plot. Options are 'mollweide' or 'gnomonic'.
        Tmax : float
            Maximum temperature for normalization in the plot.

        Returns:
        -------
        Tb : ndarray
            Brightness temperature map in Kelvin, shape (npix,).
        """
        if not hasattr(self, 'Tb_map'):
            self.fit_bolometric_brightness_temperature()

        self.R_map = self.Tb_map/Tmax
        
        title = f"Bolometric Temperature map ({self.planet_name})"
        f = plt.figure(figsize=figsize_2column_wide)
        projview(
                    self.R_map,
                    coord="C",
                    graticule=True,
                    graticule_labels=False,
                    unit=r"$R = T_b/T_{max}$",
                    xlabel="solar longitude",
                    ylabel="latitude",
                    cb_orientation="vertical",
                    projection_type=projection,
                    title=title,flip='geo',
                    cmap='magma',nest=False,hold=True,
                    fig=f,
                    **kwargs,
                    # shrink
                    
        )
        plot_graticule_label(size=10,color='white',path_effects=[pe.withStroke(linewidth=1, foreground="k")])
        return self.R_map

    def plot_radiance_map(self, wl_idx=None, wl=None, projection="mollweide", rot=(0, 0),plot=True,**kwargs):
        """
        Plot the radiance map at a specific wavelength index or wavelength.

        Parameters:
        -----------
        wl_idx : int, optional
            Index of the wavelength to plot. If provided, this takes precedence over wl.
        wl : float, optional
            Wavelength in microns to plot. If provided, the closest wavelength index will be used.
        projection : str
            Type of projection to use for the plot. Options are 'mollweide', 'gnomonic', or 'orthographic'.
        rot : tuple
            Rotation parameters for the projection. Default is (0, 0).
        plot : bool
            If True, the map will be plotted. If False, the function will return the radiance map without plotting.

        Returns:
        -------
        figure : matplotlib.figure.Figure
        """
        
        if wl is not None and wl_idx is not None:
            raise ValueError("Specify either wl or wl_idx, not both.")
        if wl_idx is None and wl is None:
            raise ValueError("Specify either wl or wl_idx.")
        if wl_idx is not None:
            wl_idx = wl_idx
        elif wl is not None:
            wl_idx = np.argmin(np.abs(self.wavelengths - wl))

        print (f"we will plot map at wavelength index = {wl_idx} corresponding to wavelength = {self.wavelengths[wl_idx].value:.2f} µm")
        
        my_dpi = 100
        f,(ax1,ax2) = plt.subplots(nrows=2,figsize=figsize_2column_wide,height_ratios=[1,7],sharex=False,constrained_layout=True)
        
        try:
            Tb_raw = self.Tb_raw
        except AttributeError:
            Tb_raw = self.radiance_to_brightness_temperature()
        
        m = Tb_raw[wl_idx] 

        mean_spectrum = np.mean(Tb_raw, axis=1)
        spectrum_line, = ax1.plot(self.wavelengths, mean_spectrum)
        indicator_line = ax1.axvline(self.wavelengths[wl_idx].value, color='r', linestyle='--')
        ax1.set_xlabel(f"Wavelength ($\mu$m)")
        ax1.set_ylabel(f"$T_b$ ($K$)")

        ax1.set_title("Global Spectrum")

        plt.axes(ax2)

        projview(
                    m,
                    coord="C",
                    graticule=True,
                    graticule_labels=False,
                    unit=f"$T_b$ ($K$)",
                    xlabel="solar longitude",
                    ylabel="latitude",
                    cb_orientation="vertical",
                    projection_type=projection,
                    title=f'{self.planet_name} at {self.wavelengths[wl_idx].value:.2f} $\mu$m ',flip='geo',
                    cmap='magma',nest=False,hold=True,
                    # shrink
                    fig=f.number,sub=(2,1,2),
                    **kwargs
        )
        
        plot_graticule_label(size=10,color='white',path_effects=[pe.withStroke(linewidth=1, foreground="k")])
        return f

    def plot_spectrum(self, phi_interest=0.0,**kwargs):
        """
        Plot the disk-integrated spectrum at a given sub-observer longitude (phi_interest).

        Parameters:
        -----------
        phi_interest : float or str
            Sub-observer longitude in radians. If 'sphere', computes flux over the entire sphere.
            Same as in disk_integrated_flux method.
        Returns:
        -------
        flux : ndarray
            Disk-integrated flux at each wavelength in W/m2/um.
        """
        if type(phi_interest) is float:
            flux = self.disk_integrated_flux(phi_interest)
            plt.plot(self.wavelengths, flux,label=f"{self.planet_name} disk spectrum ({phi_interest:.2f} rad)",**kwargs)
            return flux
        elif phi_interest == 'sphere':
            flux_lambda_total = self.disk_integrated_flux(phi_interest)
            plt.plot(self.wavelengths, flux_lambda_total, label=f"{self.planet_name}'s total spectrum",**kwargs)
            return flux_lambda_total
        else:
            raise ValueError("phi_interest must be a float or 'sphere'")
        plt.xlabel("Wavelength [um]")
        plt.ylabel("Flux [W/m2/um]")

    def calculate_phase_curve(self, phase_angles):
        """
        Calculate the phase curve over a range of phase angles.
        -----------
        phase_angles : array-like
            Array of phase angles = sub-observer longitudes in radians.
        Returns:
        -------
        phase_curve : ndarray
            Array of instrument-convolved fluxes at each phase angle.
        """
        phase_curve = []
        for angle in phase_angles:
            flux = self.disk_integrated_flux(phi_interest=angle)
            phase_curve.append(flux.value)
        output = np.array(phase_curve)*u.W/(u.m**2*u.micron)
        return output

def plot_graticule_label(**kwargs):
        
        """
        Plot graticule labels for the map.
        Parameters:
        -----------
        **kwargs : dict
            Additional keyword arguments for text properties (e.g., fontsize, color).
        
        Returns:
        -------
        None
        """
        plt.text(0,0,s = f"0$\degree$",**kwargs)
        plt.text(1,0,s = f"60$\degree$",**kwargs)
        plt.text(2,0,s = f"120$\degree$",**kwargs)
        plt.text(-1.1,0,s = f"-60$\degree$",**kwargs)
        plt.text(-2.2,0,s = f"-120$\degree$",**kwargs)
        plt.text(-3.3,0,s = f"0$\degree$",**kwargs)
        plt.text(-3.4,0.5,s = f"30$\degree$",**kwargs)
        plt.text(-3.4,1.0,s = f"60$\degree$",**kwargs)
        plt.text(-3.6,-0.6,s = f"-30$\degree$",**kwargs)
        plt.text(-4.0,-1.1,s = f"-60$\degree$",**kwargs)






