import astropy.units as u 
from .sampling import _generate_radii
from .factory import create_profile, available_profiles
from .factoryimf import create_imf, available_imfs
from typing import Dict, List, Optional, Callable, Union
from types import MethodType
import inspect
import warnings

class TimescaleEnsemble:
    def __init__(self,
                grid, 
                densityModel = "powerLaw", 
                Nsampling = 20, #these parameters determine the radii for each model's computations
                r_min_log10 = -3,
                imfModel = "salpeter",
                profile_kwargs: Optional[Dict] = None, #dict containing any parameters necessary for computation of the density, such as pl index
                timescales_kwargs: Optional[Dict] = None, #dict contatining any parameters necessary for computation of the timescales, such as mass of stars in cluster
                imf_kwargs: Optional[Dict] = None, #dict containing optional parameters for IMF
                alpha = None, # deprecated
                Mstar = None, #deprecated
                e = None #deprecated
                ):
        """
        Initializes a set of model systems and parameters for dynamical timescale calculation. 

        Args:
            grid (dict)
            densityModel (string): choice of density model 
        """
        self.grid = grid
        self.Nsystems = len(grid['R'])
        self.Nsampling = Nsampling

        #set the profile parameters. If none are given, default to power law profile. 
        if alpha is not None: 
            warnings.warn("DeprecationWarning: Adding alpha as keyword argument is deprecated. Use profile_kwargs instead.")
        if profile_kwargs:
            self.densityModel = densityModel
            self.profile_kwargs = dict(profile_kwargs)
            print("Using "+str(densityModel) + " model with properties:")
            for key in self.profile_kwargs.keys():
                print(str(key)+"=" +str(self.profile_kwargs[key]))
        else:
            print("No profile arguments given. Defaulting to power law with alpha = 1.25")
            self.densityModel = "powerLaw"
            self.profile_kwargs =  {"alpha":1.25}
        for key, value in self.profile_kwargs.items():
            setattr(self, key, value)

        #set parameters used for timescales calculations using the timescales kwargs:
        if Mstar is not None: 
            warnings.warn("DeprecationWarning: Adding Mstar as keyword argument is deprecated. Use timescales_kwargs instead.")
        if e is not None: 
            warnings.warn("DeprecationWarning: Adding e as keyword argument is deprecated. Use timescales_kwargs instead.")
        if timescales_kwargs:
            self.timescales_kwargs = dict(timescales_kwargs)
            print("Using parameters for timescale evaluation")
            for key in self.timescales_kwargs.keys():
                print(str(key)+"=" +str(self.timescales_kwargs[key]))
        else:
            print("No timescale arguments given. Defaulting to eccentricity 0, Mstar 1Msun.")
            self.timescales_kwargs= {"e":0, "Mstar":1*u.Msun}
        for key, value in self.timescales_kwargs.items():
            setattr(self, key, value)


        #Stellar IMF: 
        if imf_kwargs:
            self.imf_kwargs = dict(imf_kwargs)
            self.imfModel = imfModel
            print("Using parameters for IMF")
            for key in self.imf_kwargs.keys():
                print(str(key)+"=" +str(self.imf_kwargs[key]))
        else:
            print(f"No imf arguments given. Defaulting to '{imfModel}'.")
            self.imfModel = imfModel
            self.imf_kwargs= {}
        for key, value in self.imf_kwargs.items():
            setattr(self, key, value)
        try:
            self.imf = create_imf(
                self.imfModel,
                **self.imf_kwargs,
            )
        except KeyError as er:
            # Helpful error if someone passes an unknown model name
            raise ValueError(
                f'Unknown imf "{self.imfModel}". '
                f"Available: {', '.join(available_imfs())}"
            ) from er

        self.radii = _generate_radii(grid,self.Nsystems, Nsampling = Nsampling, rMin = r_min_log10)


        # 2) Build a profile instance per system via the factory
        self.profiles = []
        for i in range(self.Nsystems):
            M_i = grid["M"][i]
            R_i = grid["R"][i]
            if "V" in grid.keys():
                V_i = grid["V"][i]
            else:
                V_i = None

            try:
                prof = create_profile(
                    self.densityModel,           # e.g., "powerlaw"
                    # For power-law, normalize using mass within R:
                    M_ref=M_i,
                    R_ref=R_i,
                    V_c = V_i,
                    r0=R_i,
                    **self.profile_kwargs,
                )
            except KeyError as er:
                # Helpful error if someone passes an unknown model name
                raise ValueError(
                    f'Unknown density profile "{densityModel}". '
                    f"Available: {', '.join(available_profiles())}"
                ) from er

            self.profiles.append(prof)

        # 3) Compute structural fields per system (rho, Menc, sigma) now (eager)
        self.rho = []
        self.Menc = []
        self.sigma = []
        self.n = []
        for i, prof in enumerate(self.profiles):
            r_i = self.radii[i]
            rho_i = prof.density(r_i)
            Menc_i = prof.enclosed_mass(r_i)
            sigma_i = prof.velocity_dispersion(r_i)

            self.rho.append(rho_i)
            self.Menc.append(Menc_i)
            self.sigma.append(sigma_i)
            # Generic number density (TODO allow for other ways to calcualte this)
            self.n.append(rho_i / self.Mstar)
            

    def __str__(self):
        """
        Returns a human-readable string.
        This method is called by the `print()` function.
        """
        return f"timescaleModel object with mass range: [{min(self.grid['M'])}, {max(self.grid['M'])}]; velocity range: [{min(self.grid['V'])}, {max(self.grid['V'])}; radius range: [{min(self.grid['R'])}, {max(self.grid['R'])}]]"

