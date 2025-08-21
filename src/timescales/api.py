import astropy.units as u 
from .sampling import _generate_radii
from .factory import create_profile, available_profiles
from typing import Dict, List, Optional

class TimescaleEnsemble:
    def __init__(self,
                grid, 
                densityModel = "powerLaw", 
                alpha = 1.25,
                Nsampling = 20,
                Mstar = 1*u.Msun,
                profile_kwargs: Optional[Dict] = None,
                r_min_log10 = -3
                ):
        """
        Initializes a set of model systems and parameters for dynamical timescale calculation. 

        Args:
            grid (dict)
            densityModel (string): choice of density model 
            Mstar = mass of the particles/stars in the cluster (default 1 Msun)
        """
        self.grid = grid
        self.Nsystems = len(grid['R'])
        self.Nsampling = Nsampling
        self.alpha = alpha
        self.Mstar = Mstar
        self.radii = _generate_radii(grid,self.Nsystems, Nsampling = Nsampling, rMin = r_min_log10)
        self.profile_kwargs = {} if profile_kwargs is None else dict(profile_kwargs)


        # 1) Sample radii per system (ragged: each system gets its own array)
        # self.radii: List[Quantity] = _generate_radii(
        #     grid["R"], Nsampling=Nsampling, rMin=r_min_log10
        # )

        # 2) Build a profile instance per system via the factory
        self.profiles = []
        for i in range(self.Nsystems):
            M_i = grid["M"][i]
            R_i = grid["R"][i]

            try:
                prof = create_profile(
                    densityModel,           # e.g., "powerlaw"
                    alpha=self.alpha,  # shared param
                    # For power-law, normalize using mass within R:
                    M_ref=M_i,
                    R_ref=R_i,
                    r0=R_i,            # common choice for reference radius
                    **self.profile_kwargs,
                )
            except KeyError as e:
                # Helpful error if someone passes an unknown model name
                raise ValueError(
                    f'Unknown density profile "{densityModel}". '
                    f"Available: {', '.join(available_profiles())}"
                ) from e

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
            # Generic number density (override if your model defines it differently)
            self.n.append(rho_i / self.Mstar)

        # (Optional) example: compute a timescale immediately
        # self.t_relax = [relaxation_timescale(s, rh, self.Mstar) for s, rh in zip(self.sigma, self.rho)]

    def __str__(self):
        """
        Returns a human-readable string.
        This method is called by the `print()` function.
        """
        return f"timescaleModel object with mass range: [{min(self.grid['M'])}, {max(self.grid['M'])}]; velocity range: [{min(self.grid['V'])}, {max(self.grid['V'])}; radius range: [{min(self.grid['R'])}, {max(self.grid['R'])}]]"

