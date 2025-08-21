# src/timescales/sampling.py

import astropy.units as u
import numpy as np

def _generate_radii(grid,Nsystems, Nsampling = 20, rMin = -3):
    """
    creates logarithmically spaced arrays of the interior radii for all systems in the grid

    Args: 
        rMin (float): minimum radius to sample in log10(r/pc). Default -3.
    """
    radii = np.empty(Nsystems, dtype=object)
    for i in range(Nsystems):
        maxrad = grid['R'][i].to('pc')
        radii[i] = np.logspace(rMin,np.log10(maxrad.value),Nsampling) *u.pc
    return radii
