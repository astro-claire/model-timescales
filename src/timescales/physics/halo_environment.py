#src/timescales/physics/halo_environment.py 
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u 
import astropy.constants as c
import numpy as np
import importlib.resources as ir

from .registry import register_timescale
from ..utils.units import as_quantity

@register_timescale("t_int", aliases = ("interaction", "halo-merger","halo-interaction"))
def interaction_timescale(M, R, z, halomass, cosmo,v= 16.6 *u.km/u.s, N = 7, d = 10,fstar_DM=0.1,M_upper=10, fixed_N= True): 
    """
    n sigma v calculation
    d is unitless - comoving for conversion to physical 
    """
    #make sure units are all good
    d = d * u.kpc / 0.71 * (1./(1+z))
    v = as_quantity(v, u.km/u.s)
    M = as_quantity(M, u.Msun)
    halomass = as_quantity(halomass, u.Msun)
    R = as_quantity(R, u.pc)
    #first get n 
    f =  get_densityFraction(halomass,M_upper=M_upper) # Get the fraction between halomass and M_upper* halomass according to the ST function
    if fixed_N:
        N=N
    else:
        N = get_N(M)
    stmasses = np.logspace(4,11,10000)
    normalization_offset= get_normalization()
    sigma_norm = normalization_offset[closest_idx(stmasses, halomass.to_value('Msun'))][0] #correct for over clustered box
    n = N / (4./3. * np.pi * d**3) *f /sigma_norm
    #then get sigma 
    r = R+get_rvirz(halomass, z,cosmo).to('pc')
    sigma = np.pi * r**2
    #finally, gamma
    gamma = n*sigma*v
    return (1./gamma)

@register_timescale("t_neighbor",aliases=("neighbor-interaction","neighbor-merger"))
def neighbor_merger_timescale(M,R,z,halomass,cosmo, v= 16.6 *u.km/u.s,fstar_DM=0.1,M_upper=10, fixed_N= True):
    """ 
    calculate the interaction timescale for the nearest neighbor using the simulation parameters from Williams + 25
    """
    result = interaction_timescale(M, R, z, halomass, cosmo,v= v, N =1, d = 1.5,fstar_DM=fstar_DM,M_upper=M_upper, fixed_N= fixed_N)
    return result

@register_timescale("t_local",aliases=("local-interaction","local-merger"))
def local_merger_timescale(M,R,z,halomass,cosmo, v= 16.6 *u.km/u.s,fstar_DM=0.1,M_upper=10, fixed_N= True):
    """ 
    calculate the interaction timescale for the local environemntusing the simulation parameters from Williams + 25
    """
    result = interaction_timescale(M, R, z, halomass, cosmo,v= v, N =6, d = 10,fstar_DM=fstar_DM,M_upper=M_upper, fixed_N= fixed_N)
    return result


def closest(lst, K):     
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def closest_idx(lst,K):
    return np.where(lst==closest(lst,K))[0]

def get_STcounts():
    """
    Get the Sheth & Tormen number counts from the data file
    """
    with ir.files("timescales.cosmodata").joinpath("numGreaterThanM_s8_17new.txt").open("r") as f:
        stfunction17 = np.loadtxt(f)
    return stfunction17

def get_densityFraction(M,  M_upper=10):
    """
    compute fraction of 
    """
    stfunction17 = get_STcounts()
    stmasses = np.logspace(4, 11, 10000)

    # Ensure M can be converted to a NumPy array for uniform operations
    M = np.atleast_1d(M.to('Msun').value)  # Convert M to a NumPy array in solar masses

    # Compute M_u as an array
    M_u = M * M_upper

    # Use vectorized operations to calculate indices and N_M
    idx_M = np.searchsorted(stmasses, M, side='left')
    idx_M_u = np.searchsorted(stmasses, M_u, side='left')

    # Calculate N_M using the indices
    N_M = stfunction17[idx_M] - stfunction17[idx_M_u]

    # Calculate N_tot
    N_tot = stfunction17[0]

    # Compute and return the density fraction (handles scalar or array return automatically)
    result = N_M / N_tot

    # Return a scalar if input was scalar; otherwise, return the array
    return result if result.size > 1 else result[0]


def get_N(M):
    M= M.to('Msun').value
    a0 = -2.84*10**(-13)
    a1 = 1.74*10**(-6)
    a2 = 5.67
    return (a0*M**2) + ( a1 * M ) + a2

def get_delta(z,cosmo): 
    d = cosmo.Om(z)-1
    return (18* np.pi **2 ) + (82* d ) - (39* d**2)

def get_rvirz(M_DM, z,cosmo):
    """
    Virial radius of  DM halo that hosts it)
    """
    om_z = cosmo.Om(z)
    deltac = get_delta(z,cosmo)
    h = (cosmo.H0/100).value
    mterm = (M_DM/ (1e8 *h * u.Msun))**(1./3.)
    cosmoterm = (cosmo.Om0 / om_z * deltac/(18*np.pi**2))**(-1/3.)
    zterm =((1+z)/10.)**(-1)
    return 0.784 * mterm * cosmoterm * zterm * h**(-1) * u.kpc

def get_normalization():
    """ 
    Normalization conversion from sigma 8 = 1.7 to 0.8
    """
    # print("Normalization conversion between sigma8 = 1.7 and 0.8")
    with ir.files("timescales.cosmodata").joinpath("numGreaterThanM_s8_17new.txt").open("r") as f:
        stfunction17 = np.loadtxt(f)
    with ir.files("timescales.cosmodata").joinpath("numGreaterThanM_s8_08new.txt").open("r") as f:
        stfunction = np.loadtxt(f)
    stfunction = np.array(stfunction)
    stfunction17 = np.array(stfunction17)
    normalization_offset = stfunction17/stfunction
    return normalization_offset

    
def closest(lst, K):     
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def closest_idx(lst,K):
    return np.where(lst==closest(lst,K))[0]