#src/timescales/physics/halo_environment.py 
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u 
import astropy.constants as c
import numpy as np
from .registry import register_timescale

@register_timescale("t_int", aliases = ("interaction", "halo-merger","halo-interaction"))
def get_tinteract(M,rho, z, halomass, v= 16.6 *u.km/u.s, N = 7, d = 10, cosmo = cosmo,fstar_DM=0.1,M_upper=10, fixed_N= True): 
    """
    n sigma v calculation
    d is unitless for conversion to in person 
    """
    #first get n 
    d = d * u.kpc / 0.71 * (1./(1+z))
    f =  get_densityFraction(halomass,M_upper=M_upper)
    if fixed_N:
        N=N
    else:
        N = get_N(M)
    stmasses = np.logspace(4,11,10000)
    sigma_norm = normalization_offset[closest_idx(stmasses, halomass)][0]
    n = N / (4./3. * np.pi * d**3) *f *sigma_norm
    #then get sigma 
    r = get_rvirz(M, z,cosmo, fstar_DM= fstar_DM)+get_rvirz(halomass, z,cosmo, fstar_DM= fstar_DM)
    sigma = np.pi * r**2
    #finally, gamma
    gamma = n*sigma*v
    return (1./gamma)


def closest(lst, K):     
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def closest_idx(lst,K):
    return np.where(lst==closest(lst,K))[0]

def get_STcounts():
    stfunction17 = np.loadtxt('../cosmodata/numGreaterThanM_s8_17new.txt')
    return stfunction17

# stfunction17 = get_STcounts()
# stmasses = np.logspace(4,11,10000)

def get_densityFraction(M,  M_upper=10):
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

def get_rvirz(M_dm, z,cosmo, fstar_DM = 0.1):

    Mtot = Mstar + Mstar/fstar_DM
    om_z = cosmo.Om(z)
    deltac = get_delta(z,cosmo)
    h = (cosmo.H0/100).value
    mterm = (Mtot/ (1e8 *h * u.Msun))**(1./3.)
    cosmoterm = (cosmo.Om0 / om_z * deltac/(18*np.pi**2))**(-1/3.)
    zterm =((1+z)/10.)**(-1)
    return 0.784 * mterm * cosmoterm * zterm * h**(-1) * u.kpc

def get_vcz(M,z,cosmo):
    return np.sqrt(c.G * M / get_rvirz(M,z,cosmo)).cgs

def get_tdynz(M,z,cosmo):
    v = get_vcz(M,z,cosmo)
    r = get_rvirz(M,z,cosmo)
    return (r/v).to('Myr')

def get_normalization():
    print("Normalization conversion between sigma8 = 1.7 and 0.8")
    stfunction = np.loadtxt('../cosmodata/numGreaterThanM_s8_08new.txt')
    stfunction17 = np.loadtxt('../cosmodata/numGreaterThanM_s8_17new.txt')
    normalization_offset = stfunction17/stfunction
    return normalization_offset

    
def closest(lst, K):     
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def closest_idx(lst,K):
    return np.where(lst==closest(lst,K))[0]