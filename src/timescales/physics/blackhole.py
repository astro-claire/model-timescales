#timescales/physics/dynamical_friction
import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c
from .stars import stellar_radius_approximation


def sphere_of_influence(alpha,r0,rho0,MBH):
    cm= 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))

    return (MBH/cm)**(1./(3.-alpha))

def tidal_radius(MBH, Mstar):
    return (stellar_radius_approximation(Mstar)*(MBH/Mstar)**(1./3.)).to('pc')


def bondi_accretion_rate(sigma, M_BH, rho,*, M_enc=0.*u.Msun,Mgas =0.*u.Msun, gamma = 5./3.):
    """assuming the sigma is related to the sound speed as below."""
    cs = np.sqrt(gamma/3 * sigma**2)
    Mdot = np.pi * rho * c.G**2 * M_BH**2 / cs**3
    return Mdot.to(u.g/u.s)

def eddington_rate(M_BH):
    return 1e-8 *M_BH.to('Msun')/ u.yr