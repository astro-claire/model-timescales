#timescales/physics/dynamical_friction
import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c


def sphere_of_influence(alpha,r0,rho0,MBH):
    cm= 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))

    return (MBH/cm)**(1./(3.-alpha))