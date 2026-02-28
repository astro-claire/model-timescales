import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c
from ..utils import as_quantity
from .registry import register_timescale

@register_timescale("t_relax", aliases=("relaxation",))
def relaxation_timescale(v,rho, mass, coulomb=10, v_unit = u.km/u.s, rho_unit = u.g/u.cm**3,mass_unit = u.Msun):
    
    v = as_quantity(v,v_unit)
    rho = as_quantity(rho, rho_unit)
    mass = as_quantity(mass, mass_unit)

    return (0.34 * v**3 / (c.G**2 * rho * mass *coulomb)).to('yr')



def r_no_relax(rho0,r0, Mstar, coulomb, cv, alpha):
    """ 
    Radius at which
    Relaxation time = orbital period
    """
    num = 8. *np.pi * 0.34 * cv**(3./2.) *rho0 * r0**alpha
    denom = (1.+alpha)**(3./2.)* (3.-alpha)**2. * Mstar * coulomb

    return ((num/denom)**(1./(alpha-3.))).to('pc')

def r_no_relax_bh(rho0,r0, Mstar, coulomb, cv, alpha, MBH):
    """ 
    Radius at which
    Relaxation time = orbital period
    Assuming you're inside the BH sphere of influence
    """
    box = 0.34 * cv**(3./2.) / 2. / np.pi / (1.+alpha)**(3./2.) / Mstar  / coulomb
    num = box * MBH**2 
    denom = rho0*r0**alpha
    return ((num/denom)**(1./(3.-alpha))).to('pc')
