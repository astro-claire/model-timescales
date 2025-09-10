#timescales/physics/dynamical_friction
import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c
from ..utils import as_quantity
from .registry import register_timescale

@register_timescale("t_df", aliases=("dynamical-friction",))
def dynamical_friction_timescale(M_cl,M_BH,r_h,*, 
                            mass_units = u.Msun,
                            radius_units = u.pc):
    """
    Calculate the dynamical friction timescale (Chandrasekhar 1943 timescale, quoted in Fragione & Rasio 2023)\
    
    Parameters
    ----------
    M_cl (astropy quantity): Cluster Mass
    M_BH (astropy quantity): black hole mass (or mass of sinking object)
    r_h (astropy quantity): half mass radius
    """
    M_cl = as_quantity(M_cl,mass_units)
    M_BH = as_quantity(M_BH,mass_units)
    r_h = as_quantity(r_h, radius_units)
    return 20 * u.Myr * (20.*u.Msun/ M_BH) * (M_cl/ (1e5 *u.Msun))**0.5 * (r_h/(1*u.pc))**(3./2.)