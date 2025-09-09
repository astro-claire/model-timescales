#src/timescales/physics.py 
"""
Functions for calculating the Coulomb logarithm

"""
import astropy.units as u 
import astropy.constants as c 
import numpy as np
from typing import Float, Optional
from astropy.units import Quantity
from ..utils.units import as_quantity

def coulomb_log_BH(r, sigma,*, 
                    Mstar: Optional[Quantity] =1*u.Msun,
                    m_units = u.Msun,
                    r_unit = u.pc, 
                    sigma_units = u.km/u.s):
    """ 
    Calculate the coulomb log for a galactic center following Naoz & Sari review eqns 15-16
    ln(Lambda) = ln(bmax/bmin)= log (r sigma^2/2Gm)
    Returns
    -------
    (Quantity or Array) Coulomb Logarithm (log(Lambda))
    """
    r = as_quantity(r, r_unit)
    sigma = as_quantity(sigma, sigma_units)
    Mstar = as_quantity(Mstar, m_units)
    return np.log(r * sigma**2 / 2 / c.G / Mstar)

def coulomb_log(M, *, 
                Mstar: Optional[Quantity] =1*u.Msun,
                coulomb_little_lambda: Optional[Float]  = 0.1,
                m_units = u.Msun):
    """ 
    Calculate the coulomb log following Hamilton et  al 2018 framework / Binney and Tremaine

    log(Lambda) = log( lambda* N) 

    Returns
    -------
    (Quantity or Array) Coulomb Logarithm (log(Lambda))
    """
    M = as_quantity(M, m_units)
    Mstar = as_quantity(Mstar, m_units)

    return np.log(coulomb_little_lambda * M/Mstar)