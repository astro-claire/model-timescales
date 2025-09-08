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
