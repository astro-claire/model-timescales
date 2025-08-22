
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c
import numpy as np

def kinetic_energy(M: Quantity, V: Quantity, *, out_unit=u.J) -> Quantity:
    """K = 0.5 M V^2"""
    M = M.to(u.kg)
    V = V.to(u.m/u.s)
    return (0.5 * M * V**2).to(out_unit)

def gravitational_potential_energy(M: Quantity, R: Quantity, *,
                                   alpha=3/5, out_unit=u.J) -> Quantity:
    """
    U = -alpha * G M^2 / R
    alpha = 3/5 corresponds to a uniform-density sphere.
    """
    M = M.to(u.kg)
    R = R.to(u.m)
    return (-alpha * c.G * M**2 / R).to(out_unit)

def escape_velocity(M:Quantity,R: Quantity,*,
                    out_unit =u.km/u.s):
    M = M.to(u.kg)
    R = R.to(u.m)
    return np.sqrt(2*c.G*M/R)
