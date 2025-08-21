import astropy.units as u
from astropy.units import Quantity
import numpy as np

def as_quantity(x, unit: u.UnitBase) -> Quantity:
    """
    Coerce x to an astropy Quantity with the given unit.
    - If x already has units, it's converted.
    - If x is unitless (float/int/ndarray), the provided `unit` is assumed.
    """
    if isinstance(x, Quantity):
        return x.to(unit)
    return np.asarray(x) * unit