#src/timescales/physics/stars.py

import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c

def stellar_radius_approximation(mass, *, mass_unit=u.M_sun) -> Quantity:
    """
    Approximate stellar radius from stellar mass using a piecewise power law.

        R / R_sun = (M / M_sun)^alpha,
        where alpha = 0.8 for M < 1 M_sun, and alpha = 0.57 for M >= 1 M_sun.

    Parameters
    ----------
    mass : float, array-like, or `~astropy.units.Quantity`
        Stellar mass(es). If unitless, `mass_unit` is assumed.
    mass_unit : `~astropy.units.Unit`, optional
        Unit to assume for unitless `mass`. Default: M_sun.

    Returns
    -------
    radius : `~astropy.units.Quantity`
        Stellar radius(es) in solar radii.

    Notes
    -----
    - Works with scalars or arrays (NumPy broadcasting).
    - Non-positive masses return NaN radii.
    """
    # Coerce to Quantity
    if isinstance(mass, Quantity):
        m_solar = mass.to_value(u.M_sun)
    else:
        m_solar = (np.asarray(mass) * mass_unit).to_value(u.M_sun)

    # Piecewise exponent
    alpha = np.where(m_solar < 1.0, 0.8, 0.57)

    # Guard against non-physical inputs
    bad = ~np.isfinite(m_solar) | (m_solar <= 0)
    m_safe = np.where(bad, np.nan, m_solar)

    radius = (m_safe ** alpha) * u.R_sun
    return radius
