
import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c
from ..utils import as_quantity
import scipy
from .stars import stellar_radius_approximation

def collision_timescale(n, v, Mstar,*,
                        e = 0.,
                        alpha = 1.25,
                        Mcollisions = 1 * u.Msun,
                        n_unit=1./u.cm**3,
                        v_unit=u.cm/u.s,
                        Mstar_unit = u.Msun) -> Quantity:
    """
    Calcualtes collision timescale following rose et al 2023 eqn 4

    Parameters
    ----------
    rho : float | array-like | Quantity
        Mass density (default unit: cm^-3 unless a Quantity is provided).
    v : float | array-like | Quantity
        Relative speed (default unit: cm/s unless a Quantity is provided).
    rc : float | array-like | Quantity


    Keyword-only units let you choose a different assumed unit system
    for plain numbers (e.g., set n_unit=1/u.m**3, sigma_unit=u.m**2, v_unit=u.m/u.s).

    Returns
    -------
    t : Quantity
        Collision timescale with time units (seconds).
    """
    n = as_quantity(n, n_unit)
    # sigma = _as_quantity(sigma, sigma_unit)
    v = as_quantity(v, v_unit)
    Mstar = as_quantity(Mstar,Mstar_unit)
    Mcollisions = as_quantity(Mcollisions, Mstar_unit)
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)

    #functions from Rose et al 2020
    f1 = (1.-e)**(0.5-alpha)*0.5*scipy.special.hyp2f1(
        0.5,alpha -0.5, 1., 2.*e/(e-1.)
        ) + (1.+e)**(0.5-alpha)*0.5*scipy.special.hyp2f1(
            0.5,alpha -0.5, 1., 2.*e/(e+1.)
            )

    f2 = (1.-e)**(1.5-alpha)*0.5*scipy.special.hyp2f1(
        0.5,alpha -1.5, 1., 2.*e/(e-1.)) + (1.+e)**(1.5-alpha)*0.5*scipy.special.hyp2f1(
            0.5,alpha -1.5, 1., 2.*e/(e+1.))

    #rate - rose et al 2023
    rate = (np.pi * n * v * (

        (f1*rc**2)   +   (f2 * rc * (2* c.G * (Mstar+Mcollisions) / (v**2)) ).to('cm**2')

        )).to(1/u.s)          # broadcast-safe


    with np.errstate(divide='ignore', invalid='ignore'):
        t = (1.0 / rate).to(u.yr)

    # Optionally set exact zeros to inf (cleaner than warnings)
    zero_mask = (rate.value == 0)
    if np.any(zero_mask):
        t = t.copy()
        t.value[zero_mask] = np.inf

    return t
