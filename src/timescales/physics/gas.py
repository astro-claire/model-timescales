"""
Gas physics for collision-driven gas accumulation and gas dynamical friction.

Implements the model described in the companion notes "Core collapse and gas effects":
a star cluster accumulates gas from stellar collisions, which can drive inward migration
via gas dynamical friction on the gas buildup timescale t_gb.

Three dynamical regimes are identified by comparing t_gb, t_SN, and t_GDF:
    1. t_gb > t_SN:                        gas buildup too slow to matter
    2. t_gb < t_SN,  t_GDF > t_gb:        gas important but runaway collapse avoided
    3. t_gb < t_SN,  t_GDF < t_gb:        runaway gas-driven core collapse
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as c
from .stars import stellar_radius_approximation


def gas_mass_per_collision(sigma, Mstar, Mcollisions):
    """
    Mass of gas released per stellar collision.

    M_g,coll = (M_* + M_c) * mu * sigma^2 / (G*M_c^2/R_c + G*M_*^2/R_*)

    The numerator mu*sigma^2 is twice the kinetic energy in the center-of-mass frame;
    the denominator is the sum of stellar binding energies. Their ratio gives the
    fraction of total mass unbound in the collision.

    Parameters
    ----------
    sigma : Quantity [velocity]
        Local velocity dispersion at the collision radius.
    Mstar : Quantity [mass]
        Mass of the target star (M_*).
    Mcollisions : Quantity [mass]
        Mass of the colliding/impacting star (M_c).

    Returns
    -------
    Quantity [mass]
        Gas mass released per collision, in solar masses.
    """
    mu = (Mstar * Mcollisions) / (Mstar + Mcollisions)
    R_star = stellar_radius_approximation(Mstar)
    R_coll = stellar_radius_approximation(Mcollisions)
    E_bind = c.G * (Mcollisions**2 / R_coll + Mstar**2 / R_star)
    return ((Mstar + Mcollisions) * mu * sigma**2 / E_bind).to(u.Msun)


def gas_buildup_timescale(t_coll, sigma, Mstar, Mcollisions):
    """
    Timescale for gas to reach 1% of the local stellar density (t_gb).

    Derived from t_gb = 0.01 * rho_*(r) * t_coll * 4*pi*r^2 / (M_g,coll * dN/dr).
    Since dN/dr = (rho_*/M_*) * 4*pi*r^2, the r^2 and rho_* factors cancel:

        t_gb = 0.01 * M_* * t_coll / M_g,coll(sigma)

    This is a local (per-radius) timescale. The minimum over all radii gives the
    fastest gas buildup in the system.

    Parameters
    ----------
    t_coll : Quantity [time]
        Local collision timescale at radius r.
    sigma : Quantity [velocity]
        Local velocity dispersion at radius r.
    Mstar : Quantity [mass]
        Stellar mass M_*.
    Mcollisions : Quantity [mass]
        Collider mass M_c.

    Returns
    -------
    Quantity [time]
        Gas buildup timescale in years.
    """
    Mg_coll = gas_mass_per_collision(sigma, Mstar, Mcollisions)
    return (0.01 * Mstar * t_coll / Mg_coll).to(u.yr)


def gas_dynamical_friction_timescale(sigma, Mstar, rho_gas):
    """
    Gas dynamical friction timescale (t_GDF).

    t_GDF = sigma^3 * M_* / (4*pi * (G*M_*)^2 * rho_gas)

    Derived from the GDF deceleration rate dv/dt = 4*pi*(G*M_*)^2*rho_g/sigma^2,
    setting t_GDF = v / |dv/dt| with v = sigma and I(M) = 1.

    Parameters
    ----------
    sigma : Quantity [velocity]
        Characteristic velocity dispersion of the system.
    Mstar : Quantity [mass]
        Stellar mass M_*.
    rho_gas : Quantity [mass / volume]
        Effective gas density driving the friction.

    Returns
    -------
    Quantity [time]
        Gas dynamical friction timescale in years.
    """
    return (sigma**3 * Mstar / (4 * np.pi * (c.G * Mstar)**2 * rho_gas)).to(u.yr)


def classify_gas_regime(t_gb, t_SN, t_GDF):
    """
    Classify a system into one of three gas-collapse regimes.

    Regimes:
        1 — t_gb > t_SN:                  gas buildup too slow; SN disperses gas first
        2 — t_gb < t_SN, t_GDF > t_gb:   gas matters but runaway collapse is avoided
        3 — t_gb < t_SN, t_GDF < t_gb:   runaway gas-driven core collapse

    Parameters
    ----------
    t_gb : Quantity [time]
        Minimum gas buildup timescale for the system.
    t_SN : Quantity [time]
        Supernova timescale (~10 Myr).
    t_GDF : Quantity [time]
        Gas dynamical friction timescale.

    Returns
    -------
    int
        Regime number (1, 2, or 3).
    """
    if t_gb >= t_SN:
        return 1
    return 2 if t_GDF > t_gb else 3
