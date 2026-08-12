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


def gas_buildup_timescale(t_coll, sigma, Mstar, Mcollisions, f = 0.01):
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
    return (f * Mstar * t_coll / Mg_coll).to(u.yr)


def gas_dynamical_friction_timescale(sigma, Mstar, rho_gas, IM=1):
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
    return (sigma**3 * Mstar / (4 * np.pi * (c.G * Mstar)**2 * rho_gas *IM)).to(u.yr)


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

# def CD(M):
#     M = np.asarray(M, dtype=float)
#     result = np.where(M < 1, 0.47 * (1 + 0.27 * M**2), 2.0)
#     return result
def IM(M):
    M = np.asarray(M, dtype=float)
    result = np.zeros_like(M)
    
    subsonic = M < 1
    supersonic = M > 1
    sonic = M == 1
    
    result[subsonic] = 0.5 * np.log((1 + M[subsonic]) / (1 - M[subsonic])) - M[subsonic]
    result[supersonic] = 0.5 * np.log((1 + M[supersonic]) / (M[supersonic] - 1))
    
    if np.any(sonic):
        print("WARNING: Mach 1 is undefined; setting to NaN")
        result[sonic] = np.nan
    
    return result

def CD(Mach,  v, T, mu,Kn=1):
    """
    Drag coefficient C_d for a sphere, covering free-molecular and continuum
    limits across subsonic and supersonic regimes.

    Subsonic, free-molecular (Kn >> 1): Epstein (1924) drag, recast as an
        effective C_d by dividing the Epstein force by (1/2 rho v^2 pi R^2).
        F_Epstein = (4pi/3) * R^2 * rho * v_th * v * (1 + 9pi/64)
        => C_d = (8/3) * (v_th / v) * (1 + 9pi/64)

    Subsonic, continuum (Kn << 1): C_d = 0.47 (high-Re sphere, Batchelor 1967)

    Supersonic (either regime): C_d = 2 (Rephaeli & Salpeter 1980)

    The two subsonic branches are blended via a smooth Knudsen-number weight
    so there is no discontinuity at intermediate Kn.

    Parameters
    ----------
    Mach : array-like, dimensionless Mach number
    Kn   : array-like, Knudsen number = lambda_mfp / R_star
    v    : astropy Quantity array, velocity of the star
    T    : astropy Quantity, gas temperature (for thermal speed)
    mu   : float, mean molecular weight

    Returns
    -------
    C_d : ndarray, dimensionless
    """
    Mach = np.asarray(Mach, dtype=float)
    Kn   = np.asarray(Kn,   dtype=float)

    # thermal speed of gas particles
    v_th = np.sqrt(8 * c.k_B * T / (np.pi * mu * c.m_p)).to(u.cm / u.s)
    v_cgs = v.to(u.cm / u.s).value

    # Epstein C_d (free-molecular subsonic limit)
    Cd_epstein   = (8 / 3) * (v_th.value / v_cgs) * (1 + 9 * np.pi / 64)

    # Continuum C_d (high-Re sphere, subsonic)
    Cd_continuum = 0.47 * np.ones_like(Mach)

    # Blend: weight = sigmoid in log(Kn), transition at Kn ~ 1
    # w -> 1 (free-molecular) for Kn >> 1, w -> 0 (continuum) for Kn << 1
    w = 1 / (1 + np.exp(-2 * np.log(Kn)))

    Cd_subsonic = w * Cd_epstein + (1 - w) * Cd_continuum

    # Supersonic: C_d = 2 everywhere (both regimes converge here)
    Cd = np.where(Mach >= 1, 2.0, Cd_subsonic)

    return Cd

def aerodynamic_drag_timescale(sigma, Mstar, rho_gas, M = 2, T=1e4*u.K, mu = 1.4, Kn = 1):
    R_star = stellar_radius_approximation(Mstar)
    C_D = CD(M, sigma, T, mu, Kn=Kn)
    return 2 * Mstar / (rho_gas * sigma * C_D * (np.pi * R_star**2))



def sound_speed(T, mu = 0.5):
    """Adiabatic sound speed for ideal monatomic gas."""
    gamma = 5/3
    return np.sqrt(gamma * c.k_B * T / (mu * c.m_p)).to(u.km/u.s)

def aerodynamic_2phase_drag_timescale(sigma, Mstar, rho_gas1, T1, M1, mu1,rho_gas2, T2, M2, mu2, f):
    f2 = (1-f)
    F_aero_bh1 = 0.5*CD(M1, sigma,T1,mu1)*np.pi*(stellar_radius_approximation(Mstar))**2 * rho_gas1 * sigma**2
    F_aero_bh2 = 0.5*CD(M2, sigma,T2,mu2)*np.pi*(stellar_radius_approximation(Mstar))**2 * rho_gas2 * sigma**2
    F_aero_bh = ( F_aero_bh1) + (f2 * F_aero_bh2)
    return (sigma/(F_aero_bh/Mstar)).to('yr')

def gas_2phase_dynamical_friction_timescale(sigma, Mstar, rho_gas1, T1, M1, mu1,rho_gas2, T2, M2, mu2, f):
    f2 = (1-f)

    F_gdf_bh1  = 4.*np.pi*(c.G*Mstar)**2 * rho_gas1 / (sigma**2) * IM(M1)
    F_gdf_bh2  = 4.*np.pi*(c.G*Mstar)**2 * rho_gas2 / (sigma**2) * IM(M2)
    F_gdf_bh = ( F_gdf_bh1) + (f2 * F_gdf_bh2)
    return (sigma/(F_gdf_bh/Mstar)).to('yr')
