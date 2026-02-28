#timescales/physics/dynamical_friction
import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.constants as c
from ..utils import as_quantity
from .registry import register_timescale
from .relaxation import relaxation_timescale

@register_timescale("t_df_b", aliases=("bulk_dynamical-friction",))
def bulk_dynamical_friction_timescale(M_cl,M_BH,r_h,*, 
                            mass_units = u.Msun,
                            radius_units = u.pc):
    """
    Calculate the dynamical friction timescale (Chandrasekhar 1943 timescale, quoted in Fragione & Rasio 2023)
    
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

@register_timescale("t_df", aliases = ("dynamical_friction","local_dynamical_friction"))
def dynamical_friction_timescale(v, rho,*,
                                    M_obj = 10*u.Msun,
                                    Mstar = 1 * u.Msun,
                                    coulomb=10, 
                                    v_unit = u.km/u.s, 
                                    rho_unit = u.g/u.cm**3,
                                    mass_unit = u.Msun):
    """ 
    Dynamical friction timescale at a given radius (local) 
    """
    M_obj = as_quantity(M_obj, mass_unit)
    Mstar = as_quantity(Mstar, mass_unit)
    massratio = Mstar/M_obj
    t_relax = relaxation_timescale(v,rho, Mstar, coulomb=10, v_unit = u.km/u.s, rho_unit = u.g/u.cm**3,mass_unit = u.Msun)
    return t_relax * massratio


def stellar_df_time(r, Mstar, lnLambda, Mcollisions, alpha, r0, rho0, cv=1.0):
    """
    t_df(r) in the stellar-dominated regime: M(r) >> MBH.

    """
    G = c.G

    # --- CGS ---
    r     = r.to(u.cm)
    r0    = r0.to(u.cm)
    rho0  = rho0.to(u.g/u.cm**3)
    Mstar = Mstar.to(u.g)
    Mi    = Mcollisions.to(u.g)
    Gc    = G.cgs

    c_rho = (rho0 * r0**alpha).to(u.g * u.cm**(alpha - 3))
    c_M   = (4*np.pi * c_rho / (3 - alpha)).to(u.g * u.cm**(alpha - 3))

    q = (Mi / Mstar).decompose().value

    pref  = (0.34 * q) / (Gc**2 * Mstar * lnLambda)
    scale = ((cv * Gc) / (1 + alpha))**(3/2) * (c_M**(3/2) / c_rho)

    tdf = (pref * scale * r**(3 - alpha/2)).to(u.yr)
    return tdf


def bh_df_time(r, Mstar, lnLambda, Mcollisions, alpha, r0, rho0, MBH, cv=1.0):
    """
    t_df(r) in the BH-dominated regime: MBH >> M(r).
    """
    G = c.G

    # --- CGS ---
    r    = r.to(u.cm)
    r0   = r0.to(u.cm)
    rho0 = rho0.to(u.g/u.cm**3)
    Mstar = Mstar.to(u.g)
    Mi   = Mcollisions.to(u.g)
    MBH  = MBH.to(u.g)
    Gc   = G.cgs

    c_rho = (rho0 * r0**alpha).to(u.g * u.cm**(alpha - 3))

    q = (Mi / Mstar).decompose().value  # dimensionless

    pref = (0.34 * q) / (Gc**2 * Mstar * lnLambda)
    scale = ((cv * Gc) / (1 + alpha))**(3/2) * (MBH**(3/2) / c_rho)

    tdf = (pref * scale * r**(alpha - 3/2)).to(u.yr)
    return tdf


def stellar_df_radius(td, Mstar, lnLambda, Mcollisions, alpha, r0, rho0, cv=1.0):
    """
    Stellar-dominated DF radius from t_df(r)=t_d using the power-law model.
    Computes in CGS to avoid fractional-power unit chaos.
    """
    G = c.G
    td   = td.to(u.s)
    Mstar = Mstar.to(u.g)
    Mi   = Mcollisions.to(u.g)
    r0   = r0.to(u.cm)
    rho0 = rho0.to(u.g/u.cm**3)
    Gc   = G.cgs

    # rho(r) = rho0 (r/r0)^(-alpha) = c_rho r^(-alpha)
    c_rho = (rho0 * r0**alpha).to(u.g * u.cm**(alpha - 3))

    # M(r) = c_M r^(3-alpha), with c_M = 4π c_rho/(3-alpha)
    c_M = (4*np.pi * c_rho / (3 - alpha)).to(u.g * u.cm**(alpha - 3))

    q = (Mi / Mstar).decompose().value  # dimensionless

    # RHS for r^(3 - alpha/2)
    RHS = (Gc**2 * Mstar * td * lnLambda) / (0.34 * q)
    RHS *= c_rho / ((cv * Gc / (1 + alpha))**(3/2) * c_M**(3/2))

    # enforce expected units before taking the power
    RHS = RHS.to(u.cm**(3 - alpha/2))

    r = RHS**(1 / (3 - alpha/2))
    return r.to(u.pc)


def bh_df_radius(td, Mstar, lnLambda, Mcollisions, alpha, r0, rho0, MBH, cv=1.0):
    """
    BH-dominated DF radius from t_df(r)=t_d in the regime MBH >> M(r).

    Returns the radius implied by the inequality; note:
    - if alpha > 3/2, this is an *upper* bound (maximum radius to sink within td)
    - if alpha < 3/2, the inequality gives a *lower* bound (friction speeds up inward),
      so interpreting this as "maximum radius" is not correct.
    """
    G = c.G
    # --- force CGS early ---
    td    = td.to(u.s)
    Mstar = Mstar.to(u.g)
    Mi    = Mcollisions.to(u.g)
    MBH   = MBH.to(u.g)
    r0    = r0.to(u.cm)
    rho0  = rho0.to(u.g/u.cm**3)
    Gc    = G.cgs

    # rho(r) = rho0 (r/r0)^(-alpha) = c_rho r^(-alpha)
    c_rho = (rho0 * r0**alpha).to(u.g * u.cm**(alpha - 3))

    q = (Mi / Mstar).decompose().value  # dimensionless

    # RHS for r^(alpha - 3/2)
    RHS = (Gc**2 * Mstar * td * lnLambda) / (0.34 * q)
    RHS *= c_rho / ((cv * Gc / (1 + alpha))**(3/2) * MBH**(3/2))

    # enforce expected units before fractional power
    RHS = RHS.to(u.cm**(alpha - 3/2))

    expo = (alpha - 1.5)
    if np.isclose(expo, 0.0):
        # alpha = 3/2 => condition becomes independent of r in this approximation
        # (either always satisfies or never satisfies depending on parameters)
        return np.inf * u.pc

    r = RHS**(1 / expo)
    return r.to(u.pc)