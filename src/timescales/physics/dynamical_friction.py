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


def stellar_df_time(r,Mstar,coulomb, Mcollisions,alpha,r0,rho0, cv =1.):
    cm= 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    A1 = (c.G**2 * Mstar* coulomb)/(0.34 * massratio*c_rho)
    A2 = ((1+alpha)/(c_v * G))**(3/2)
    A = A1*A2
    return (A/cm *(r)**(3.-(alpha/2.))).to('yr')


def bh_df_time(r,Mstar,coulomb, Mcollisions,alpha,r0,rho0,MBH, cv =1.):
    cm= 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    A1 = (c.G**2 * Mstar* coulomb)/(0.34 * massratio*c_rho)
    A2 = ((1+alpha)/(c_v * G))**(3/2)
    A = A1*A2
    return (A/M_BH *(r)**(alpha-(3./2.))).to('yr')


def stellar_df_radius(r,td, Mstar,coulomb, Mcollisions,alpha,r0,rho0, cv =1.):
    cm= 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    A1 = (c.G**2 * Mstar* coulomb)/(0.34 * massratio*c_rho)
    A2 = ((1+alpha)/(c_v * G))**(3/2)
    A = A1*A2
    return ((A*td/cm)**(1/(3.-(alpha/2.)))).to('pc')



def bh_df_radius(r,td,Mstar,coulomb, Mcollisions,alpha,r0,rho0,MBH, cv =1.):
    cm= 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    A1 = (c.G**2 * Mstar* coulomb)/(0.34 * massratio*c_rho)
    A2 = ((1+alpha)/(c_v * G))**(3/2)
    A = A1*A2
    return ((A*td/M_BH)**(1/(alpha-(3./2.)))).to('pc')