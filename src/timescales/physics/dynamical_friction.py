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
    Calculate the dynamical friction timescale (Chandrasekhar 1943 timescale, quoted in Fragione & Rasio 2023)\
    
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

@register_timescale("t_df", aliases = ("dynamical_friction"))
def dynamical_friction_timescale(M_obj, Mstar_cluster,v, rho, mass,*,
                                    coulomb=10, 
                                    v_unit = u.km/u.s, 
                                    rho_unit = u.g/u.cm**3,
                                    mass_unit = u.Msun):
    """ 
    Dynamical friction timescale at a given radius (local) 
    """
    M_obj = as_quantity(M_obj, mass_unit)
    Mstar_cluster = as_quantity(Mstar_cluster, mass_unit)
    massratio = Mstar_cluster/M_obj
    t_relax = relaxation_timescale(v,rho, mass, coulomb=10, v_unit = u.km/u.s, rho_unit = u.g/u.cm**3,mass_unit = u.Msun)
    return t_relax * massratio