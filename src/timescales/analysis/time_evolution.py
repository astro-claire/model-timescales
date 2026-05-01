"""
Time evolution scripts
"""
import numpy as np
import astropy.units as u
from ..physics import collision_timescale
from ..physics import gas_mass_per_collision


def per_system_te(radii, profile, f =0.01, alpha =1.75, rho0=1e5 *(u.Msun /u.pc**3), r0 =1 *u.pc, end = 1e4):
    """
    For each system, update properties based on time evolution of density
    
    For each radius, need to calculate initial density, and delta_t to accumulate frho_* of gas
    """
    # Set initial values
    rhostars = rho0*(radii/r0)**(-alpha)
    rhogas = np.zeros(len(radii.value)) *(u.Msun /u.pc**3)
    timestamp = 0
    #initialize arrays
    t_array = [timestamp]
    rho_star_array = [rhostars]
    rho_gas_array = [rhogas]
    while timestamp < end: 
        # Save previous values
        rhostar_old = rhostars
        rhogas_old = rhogas

        #determine needed delta_t
        t_buildup_f = calculate_buildup_time() #  f * Mstar * t_coll(radii, profile)/ mg_coll(radii)
        delta_t = 100

        #calculate new rhos
        rhogas = rhostar_old /3

        rhostar = rhostar_old - rhogas
        
        break_condition = np.where(rhostar<0*(u.Msun /u.pc**3)),
        if len(break_condition)>1:
            break

        #reset
        rho_star_array.append(rhostars)
        rho_gas_array.append(rhogas)

        #evolve to next step
        timestamp += delta_t
        t_array.append(timestamp)

    print(t_array)
    print(rho_star_array)
    print(rho_gas_array)


radii = np.logspace(-2,1,10) * u.pc 
per_system_te(radii)





def t_coll(radii, rhostar, Mstar = 1*u.Msun, alpha = 1.25,
                        Mcollisions = 1 * u.Msun,
                        n_unit=1./u.cm**3,
                        v_unit=u.cm/u.s,
                        Mstar_unit = u.Msun):
    v = profile.velocity_dispersion(radii)
    nstar= rhostar/Mstar
    t_coll = collision_timescale(nstar, v,
                        alpha = 1.25,
                        Mcollisions = 1 * u.Msun,
                        n_unit=1./u.cm**3,
                        v_unit=u.cm/u.s,
                        Mstar_unit = u.Msun )
    # need to think more about this--should I update the velocity dispersion ?
    return t_coll.to('yr').value

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
    if Mg_coll > 2*Mstar:
        Mg_coll =2*Mstar
    return (f * Mstar * t_coll / Mg_coll).to(u.yr)