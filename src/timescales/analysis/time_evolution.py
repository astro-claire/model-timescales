"""
Time evolution scripts
"""
import numpy as np
import astropy.units as u
from ..physics.collisions import collision_timescale
from ..physics.gas import gas_mass_per_collision



def per_system_te(radii, profile, f =0.01, alpha =1.75, end = 1e9):
    """
    For each system, update properties based on time evolution of density
    
    For each radius, need to calculate initial density, and delta_t to accumulate frho_* of gas
    """
    # Set initial values
    rho0= profile.rho0
    r0 = profile.r0
    rhostars = rho0*(radii/r0)**(-alpha)
    rhogas = np.zeros(len(radii.value)) *(u.Msun /u.pc**3)
    timestamp = 1
    #initialize arrays
    t_array = [timestamp] # time in years
    rho_star_array = [rhostars]
    rho_gas_array = [rhogas]
    while timestamp < end: 
        # Save previous values
        rhostar_old = rhostars
        rhogas_old = rhogas

        #determine needed delta_t
        t_buildup_f = calculate_buildup_time(radii, profile,rhostars,alpha, f =f ) #  f * Mstar * t_coll(radii, profile)/ mg_coll(radii)
        delta_t = min(t_buildup_f.to('yr').value)
        if delta_t > end/10:
            delta_t = end/10
        elif delta_t < end/1e9:
            print("Timestep became too short, exiting")
            break
        print(delta_t)
        # delta_t = 100

        #number of collisions & accumulated gas mas
        Mstar = 1*u.Msun
        v = profile.velocity_dispersion(radii)
        Mcollisions = 1 * u.Msun
        N_r = delta_t*u.yr/ t_coll(radii,profile, rhostars,v, alpha = alpha)
        Mg_coll = gas_mass_per_collision(v, Mstar, Mcollisions)
        Mstars = np.full(len(Mg_coll), 2*Mstar.to('Msun').value) * u.Msun
        Mg_coll_r = np.where(Mg_coll > 2*Mstar, Mstars,Mg_coll) #Replace any >2 Mstar values with 2 Mstar (can't collide more mass than the two stars combined)
        
        rhogas = rhogas_old+( Mg_coll_r * N_r * rhostars /Mstar) 

        #calculate new rhos
        # rhogas = rhostar_old / 3
        rhostars = rhostar_old - ( Mg_coll_r * N_r * rhostars /Mstar)   #CW Need to add depletion diffusion

        if np.any(rhostars< 0 * (u.Msun / u.pc**3)):
            break

        # Append the newly computed values
        rho_star_array.append(rhostars)
        rho_gas_array.append(rhogas)


        # Evolve to next step
        timestamp += delta_t
        t_array.append(timestamp)
    print("iterated through "+ str(len(t_array))+ " Steps")
    return dict([('t',t_array),('rhostar',rho_star_array),('rhogas',rho_gas_array), ('r', radii)] )




def calculate_buildup_time(radii, profile,rhostar,alpha,f, Mstar =1*u.Msun, Mcollisions =1*u.Msun):
    v = profile.velocity_dispersion(radii)
    tcollisions = t_coll(radii,profile, rhostar,v, alpha = alpha)
    return gas_buildup_timescale(tcollisions, v,Mstar, Mcollisions, f= f)



def t_coll(radii,profile, rhostar, v, Mstar = 1*u.Msun, alpha = 1.25,
                        Mcollisions = 1 * u.Msun,
                        n_unit=1./u.cm**3,
                        v_unit=u.cm/u.s,
                        Mstar_unit = u.Msun):
    
    nstar= rhostar/Mstar
    t_coll = collision_timescale(nstar, v,Mstar,
                        alpha = 1.25,
                        Mcollisions = 1 * u.Msun,
                        n_unit=1./u.cm**3,
                        v_unit=u.cm/u.s,
                        Mstar_unit = u.Msun )
    # need to think more about this--should I update the velocity dispersion ?
    return t_coll.to('yr')

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
    Mstars = np.full(len(Mg_coll), 2*Mstar.to('Msun').value) * u.Msun
    Mg_coll = np.where(Mg_coll > 2*Mstar, Mstars,Mg_coll) #Replace any >2 Mstar values with 2 Mstar (can't collide more mass than the two stars combined)

    return (f * Mstar * t_coll / Mg_coll).to(u.yr)
