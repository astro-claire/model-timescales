"""
Time evolution scripts
"""
import numpy as np
import astropy.units as u
from astropy.constants import m_p
from ..physics.collisions import collision_timescale
from ..physics.gas import gas_mass_per_collision
from ..utils.energy import escape_velocity
from ..physics.stars import stellar_radius_approximation
from ..physics.gas import sound_speed, gas_dynamical_friction_timescale, aerodynamic_drag_timescale, aerodynamic_2phase_drag_timescale, gas_2phase_dynamical_friction_timescale
from ..physics.dynamical_friction import dynamical_friction_timescale
from ..physics.relaxation import relaxation_timescale
from ..physics.coulomb import coulomb_log_BH
from ..physics.accretion import  stellar_2phase_bondi_accretion_rate, stellar_bondi_accretion_rate, smbh_bondi_accretion_rate, bondi_radius

def per_system_te(
    radii, profile, f=0.01, alpha=1.75, end=1e9,f1 = 0.05, 
    Mstar=1 * u.Msun, 
    velocity_damping = True, gdf = True, sdf = True, gas_diff = True, collisions=True,
    sn = True, sn_time = 20*u.Myr,
    bondi_stars = True,
    bondi_bh = True,
    gas_effects_off =False,
    rlx_only =False, #these are the physical effects that can be included
):
    """
    For each system, update properties based on time evolution of density.

    For each radius, need to calculate initial density, and delta_t to
    accumulate frho_* of gas.

    Parameters
    ----------
    radii : Quantity [length]
        Array of radii to evolve.
    profile : object
        Density/velocity-dispersion profile.
    f : float
        Gas buildup fraction threshold.
    f1 : float
        Dense gas fraction in the 2phase model
    alpha : float
        Stellar density power-law slope.
    end : float
        End time in years.
    Mstar : Quantity [mass]
        Characteristic stellar mass assumed for collisions.
    """

    
    if sdf==False: 
        print("WARNING: Stellar dynamical friction is turned OFF for this run.")
    if gdf==False: 
        print("WARNING: Gas dynamical friction is turned OFF for this run.")
    if velocity_damping==False: 
        print("WARNING: Velocity damping is turned OFF for this run.")
    if gas_diff==False: 
        print("WARNING: Gas Diffusion is turned OFF for this run.")
    if sn ==False:
        print("WARNING: Supernovae turned OFF for this run.")
    if collisions == False:
        print("WARNING: collisions turned OFF for this run")
    if bondi_bh == False:
        print("WARNING: Black Hole bondi accretion turned OFF for this run")    
    if bondi_stars == False:
        print("WARNING: stellar bondi accretion turned OFF for this run")
    if gas_effects_off:
        print("TURNING OFF ALL GAS DYNAMICS")
        gdf = False
        velocity_damping = False
        gas_diff = False
        sn =False
        bondi_bh = False
        bondi_stars = False
    if gdf ==False and velocity_damping==False and sn ==False and gas_diff ==False and collisions==False:
        print("WARNING: it's a relaxation only integration")
        rlx_only = True
    
    # Set initial values
    rho0 = profile.rho0
    r0   = profile.r0
    rhostars       = rho0 * (radii / r0) ** (-alpha)
    rhostars_trackn = rhostars
    rhogas = np.zeros(len(radii.value)) * (u.Msun / u.pc**3)
    rho_prod = np.zeros(len(radii.value)) * (u.Msun / u.pc**3)
    Mcollisions = Mstar
    Mstar = np.full(len(radii), Mstar.value) * Mstar.unit #turning this into an array so we can track the mass accretion
    cs = sound_speed(1e7*u.K)
    
    #BH Bondi physcs
    r_bondi_bh = bondi_radius(profile.M_bh, cs)
    #bondi_radii = np.where(radii<r_bondi_bh, 1, 0) #radii within the black hole's bondi radius 
    cs = sound_speed(1e7 * u.K)
    r_bondi_bh = bondi_radius(profile.M_bh, cs)

    in_bondi   = radii < r_bondi_bh          # boolean, not 0/1
    n_in_bondi = int(np.count_nonzero(in_bondi))

    if bondi_bh:
        if n_in_bondi == 0:
            print("WARNING: no shells inside r_B = "
                f"{r_bondi_bh.to('pc'):.3g}; SMBH accretion will be inactive.")
        elif n_in_bondi == len(radii):
            print(f"WARNING: r_B = {r_bondi_bh.to('pc'):.3g} exceeds the outer grid "
                "radius; the entire cluster is inside the Bondi sphere.")

    M_bh_accreted = 0 * u.Msun               # cumulative, diagnostic

    #shell volumes 
    # Midpoints between adjacent radii form the interior edges
    mid = 0.5 * (radii[:-1] + radii[1:])
    mid = np.sqrt(radii[:-1]*radii[1:])
    # Extrapolate the first and last edges symmetrically
    # r_inner = radii[0]  - (mid[0]  - radii[0])   # = 2*r[0] - mid[0]
    r_inner = radii[0]**2/mid[0]
    r_outer = radii[-1] + (radii[-1] - mid[-1])   # = 2*r[-1] - mid[-1]
    # Full edge array, length n+1
    edges = np.concatenate([[r_inner], mid, [r_outer]])
    # Shell volume centered on each radius, length n
    shell_vols = (4./3.) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    shell_floors = Mstar / shell_vols #can't have less than 1 star in each shell for collisions.

    # Physical values for merger criterion
    v_esc_star   = escape_velocity(Mstar, stellar_radius_approximation(Mstar))
    constructive = np.where(profile.velocity_dispersion(radii) < v_esc_star, 1, 0)
    destructive  = np.where(profile.velocity_dispersion(radii) > v_esc_star, 1, 0)

    # Initial number density of stars at each radius
    nstars = rhostars / Mstar
    # initial velocity dispersion
    sigma=profile.velocity_dispersion(radii)
    v = sigma
    coulomb = coulomb_log_BH(profile.M_bh,radii, sigma)
    #initial timescale tracking
    resolution = len(radii)
    t_df =[dynamical_friction_timescale(v, rhostars, M_obj=2 * Mcollisions, coulomb=coulomb)]
    t_collision = [t_coll(radii, profile, rhostars, v, alpha=alpha)]
    t_gas_df= [np.full(resolution, -1)*u.yr] # initially no gas - put a dummy value
    t_gas_buildup = [calculate_buildup_time(radii, profile, rhostars, alpha, f,v)]
    t_relaxation = [relaxation_timescale(sigma,rhostars,Mstar,coulomb=coulomb)]

    # 2 phase gas model 
    ## TODO add in kwargs instead of heres
    ##-----Phase 1: dense ionized filaments
    n1     = 3e4 * u.cm**-3
    T1     = 8000 * u.K
    mu1    = 0.5              # mean molecular weight (fully ionized H)
    rho1 = (2 * mu1 * n1 * m_p).to(u.g / u.cm**3)
    cs1 = sound_speed(T1, mu=mu1)
    # Phase 2: hot volume-filling plasma
    T2     = 1e7 * u.K
    mu2    = 0.5
    cs2 = sound_speed(T2, mu=mu2)
    
    n2     = 100 * u.cm**-3
    #rho2 = (2 * mu2* n2 * m_p).to(u.g / u.cm**3)
    # rho1factor = rho1/rho2
    rho1 = f1* rhogas
    # Setup loop
    timestamp = 1
    t_array        = [timestamp]
    rho_star_array = [rhostars]
    rho_gas_array  = [rhogas]
    constructive_array = [constructive]
    central_mass_array = [0*u.Msun] #these will store the mass accreted into the center
    central_mass_rate_array = [0*u.Msun/u.yr]
    central_mass_per_r_rate_array= [np.full(resolution, 0)*(u.Msun/u.yr)] # initially no gas - put a dummy value
    M_bh_accreted_array = [0 * u.Msun]
    mdot_bh_array = [0 * u.Msun / u.yr]

    r_central_mass_max_array = [0*u.pc]
    central_volumetric_rate_array = [0* (u.Msun / u.pc**3/u.yr)]
    no_collisions = np.ones(resolution) #initially, collisions are allowed to occur everywhere in the cluster

    floor = 1e-4 * (u.Msun / u.pc**3)

    supernova_timer =1
    supernova_factor = 1. 
    sn_fired_this_step = False
    sn_fired_last_step = False  
    sn_wiggle_room = 0.
    rhogas_threshold = 1e-5 * u.Msun / u.pc**3
    while timestamp < end:
        # ------------------------------------------------------------------ #
        # Save previous values
        # ------------------------------------------------------------------ #
        coulomb = coulomb_log_BH(profile.M_bh,radii, v)
        rhostar_old = rhostars
        rhogas_old  = rhogas



    # ------------------------------------------------------------------ #
        # PREDICTOR: compute all rates from *current* state (v, rhostars,    #
        # rhogas) before choosing delta_t. None of these depend on delta_t,  #
        # so there's no staleness issue.                                     #
        # ------------------------------------------------------------------ #
        collision_t = t_coll(radii, profile, rhostars_trackn, v, alpha=alpha, Mstar = Mstar, Mcollisions=Mstar)

        Mg_coll    = gas_mass_per_collision(v, Mstar, Mstar)
        Mstars_cap = np.full(len(Mg_coll), 2 * Mstar.to('Msun').value) * u.Msun
        tinyMstars = np.full(len(Mg_coll), (1e-6 * u.Msun).value) * u.Msun
        Mg_coll_r  = np.where(Mg_coll > 2 * Mstar, Mstars_cap, Mg_coll)
        Mg_coll_r  = np.where(Mg_coll_r < 1e-6 * Mstar, tinyMstars, Mg_coll_r)

        rhogas_gdf = np.where(rhogas < 0.001 * f * rhostars, 0.001 * f * rhostars, rhogas)
        # g_df   = gas_dynamical_friction_timescale(v, Mstar, rhogas_gdf)
        g_df = gas_2phase_dynamical_friction_timescale(v,Mstar,rho1,T1,v/cs1,mu1, rhogas, T2, v/cs2, mu2, f1)
        #g_aero = aerodynamic_drag_timescale(v, Mstar, rhogas_gdf, M=2)
        g_aero = aerodynamic_2phase_drag_timescale(v,Mstar,rho1, T1,v/cs1,mu1,
                                                        rhogas, T2, v/cs2,mu2,f1) 
        g_df   = np.where(g_df > g_aero, g_aero, g_df)
        s_df   = dynamical_friction_timescale(v, rhostars_trackn, M_obj=2 * Mstar, coulomb=coulomb)
        s_rlx  = relaxation_timescale(v, rhostars_trackn, Mstar, coulomb=coulomb)

        t_buildup_f = calculate_buildup_time(radii, profile, rhostars_trackn, alpha, f, v, Mstar =Mstar)
        t_gas_buildup.append(t_buildup_f)

        # Bound delta_t by a safety fraction of *every* relevant loss/production
        # timescale — this replaces the old "growth only, stale rate" limiter.
        safety = 0.9  # tunable
        delta_t = min(
            min(t_buildup_f.to('yr').value[no_collisions.astype(bool)]),
            safety * min(g_df.to('yr').value[no_collisions.astype(bool)]),
            safety * min(s_df.to('yr').value[no_collisions.astype(bool)]),
            safety * min(s_rlx.to('yr').value[no_collisions.astype(bool)]),
        )
        # ------------------------------------------------------------------ #
        # Determine adaptive delta_t
        # ------------------------------------------------------------------ #
        # t_buildup_f = calculate_buildup_time(radii, profile, rhostars_trackn, alpha, f,v)#[no_collisions.astype(bool)]
        # t_gas_buildup.append(t_buildup_f)
        # delta_t = min(t_buildup_f.to('yr').value)
        # Prevent rhogas from changing by more than a factor of 2 per step
        # if timestamp > 1 and not sn_fired_last_step:
            # collision_rate = Mg_coll_r * N_r * rhostars / Mstar / delta_t / u.yr \
            #                 + 1e-30 * u.Msun/u.pc**3/u.yr
            # t_gas_limit = (rhogas / collision_rate).to('yr')
            # rhogas_threshold = f * rhostars
            # valid = rhogas > rhogas_threshold
            # #this only needs to be checked if the rhogas is large
            # if np.any(valid):
            #     delta_t = min(delta_t, 0.5 * min(t_gas_limit[valid].value))

        if delta_t > end / 10: #force integration over 10 steps
            delta_t = end / 10

        
        if sn:
            sn_fired_this_step = False
            time_to_next_sn = (supernova_timer + sn_time.to('yr').value) - timestamp
            if delta_t >= time_to_next_sn:
                delta_t          = time_to_next_sn
                supernova_timer  = supernova_timer + sn_time.to('yr').value
                supernova_factor = 1e-6
                sn_fired_this_step = True
                # print(f"SN went off at t={timestamp + delta_t:.3e} yr, "
                #     f"next at {supernova_timer+ sn_time.to('yr').value:.3e} yr")
                sn_wiggle_room = timestamp +delta_t+1e4
            if timestamp<sn_wiggle_room:
                sn_fired_last_step = True
            else:
                sn_fired_last_step = sn_fired_this_step  

        if rlx_only==True: # in the case of a relaxation only simulation, we can ignore the gas dynamics when calculating the timestep
            delta_t = min(relaxation_timescale(v,rhostars_trackn,Mstar,coulomb = coulomb).to('yr').value)/100
            # delta_t = end/500
            print(timestamp)
        if not sn_fired_this_step and delta_t < end / 1e9:
            print(rhogas)
            print(delta_t)
            print("Timestep became too short, exiting")
            break
        
        # ------------------------------------------------------------------ #
        # CORRECTOR: compute the delta_t-dependent updates, check that no    #
        # reservoir changes by more than max_frac in this step. If it does,  #
        # shrink delta_t and only recompute the delta_t-dependent pieces —   #
        # the rates above (g_df, s_df, s_rlx, Mg_coll_r) don't need redoing. #
        # ------------------------------------------------------------------ #
        max_frac = 0.1
        for _ in range(4):
            N_r = delta_t * u.yr / collision_t
            if collisions == False:
                N_r = N_r * 0.

            collision_loss = Mg_coll_r * N_r * rhostars / Mstar * no_collisions
            rho_lost_gdf   = rhostars * (delta_t * u.yr) / g_df
            rho_lost_sdf   = rho_prod * (delta_t * u.yr) / s_df       # include this too —
            rho_lost_srlx  = rhostars * (delta_t * u.yr) / s_rlx      # it's just as stiff near the transition

            gas_lost = rhogas * (delta_t * u.yr) / g_df
            if gas_diff == False:
                gas_lost = gas_lost * 0.
            gas_lost = np.where(gas_lost > rhogas_old, rhogas_old, gas_lost)

            gas_frac   = np.abs(collision_loss - gas_lost) / (rhogas_old + 1e-30 * u.Msun / u.pc**3)
            star_frac  = (collision_loss + rho_lost_gdf + rho_lost_sdf + rho_lost_srlx) / (rhostar_old + 1e-30*u.Msun/u.pc**3)
            worst_frac = max(np.max(gas_frac.value[no_collisions.astype(bool)]), np.max(star_frac.value[no_collisions.astype(bool)]))

            if worst_frac <= max_frac or delta_t <= end / 1e9:
                break
            delta_t = delta_t * (max_frac / worst_frac) * 0.9   # shrink with margin, retry



        # ------------------------------------------------------------------ #
        # Number of collisions & accumulated gas mass
        # ------------------------------------------------------------------ #

        
        collision_t = t_coll(radii, profile, rhostars_trackn, v, alpha=alpha, Mstar = Mstar, Mcollisions=Mstar)
        t_collision.append(collision_t)
        N_r         = delta_t * u.yr / collision_t
        if collisions ==False:
            N_r = N_r*0.
        Mg_coll     = gas_mass_per_collision(v, Mstar, Mstar)
        # Mstars      = np.full(len(Mg_coll), 2 * Mstar.to('Msun').value) * u.Msun
        tinyMstars      = np.full(len(Mg_coll), (1e-6 * u.Msun).value) * u.Msun
        Mg_coll_r   = np.where(Mg_coll > 2 * Mstar,  2 * Mstar, Mg_coll)
        Mg_coll_r   = np.where(Mg_coll_r<0.000001 *Mstar,tinyMstars, Mg_coll_r ) #mcollisions is an astropy float with the original mass
        # print(Mg_coll_r)
        # # Constructive fraction not used yet
        # frac_reduction = constructive * np.where((N_r > 1), 1, N_r)

        # ------------------------------------------------------------------ #
        # Dynamical friction: gas DF, stellar DF
        #                                                 #
        # ------------------------------------------------------------------ #
        # g_df    = gas_dynamical_friction_timescale(v, Mstar, rhogas)
        rhogas_gdf = np.where(rhogas < 0.001 * f * rhostars,0.001* f * rhostars, rhogas)
        # g_df = gas_dynamical_friction_timescale(v, Mstar, rhogas_gdf)
        # g_aero = aerodynamic_drag_timescale(v, Mstar, rhogas_gdf, M = 2) #TODO add self consistent gas temp thing
        g_df = gas_2phase_dynamical_friction_timescale(v,Mstar,rho1,T1,v/cs1,mu1, rhogas, T2, v/cs2, mu2, f1)
        g_aero = aerodynamic_2phase_drag_timescale(v,Mstar,rho1, T1,v/cs1,mu1,
                                                        rhogas, T2, v/cs2,mu2,f1) 
        g_df = np.where(g_df>g_aero, g_aero, g_df)
        s_df    = dynamical_friction_timescale(v, rhostars_trackn, M_obj=2 * Mstar, coulomb = coulomb)
        s_rlx = relaxation_timescale(v,rhostars_trackn,Mstar,coulomb = coulomb)
        # t_relaxation.append(relaxation_timescale(sigma,rhostars_trackn,Mstar,coulomb = 10))
        t_relaxation.append(s_rlx)
        t_df.append(s_df)
        t_gas_df.append(g_df)
        total_df        = np.where(g_df < s_df, g_df, s_df)
        total_df_status = np.where(g_df < s_df, 'g', 's')

        # ---- raw loss/production terms, all from start-of-step state ---- #
        rho_lost_gdf  = rhostars * (delta_t * u.yr) / g_df
        rho_lost_srlx = rhostars * (delta_t * u.yr) / s_rlx
        rho_lost_sdf  = rho_prod * (delta_t * u.yr) / s_df      # old rho_prod

        collision_loss        = Mg_coll_r * N_r * rhostars / Mstar * no_collisions
        collision_merger_gain = (2*Mstar - Mg_coll_r) * N_r * rhostars / Mstar * no_collisions

        # ---- non-finite guard ---- #
        zero = 0 * rhostar_old
        rho_lost_gdf  = np.where(np.isfinite(rho_lost_gdf.value),  rho_lost_gdf,  zero)
        rho_lost_sdf  = np.where(np.isfinite(rho_lost_sdf.value),  rho_lost_sdf,  zero)
        rho_lost_srlx = np.where(np.isfinite(rho_lost_srlx.value), rho_lost_srlx, zero)

        if gdf is False:
            rho_lost_gdf = rho_lost_gdf * 0.
        if sdf is False:
            rho_lost_sdf = rho_lost_sdf * 0.

        # ---- cap total removal at (rho_* - floor), not rho_* ---- #
        # A shell sitting exactly on the floor has max_removable == 0, so
        # scale == 0 and every loss term vanishes: the shell is frozen, no
        # mass is invented, and nothing leaks into mass_accreted_central.
        max_removable = np.maximum(rhostar_old - floor, 0 * floor)
        total_loss = collision_loss + rho_lost_gdf + rho_lost_sdf + rho_lost_srlx

        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(
                total_loss > max_removable,
                (max_removable / total_loss).to_value(u.dimensionless_unscaled),
                1.0,
            )
        scale = np.where(np.isfinite(scale), scale, 1.0)   # total_loss == 0 case

        # ---- apply the same scale to every term, including rho_prod's ---- #
        collision_loss        = collision_loss * scale
        collision_merger_gain = collision_merger_gain * scale
        rho_lost_gdf          = rho_lost_gdf * scale
        rho_lost_sdf          = rho_lost_sdf * scale
        rho_lost_srlx         = rho_lost_srlx * scale

        rho_prod = rho_prod + collision_merger_gain - rho_lost_sdf
        rho_prod = np.where(rho_prod < 0 * (u.Msun/u.pc**3),
                            0 * (u.Msun/u.pc**3), rho_prod)
        rho_prod = np.minimum(rho_prod, rhostar_old)   # products are a subset of rho_*


        # ------------------------------------------------------------------ #
        # Calculate new densities, depleting r_bondi at the bondi rate
        # ------------------------------------------------------------------ #
        rhogas   = (rhogas_old + collision_loss - gas_lost) * supernova_factor

        if bondi_bh and n_in_bondi > 0:
            # Gas currently available inside the Bondi sphere
            m_gas_shell = (rhogas * shell_vols)[in_bondi]
            M_gas_bondi = m_gas_shell.sum()

            if M_gas_bondi > 0 * u.Msun:
                # ONE global rate, set by the volume-averaged density inside r_B.
                # (Using the mean interior density rather than rho at r_B makes the
                #  rate self-limiting as the Bondi sphere empties.)
                V_bondi   = shell_vols[in_bondi].sum()
                rho_bondi = (M_gas_bondi / V_bondi).to(u.Msun / u.pc**3)
                #eddington limited FIXME have the fractional efficiency as a kwarg
                eps = 0.1
                mdot_bh = smbh_bondi_accretion_rate(rho_bondi, profile.M_bh, cs2)
                mdot_edd = (2.2e-8 * (profile.M_bh / u.Msun).decompose() / (0.1/eps)) * u.Msun / u.yr
                mdot_bh  = min(mdot_bh, mdot_edd)   # apply before computing dM_bh

                dM_bh   = (mdot_bh * delta_t * u.yr).to(u.Msun)

                # Cannot accrete more gas than the Bondi sphere holds
                if dM_bh > M_gas_bondi:
                    dM_bh   = M_gas_bondi
                    mdot_bh = (dM_bh / (delta_t * u.yr)).to(u.Msun / u.yr)
                
                # Drain each shell in proportion to its gas mass. Because the weight
                # is m_i / M_tot, this is a uniform fractional depletion, so rhogas
                # stays >= 0 by construction whenever dM_bh <= M_gas_bondi.
                frac       = float((dM_bh / M_gas_bondi).decompose())
                depletion  = np.where(in_bondi, frac, 0.0)
                rhogas     = rhogas * (1.0 - depletion)

                M_bh_accreted = M_bh_accreted + dM_bh
            else:
                mdot_bh = 0 * u.Msun / u.yr
                dM_bh   = 0 * u.Msun
        else:
            mdot_bh = 0 * u.Msun / u.yr
            dM_bh   = 0 * u.Msun


        # rho_prod = rho_prod + collision_merger_gain - rho_lost_sdf
        # rho_prod = np.where(rho_prod < 0 * (u.Msun/u.pc**3), 0 * (u.Msun/u.pc**3), rho_prod)


        #rho_lost_sdf already contains the products
        rhostars = rhostar_old - collision_loss - rho_lost_gdf - rho_lost_sdf - rho_lost_srlx
        rhostars_trackn = (
            rhostar_old
            - collision_loss - rho_lost_gdf - rho_lost_sdf - rho_lost_srlx
        )
        # rhogas    = (rhogas_old + collision_loss - gas_lost )* supernova_factor
        # rhostars  = rhostar_old - collision_loss -rho_lost_gdf-rho_lost_sdf - rho_lost_srlx#this array provides the true mass density
        # rhostars_trackn = (
        #     rhostar_old
        #     - (destructive * collision_loss)
        #     - (constructive * (N_r * rhostars))-rho_lost_gdf-rho_lost_sdf - rho_lost_srlx
        # ) # this array will give n star when divided by Mstar (since we're not actively updating Mstar)
        mass_accreted_central = shell_vols *(rho_lost_gdf+rho_lost_sdf+rho_lost_srlx)


        # ------------------------------------------------------------------ #
        # Handle any numerical overflows
        # ------------------------------------------------------------------ #        
        #First, don't let any nans or infs into the arrays
        if np.any(~np.isfinite(rhostars.value)):
            print("Non-finite values in rhostars — likely overflow. Exiting.")
            break
        if np.any(~np.isfinite(rhogas.value)):
            print("rhogas overflow — exiting.")
            break
            
        #second, don't let the density become negative
        # depleted = rhostars <= 1e-4 * (u.Msun / u.pc**3)
        depleted = rhostars<shell_floors
        no_collisions = np.where(depleted,0,1)
        # if np.any(depleted):
        #     print("stars were depleted!")
        if depleted.all():
            print("Core Collapse!!")
            break

        # rho_* is now guaranteed >= floor by construction, so no clamping.
        frozen = rhostars <= floor * (1 + 1e-12)   # diagnostic only
        depleted = rhostars < shell_floors         # < 1 star: stop collisions
        no_collisions = np.where(depleted, 0, 1)

        if frozen.all():
            print(f"All shells hit the density floor at t = {timestamp:.3e} yr")
            break

        # ------------------------------------------------------------------ #
        # gas velocity damping for the next timestep
        # ------------------------------------------------------------------ #
        if velocity_damping:
            # t_gdf = gas_dynamical_friction_timescale(v, Mstar, rhogas)
            # t_gdf = gas_2phase_dynamical_friction_timescale(v,Mstar,rho1,T1,v/cs1,mu1, rhogas, T2, v/cs2, mu2, f1)
            t_gdf = g_df
            decay = np.exp(-delta_t * u.yr / t_gdf)
            v = v * decay

            # factor = 0.1 # option A - a constant factor
            t_rlx = relaxation_timescale(sigma,rhostars,Mstar)
            # factor = np.exp(-t_rlx/t_gdf)
            factor = t_gdf / (t_gdf + t_rlx)

            vfloor = factor * sigma 
            v = np.where(v<vfloor, vfloor, v)

            v_abs_floor = 0.000001 * sigma
            v = np.where(v < v_abs_floor, v_abs_floor, v)
            # calculate the new constructive / destructive criterion
            constructive = np.where(v < v_esc_star, 1, 0)
            destructive  = np.where(v > v_esc_star, 1, 0)
        # print(constructive)
        # print(rhostars)

        v_esc_star   = escape_velocity(Mstar, stellar_radius_approximation(Mstar))


        # ------------------------------------------------------------------ #
        #advance the timestep
        # ------------------------------------------------------------------ #
       #if rhostars fall below 0, set to arbitrarily small value

        rho_star_array.append(rhostars)
        rho_gas_array.append(rhogas)
        constructive_array.append(constructive)
        central_mass_array.append(np.sum(mass_accreted_central))
        central_mass_rate_array.append(np.sum(mass_accreted_central)/delta_t/u.yr)
        central_mass_per_r_rate_array.append(mass_accreted_central/delta_t/u.yr)
        rmax = np.where((mass_accreted_central/delta_t)== max((mass_accreted_central/delta_t)))[0]
        M_bh_accreted_array.append(M_bh_accreted)
        mdot_bh_array.append(mdot_bh)
        # r_central_mass_max_array.append(radii[rmax][0])
        # central_volumetric_rate_array.append((rho_lost_sdf +rho_lost_gdf)/ delta_t / u.yr)

        timestamp += delta_t
        t_array.append(timestamp)
        supernova_factor=1. #reset SN

        #---------------------- Bondi Accretion on Stars 
        if bondi_stars:
            #bondi_mdot_stars = stellar_bondi_accretion_rate(rhogas,Mstar,cs, v)
            bondi_mdot_stars = stellar_2phase_bondi_accretion_rate(rho1,rhogas,Mstar,cs1,cs2,v, f1)
            rhogas = rhogas - (bondi_mdot_stars * delta_t * u.yr * rhostars/Mstar ) # need to subtract the gas that accreted to stars
            rhogas = np.where(rhogas<0 *u.Msun/(u.pc**3), 0*u.Msun/(u.pc**3), rhogas)
            Mstar += bondi_mdot_stars * delta_t * u.yr
        rho1 = f1* rhogas # *rho1factor
        print(timestamp)


    print("Iterated through " + str(len(t_array)) + " steps")

    result = dict(
        t=t_array,
        rhostar=rho_star_array,
        rhogas=rho_gas_array,
        r=radii,
        constructive = constructive_array,
        central_mass = central_mass_array,
        central_mass_rate = central_mass_rate_array,
        central_mass_rate_r = central_mass_per_r_rate_array,
        r_mmax = r_central_mass_max_array,
        t_coll = t_collision,
        t_df = t_df,
        t_gb = t_gas_buildup,
        t_gdf = t_gas_df,
        t_rlx = t_relaxation,
        M_bh_accreted = M_bh_accreted_array, 
        mdot_bh = mdot_bh_array,
        # central_volumetric_rate = central_volumetric_rate_array,
    )

    return result


# ------------------------------------------------------------------ #
#additional functions
# ------------------------------------------------------------------ #
def imf_representative_mass(imf, M_threshold):

    #NOT USING THIS ANYMORE BUT LEAVING FOR POSTERITY
    """
    Compute the number-weighted mean mass and mass fraction of stars above
    M_threshold, using the IMF piecewise power-law directly.

    Works entirely in stripped (float) units to avoid astropy's fractional-
    exponent unit arithmetic, which breaks inside kroupaIMF.get_normalization().
    The normalization constant A cancels in all ratios computed here.

    Parameters
    ----------
    imf : imfBase subclass (e.g. kroupaIMF)
        IMF instance. Only imf.edges and imf.alphas are used.
    M_threshold : Quantity [mass]
        Lower bound of the "massive" population.

    Returns
    -------
    M_rep : Quantity [mass]
        Number-weighted mean mass of stars with M > M_threshold.
    f_mass_massive : float
        Fraction of total stellar mass in stars with M > M_threshold.
    """
    # Strip units: work in Msun throughout
    edges  = [e.to(u.Msun).value for e in imf.edges]
    alphas = imf.alphas
    M_lo   = max(M_threshold.to(u.Msun).value, edges[0])
    M_hi   = edges[-1]

    # Continuity coefficients C[i]: k_i = A * C[i], C[0] = 1
    # Ensures dN/dM is continuous at each break.
    C = [1.0]
    for i in range(len(alphas) - 1):
        C.append(C[-1] * edges[i + 1] ** (alphas[i + 1] - alphas[i]))

    def eval_dNdM(M_arr):
        """Evaluate dN/dM ∝ C[i] * M^{-alpha_i} on a mass grid (float array)."""
        dN = np.zeros_like(M_arr)
        for i, alpha in enumerate(alphas):
            a, b = edges[i], edges[i + 1]
            mask = (M_arr >= a) & (M_arr <= b)
            if np.any(mask):
                dN[mask] = C[i] * M_arr[mask] ** (-alpha)
        return dN

    # --- number-weighted mean mass above M_threshold ---
    M_above = np.linspace(M_lo, M_hi, 5000)
    dN_above = eval_dNdM(M_above)
    numerator   = np.trapz(M_above * dN_above, M_above)   # Msun
    denominator = np.trapz(dN_above, M_above)              # dimensionless
    if denominator == 0:
        raise ValueError(
            f"No IMF probability above M_threshold={M_threshold}; "
            "check imf.Mmax or M_threshold."
        )
    M_rep = (numerator / denominator) * u.Msun

    # --- mass fraction above M_threshold (full IMF for denominator) ---
    M_full   = np.linspace(edges[0], M_hi, 10000)
    dN_full  = eval_dNdM(M_full)
    mass_full  = np.trapz(M_full * dN_full,  M_full)
    # Reuse the already-computed above-threshold mass integral
    mass_above = np.trapz(M_above * dN_above, M_above)
    f_mass_massive = float(mass_above / mass_full)

    return M_rep, f_mass_massive

def calculate_buildup_time(radii, profile, rhostar, alpha, f,v, Mstar=1 * u.Msun, Mcollisions=1 * u.Msun):
    # v = profile.velocity_dispersion(radii)
    tcollisions = t_coll(radii, profile, rhostar, v, alpha=alpha, Mstar=Mstar, Mcollisions=Mcollisions)
    return gas_buildup_timescale(tcollisions, v, Mstar, Mcollisions, f=f)


def t_coll(radii, profile, rhostar, v, Mstar=1 * u.Msun, alpha=1.25,
           Mcollisions=1 * u.Msun,
           n_unit=1. / u.cm**3,
           v_unit=u.cm / u.s,
           Mstar_unit=u.Msun):
    nstar  = rhostar / Mstar
    t_coll = collision_timescale(
        nstar, v, Mstar,
        alpha=alpha,
        Mcollisions=Mcollisions,
        n_unit=n_unit,
        v_unit=v_unit,
        Mstar_unit=Mstar_unit,
    )
    return t_coll.to('yr')


def gas_buildup_timescale(t_coll, sigma, Mstar, Mcollisions, f=0.01):
    """
    Timescale for gas to reach fraction f of the local stellar density (t_gb).

    Derived from t_gb = f * rho_*(r) * t_coll * 4*pi*r^2 / (M_g,coll * dN/dr).
    Since dN/dr = (rho_*/M_*) * 4*pi*r^2, the r^2 and rho_* factors cancel:

        t_gb = f * M_* * t_coll / M_g,coll(sigma)

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
    f : float
        Gas fraction threshold.

    Returns
    -------
    Quantity [time]
        Gas buildup timescale in years.
    """
    Mg_coll = gas_mass_per_collision(sigma, Mstar, Mcollisions)
    Mstars  = np.full(len(Mg_coll), 2 * Mstar.to('Msun').value) * u.Msun
    Mg_coll = np.where(Mg_coll > 2 * Mstar, Mstars, Mg_coll)
    return (f * Mstar * t_coll / Mg_coll).to(u.yr)