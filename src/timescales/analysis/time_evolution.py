"""
Time evolution scripts
"""
import numpy as np
import astropy.units as u
from ..physics.collisions import collision_timescale
from ..physics.gas import gas_mass_per_collision
from ..utils.energy import escape_velocity
from ..physics.stars import stellar_radius_approximation
from ..physics.gas import gas_dynamical_friction_timescale
from ..physics.dynamical_friction import dynamical_friction_timescale
from ..physics.relaxation import relaxation_timescale
from ..physics.coulomb import coulomb_log_BH


def per_system_te(
    radii, profile, f=0.01, alpha=1.75, end=1e9,
    Mstar=1 * u.Msun, imf=None, M_threshold=None,
    velocity_damping = True, gdf = True, sdf = True, gas_diff = True, collisions=True, #these are the physical effects that can be included
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
    alpha : float
        Stellar density power-law slope.
    end : float
        End time in years.
    Mstar : Quantity [mass]
        Characteristic stellar mass assumed for collisions.
    imf : imfBase subclass, optional
        If provided, uses the IMF to estimate the representative mass and
        density fraction of stars more massive than M_threshold, and tracks
        their dynamical friction timescale separately.
    M_threshold : Quantity [mass], optional
        Mass above which stars are considered "massive" for the IMF-based DF
        calculation. Defaults to Mstar if not specified.
    """
    # ------------------------------------------------------------------ #
    # IMF: pre-compute representative mass and density fraction once,     #
    # before the time loop.  Both are insensitive to overall IMF scale.   #
    # ------------------------------------------------------------------ #
    use_imf = imf is not None
    if use_imf:
        if M_threshold is None:
            M_threshold = Mstar
        M_rep_massive, f_mass_massive = imf_representative_mass(imf, M_threshold)
        print(
            f"IMF: representative mass above {M_threshold:.2f} = {M_rep_massive:.3f}; "
            f"mass fraction = {f_mass_massive:.4f}"
        )
    if sdf==False: 
        print("WARNING: Stellar dynamical friction is turned OFF for this run.")
    if gdf==False: 
        print("WARNING: Gas dynamical friction is turned OFF for this run.")
    if velocity_damping==False: 
        print("WARNING: Velocity damping is turned OFF for this run.")
    if gas_diff==False: 
        print("WARNING: Gas Diffusion is turned OFF for this run.")
    # Set initial values
    rho0 = profile.rho0
    r0   = profile.r0
    rhostars       = rho0 * (radii / r0) ** (-alpha)
    rhostars_trackn = rhostars
    rhogas = np.zeros(len(radii.value)) * (u.Msun / u.pc**3)
    
    #shell volumes 
    # Midpoints between adjacent radii form the interior edges
    mid = 0.5 * (radii[:-1] + radii[1:])
    # Extrapolate the first and last edges symmetrically
    r_inner = radii[0]  - (mid[0]  - radii[0])   # = 2*r[0] - mid[0]
    r_outer = radii[-1] + (radii[-1] - mid[-1])   # = 2*r[-1] - mid[-1]
    # Full edge array, length n+1
    edges = np.concatenate([[r_inner], mid, [r_outer]])
    # Shell volume centered on each radius, length n
    shell_vols = (4./3.) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    shell_floors = 1*u.Msun / shell_vols #can't have less than 1 star in each shell for collisions.

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
    t_df =[dynamical_friction_timescale(v, rhostars, M_obj=2 * Mstar, coulomb=coulomb)]
    t_collision = [t_coll(radii, profile, rhostars, v, alpha=alpha)]
    t_gas_df= [np.full(resolution, -1)*u.yr] # initially no gas - put a dummy value
    t_gas_buildup = [calculate_buildup_time(radii, profile, rhostars, alpha, f,v)]

    # Setup loop
    timestamp = 1
    t_array        = [timestamp]
    rho_star_array = [rhostars]
    rho_gas_array  = [rhogas]
    constructive_array = [constructive]
    df_massive_array = []   # populated only when use_imf=True
    central_mass_array = [0*u.Msun] #these will store the mass accreted into the center
    central_mass_rate_array = [0*u.Msun/u.yr]
    central_mass_per_r_rate_array= [np.full(resolution, 0)*(u.Msun/u.yr)] # initially no gas - put a dummy value

    r_central_mass_max_array = [0*u.pc]
    central_volumetric_rate_array = [0* (u.Msun / u.pc**3/u.yr)]
    no_collisions = np.ones(resolution) #initially, collisions are allowed to occur everywhere in the cluster

    while timestamp < end:
        # ------------------------------------------------------------------ #
        # Save previous values
        # ------------------------------------------------------------------ #

        rhostar_old = rhostars
        rhogas_old  = rhogas

        # ------------------------------------------------------------------ #
        # Determine adaptive delta_t
        # ------------------------------------------------------------------ #
        t_buildup_f = calculate_buildup_time(radii, profile, rhostars_trackn, alpha, f,v)#[no_collisions.astype(bool)]
        t_gas_buildup.append(t_buildup_f)
        delta_t = min(t_buildup_f.to('yr').value)
        # Prevent rhogas from changing by more than a factor of 2 per step
        if timestamp>1:
            t_gas_limit = (rhogas / (Mg_coll_r * N_r * rhostars / Mstar /delta_t / u.yr + 1e-30 * u.Msun/u.pc**3/u.yr) ).to('yr')
            delta_t = min(delta_t, 0.5 * min(t_gas_limit.value))
        if delta_t > end / 10:
            delta_t = end / 10
        elif delta_t < end / 1e9:
            print("Timestep became too short, exiting")
            break

        # ------------------------------------------------------------------ #
        # Number of collisions & accumulated gas mass
        # ------------------------------------------------------------------ #

        Mcollisions = 1 * u.Msun
        collision_t = t_coll(radii, profile, rhostars_trackn, v, alpha=alpha)
        t_collision.append(collision_t)
        N_r         = delta_t * u.yr / collision_t
        if collisions ==False:
            N_r = N_r*0.
        Mg_coll     = gas_mass_per_collision(v, Mstar, Mcollisions)
        Mstars      = np.full(len(Mg_coll), 2 * Mstar.to('Msun').value) * u.Msun
        tinyMstars      = np.full(len(Mg_coll),0.000001 * Mstar.to('Msun').value) * u.Msun
        Mg_coll_r   = np.where(Mg_coll > 2 * Mstar, Mstars, Mg_coll)
        Mg_coll_r   = np.where(Mg_coll_r<0.000001 *Mstar,tinyMstars, Mg_coll_r )
        # print(Mg_coll_r)
        # # Constructive fraction not used yet
        # frac_reduction = constructive * np.where((N_r > 1), 1, N_r)

        # ------------------------------------------------------------------ #
        # Dynamical friction: gas DF, stellar DF, and (optionally) massive-   #
        # star DF from the IMF.                                                #
        # ------------------------------------------------------------------ #
        g_df    = gas_dynamical_friction_timescale(v, Mstar, rhogas)
        s_df    = dynamical_friction_timescale(v, rhostars, M_obj=2 * Mstar, coulomb = coulomb)
        t_df.append(s_df)
        t_gas_df.append(g_df)
        total_df        = np.where(g_df < s_df, g_df, s_df)
        total_df_status = np.where(g_df < s_df, 'g', 's')

        rho_lost_gdf = rhostars*(delta_t*u.yr)/g_df

        rho_lost_sdf = constructive * (delta_t*u.yr)/s_df * N_r * rhostars *no_collisions

        collision_loss = Mg_coll_r * N_r * rhostars / Mstar *no_collisions

        gas_lost  = rhogas * (delta_t*u.yr)/g_df  # density of gas lost
        if gas_diff ==False: 
            gas_lost = gas_lost * 0.
        gas_lost = np.where(
            gas_lost > rhogas_old,
            rhogas_old,
            gas_lost
        ) # make sure no more than the current gas amount is lost. 


        # Cap loss terms to at most the available stellar density
        rho_lost_gdf = np.where(
            np.isfinite(rho_lost_gdf.value), rho_lost_gdf, rhostar_old
        )
        rho_lost_sdf = np.where(
            np.isfinite(rho_lost_sdf.value), rho_lost_sdf, rhostar_old
        )
        if gdf ==False: #option to turn off gas dynamical friction
            rho_lost_gdf= rho_lost_gdf * 0.
        if sdf ==False:  #option to turn off stellar dynamical friction
            rho_lost_sdf = rho_lost_sdf* 0. 
        # Ensure combined losses don't exceed what's available
        total_loss = (Mg_coll_r * N_r * rhostars / Mstar) + rho_lost_gdf + rho_lost_sdf
        scale = np.where(
            total_loss > rhostar_old,
            rhostar_old / total_loss,
            np.ones(len(radii.value))
        )
        collision_loss = collision_loss * scale

        rho_lost_gdf = rho_lost_gdf * scale
        rho_lost_sdf = rho_lost_sdf * scale

        if use_imf:
            # Density of stars more massive than M_threshold, scaled from the
            # current total stellar density.  The mass fraction is kept fixed at
            # its initial IMF value (appropriate for a first-pass estimate).
            rho_massive = rhostars * f_mass_massive          # Msun/pc^3

            # DF timescale experienced by a star of representative mass M_rep
            # moving through the full stellar background.
            sdf_massive = dynamical_friction_timescale(
                v, rhostars, M_obj=M_rep_massive
            )
            gdf_massive = gas_dynamical_friction_timescale(v,M_rep_massive,rhogas)
            df_massive = np.where(gdf_massive < sdf_massive, gdf_massive, sdf_massive)
            df_massive_array.append(df_massive)


        # ------------------------------------------------------------------ #
        # Calculate new densities
        # ------------------------------------------------------------------ #

        rhogas    = rhogas_old + collision_loss - gas_lost
        rhostars  = rhostar_old - collision_loss -rho_lost_gdf-rho_lost_sdf#this array provides the true mass density
        rhostars_trackn = (
            rhostar_old
            - (destructive * collision_loss)
            - (constructive * (N_r * rhostars))-rho_lost_gdf-rho_lost_sdf
        ) # this array will give n star when divided by Mstar (since we're not actively updating Mstar)
        mass_accreted_central = shell_vols *(rho_lost_gdf+rho_lost_sdf)

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

        floor = 1e-4 * (u.Msun / u.pc**3)
        completely_depleted = rhostars <= 1e-4 * (u.Msun / u.pc**3)
        rhostars       = np.where(completely_depleted, floor, rhostars)
        rhostars_trackn = np.where(
            rhostars_trackn <= 0 * (u.Msun / u.pc**3), floor, rhostars_trackn
        )

        # ------------------------------------------------------------------ #
        # gas velocity damping for the next timestep
        # ------------------------------------------------------------------ #
        if velocity_damping:
            t_gdf = gas_dynamical_friction_timescale(v, Mstar, rhogas)
            decay = np.exp(-delta_t * u.yr / t_gdf)
            v = v * decay

            # factor = 0.1 # option A - a constant factor
            t_rlx = relaxation_timescale(sigma,rhostars,Mstar)
            # factor = np.exp(-t_rlx/t_gdf)
            factor = t_gdf / (t_gdf + t_rlx)

            vfloor = factor * sigma 
            v = np.where(v<vfloor, vfloor, v)

            v_abs_floor = 0.001 * np.min(sigma)
            v = np.where(v < v_abs_floor, v_abs_floor, v)
            # calculate the new constructive / destructive criterion
            constructive = np.where(v < v_esc_star, 1, 0)
            destructive  = np.where(v > v_esc_star, 1, 0)
        # print(constructive)
        # print(rhostars)


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
        r_central_mass_max_array.append(radii[rmax][0])
        # central_volumetric_rate_array.append((rho_lost_sdf +rho_lost_gdf)/ delta_t / u.yr)

        timestamp += delta_t
        t_array.append(timestamp)

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
        # central_volumetric_rate = central_volumetric_rate_array,
    )
    if use_imf:
        result['df_massive']      = df_massive_array
        result['M_rep_massive']   = M_rep_massive
        result['f_mass_massive']  = f_mass_massive
    return result


# ------------------------------------------------------------------ #
#additional functions
# ------------------------------------------------------------------ #
def imf_representative_mass(imf, M_threshold):
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