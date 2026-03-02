"""
Model for each system! 
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import warnings
from .tables import structural_table, timescale_table
from .tools import get_system, condition_test
from .integrals import Ncoll_pl_no_bh,Ncoll_pl_no_bh_limits,N_coll_bh_limits,Mdot_pl_no_bh_limits, Mdot_binaries_pl_limits, Mdot_deplete_noBH_limits
from .recipes import per_system_comparison, destructive_colllision_criterion
from ..physics.stars import main_sequence_lifetime_approximation, stellar_radius_approximation  
from ..physics.halo_environment import local_merger_timescale, neighbor_merger_timescale, interaction_timescale
from ..physics.collisions import collision_timescale
from ..physics.relaxation import r_no_relax, r_no_relax_bh
from ..utils.energy import escape_velocity
from ..physics.dynamical_friction import stellar_df_radius, stellar_df_time, bh_df_radius, bh_df_time
from ..physics.blackhole import sphere_of_influence, tidal_radius
from ..utils.filtering import filter_kwargs_for
from .tools import select_coulomb_calculator
from ..utils.energy import escape_velocity

def create_dynamical_model_integral(ensemble,*,
                        deltamstar = 0.5*u.Msun,
                        as_: Literal["dict", "pandas"] = "dict",
                        verbose = True, 
                        z_final = 0,
                        mass_accretion_ratio = 0.5,
                        f_vms = 2e-2, 
                        # t_dist_cc = None,
                        timescale_override = None
                            ):
    """ 
    Create dynamical model using exact integral
    """
    #get per radius information
    timescales_by_radius = timescale_table(ensemble, include=("t_relax","t_coll","t_df"))
    denclosedmass_byradius = structural_table(ensemble, fields = ("Menc","dMencdR","sigma"))
    # massloss_byradius = destructive_colllision_criterion(ensemble)
    timescales_by_radius['stickytdf'] = timescale_table(ensemble,include = ("t_df"),override_args={'M_obj':2*ensemble.Mstar})["t_df"]
    # sticky_df_byradius = timescale_table(ensemble, override_args={'M_obj':2*ensemble.Mstar})
    
    #initialize a table of outputs. One row for each system (bulk)
    out = Table = {
        "mass": list(ensemble.grid['M']),
        "radius": list(ensemble.grid['R']),
        "velocity": list(ensemble.grid['V']),
        "kinetic": list(ensemble.grid['K']),
        "potential": list(ensemble.grid['U']),
        "t_ms": [],
        "t_merger":[], 
        }

    #set up kwargs for the timescale functions
    all_possible_kwargs = ensemble.timescales_kwargs | ensemble.profile_kwargs
    #these are needed as positional arguments for the halo calculations
    all_possible_kwargs['redshift']=z_final
    # if 'redshift' not in all_possible_kwargs.keys():
    #     all_possible_kwargs['redshift']=12
    #     if verbose == True: 
    #         print("Using default z = 12 since no redshift provided")
    if 'cosmology' not in all_possible_kwargs.keys():
        all_possible_kwargs['cosmology']=FlatLambdaCDM(71,0.27,Ob0=0.044, Tcmb0=2.726 *u.K)
        if verbose == True:
            print("No cosmology provided. Initializing flat LCDM with H0 = 71,Om = 0.27, Ob0 = 0.044,Tcmb0=2.726 ")
    if 'int_type' not in all_possible_kwargs.keys():
        all_possible_kwargs['int_type'] = 'neighbor'
        if verbose ==True:
            print("Using nearest neighbor for interaction type.")
    if 'Z' not in ensemble.imf_kwargs:
        if verbose ==True:
            print("No Metallicity given! Using 0.1 solar")
        ensemble.imf_kwargs['Z']=0.1
    coulomb_func = select_coulomb_calculator(ensemble)

    #quantities that only need to be calculated once
    t_universe = all_possible_kwargs['cosmology'].age(z_final).to('yr')
    f_IMF_m = ensemble.imf.mass_fraction(ensemble.Mstar,ensemble.Mstar + deltamstar )

    #now, iterate through all the systems to create the minimum disruption timescales
    minimum_disruption_time = []
    which_disruption_time = []
    print(f"Getting disruption times for {ensemble.Nsystems} systems")
    for sys_id in range(ensemble.Nsystems):
        #calculate disruptive timescales:
        #First, main sequence
        t_ms = main_sequence_lifetime_approximation(ensemble.timescales_kwargs['Mstar']).to('yr')
        out['t_ms'].append(t_ms)
        #next, interaction timescale:
        t_merger = _get_t_merger(all_possible_kwargs,sys_id, ensemble, verbose = verbose).to('yr')
        out['t_merger'].append(t_merger)
        #find out what's the limiting time for the system:
        disrupt_list = [t_merger,t_ms,t_universe]
        tsc = get_system(timescales_by_radius, sys_id)
        t_relax = tsc['t_relax'][-1]
        disrupt_list.append(t_relax*0.2)
        if timescale_override is not None: 
            disrupt_list.append(timescale_override)
        minimum_disruption_time.append(min(disrupt_list))
        which_disruption_time.append(disrupt_list.index(min(disrupt_list)))
    out['which_disruption_time']=which_disruption_time
    out['minimum_disruption_time']= minimum_disruption_time
    comparison =per_system_comparison(timescales_by_radius, 't_coll', 'lt', value = minimum_disruption_time)
    out['coll_occur_within_tmin'] = comparison['condition']

    coll_sys_id = np.where(out['coll_occur_within_tmin'])[0]
    #output empty lists
    out['N_collisions'] = [0]* ensemble.Nsystems
    out['rho0'] = [0]* ensemble.Nsystems
    out['M_BH'] = [0]* ensemble.Nsystems
    out['Mdot_BH'] = [0]* ensemble.Nsystems
    out['Mdot_Edd'] = [0]* ensemble.Nsystems
    out['M_VMS'] = [0]* ensemble.Nsystems #following Paccuci et al for this
    out['mass_fraction_retained'] = [0]* ensemble.Nsystems
    out['N_collisions_massloss']= [0]* ensemble.Nsystems
    out['fraction_sticky'] =[0]* ensemble.Nsystems
    out['N_collisions_df'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction region
    out['N_collisions_ml'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction region
    out['rhotot_ml'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction region
    out['N_collisions_df_massloss'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction + massloss region
    out['mass_accretion_rate']=[0] * ensemble.Nsystems
    out['mass_df_rate']=[0] * ensemble.Nsystems
    out['mass_depletion_rate']=[0] * ensemble.Nsystems
    out['mass_binaries_rate']=[0] * ensemble.Nsystems
    if 'Mcollisions' not in ensemble.timescales_kwargs.keys():
        ensemble.timescales_kwargs['Mcollisions'] = 1.0*u.Msun


    print(f"Integrating outputs for {ensemble.Nsystems} systems")
    for sys_id in range(ensemble.Nsystems):
        #system_properties
        prof = ensemble.profiles[sys_id]
        sys_data = get_system(denclosedmass_byradius, sys_id)
        coulomb_log = coulomb_func(sys_data['Menc'][-1],sys_data['r'][-1],sys_data["sigma"][-1], Mstar = ensemble.timescales_kwargs["Mstar"])
        cv=prof.get_veldisp_constant()
        out['rho0'][sys_id]= prof.rho0

        #disruption time
        ts = minimum_disruption_time[sys_id]
        #reset the disruption time -FIXME
        newts = min(ts, main_sequence_lifetime_approximation(2*ensemble.Mstar))
        #collision properties:
        Rcollisions = stellar_radius_approximation(ensemble.timescales_kwargs["Mcollisions"])
        reduced_mass = (ensemble.timescales_kwargs['Mcollisions']*ensemble.timescales_kwargs['Mstar'])/(ensemble.timescales_kwargs['Mcollisions']+ensemble.timescales_kwargs['Mstar'])

        if "BH" in ensemble.densityModel:
            out['M_BH'][sys_id]= ensemble.profile_kwargs['M_bh']
            #first we need to decide on the radii structure
            r_soi = sphere_of_influence(prof.alpha,prof.r0,prof.rho0, ensemble.profile_kwargs['M_bh'])
            t_df_bh = bh_df_time(r_soi,ensemble.timescales_kwargs["Mstar"],coulomb_log,ensemble.timescales_kwargs['Mcollisions'],prof.alpha,prof.r0,prof.rho0, ensemble.profile_kwargs['M_bh'],cv=prof.get_veldisp_constant())
            t_df_stars = stellar_df_time(r_soi,ensemble.timescales_kwargs["Mstar"],coulomb_log,ensemble.timescales_kwargs['Mcollisions'],prof.alpha,prof.r0,prof.rho0,cv=prof.get_veldisp_constant())
            if t_df_bh<t_df_stars:
                r_df = stellar_df_radius(ts, ensemble.timescales_kwargs["Mstar"],coulomb_log,ensemble.timescales_kwargs['Mcollisions'],prof.alpha,prof.r0,prof.rho0, cv=prof.get_veldisp_constant())
            else:
                r_df = bh_df_radius(ts, ensemble.timescales_kwargs["Mstar"],coulomb_log,ensemble.timescales_kwargs['Mcollisions'],prof.alpha,prof.r0,prof.rho0,ensemble.profile_kwargs['M_bh'], cv=prof.get_veldisp_constant())
            r_tidal = tidal_radius(ensemble.profile_kwargs['M_bh'], ensemble.timescales_kwargs["Mstar"]) #FIXME with the collisions raduys
            rmin = max(r_tidal, 2*stellar_radius_approximation(ensemble.timescales_kwargs["Mstar"]))
            r_ml = cv *c.G * ensemble.profile_kwargs['M_bh']/(escape_velocity(ensemble.timescales_kwargs["Mstar"],stellar_radius_approximation(ensemble.timescales_kwargs["Mstar"])))**2/ (1+prof.alpha)
            # print(r_ml.to('pc'))
            # rmin = 1e-2 * u.pc
            #Calculated this radius - this would be for Dynamical friction but that's not currently implemented
            rmin_relax = r_no_relax_bh(prof.rho0, prof.r0, ensemble.timescales_kwargs["Mstar"], coulomb_log, cv, prof.alpha,ensemble.profile_kwargs['M_bh'])
            out['N_collisions_df'][sys_id] = N_coll_bh_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        ensemble.profile_kwargs['M_bh'],
                                        Mstar =  ensemble.timescales_kwargs["Mstar"],
                                        Mcollisions=2 *ensemble.timescales_kwargs['Mcollisions'], 
                                        rmin =rmin,
                                        rmax =r_df,
                                        e = ensemble.timescales_kwargs["e"])
            if r_ml>rmin:
                out['N_collisions_ml'][sys_id] = N_coll_bh_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        ensemble.profile_kwargs['M_bh'],
                                        Mstar =  ensemble.timescales_kwargs["Mstar"],
                                        Mcollisions=2 *ensemble.timescales_kwargs['Mcollisions'], 
                                        rmin =rmin,
                                        rmax =r_ml,
                                        e = ensemble.timescales_kwargs["e"])
                mass_encl= prof.enclosed_mass(r_ml)
                if out['N_collisions_ml'][sys_id] *u.Msun >mass_encl:
                    masscoll = 0.5*mass_encl
                else:
                    masscoll = out['N_collisions_ml'][sys_id] *u.Msun
                out['rhotot_ml'][sys_id]=masscoll / (4*np.pi/3.*r_ml**3)

                #Bondi Hoyle accretion onto the bh
                c_s = np.sqrt(5./3. /3.* prof.velocity_dispersion(r_tidal)**2) #Take value inside sphere of influence
                out['Mdot_BH'][sys_id] = np.pi * out['rhotot_ml'][sys_id] * c.G**2 * ensemble.profile_kwargs['M_bh']**2/(c_s**3)
                out['Mdot_Edd'][sys_id] = 1e-8 * ( ensemble.profile_kwargs['M_bh'].to('Msun'))/u.yr
            else:
                out['rhotot_ml'][sys_id] = 0*u.g/(u.cm**3)
        else: #STAR ONLY CASE
            out['M_BH'][sys_id]= 0 *u.Msun
            #first we need to calculate the relevant radii
            r_df = stellar_df_radius(ts, ensemble.timescales_kwargs["Mstar"],coulomb_log,ensemble.timescales_kwargs['Mcollisions'],prof.alpha,prof.r0,prof.rho0, cv=prof.get_veldisp_constant())
            rmin = 20000* stellar_radius_approximation(ensemble.timescales_kwargs["Mstar"])
            rtest = stellar_df_radius(10*u.yr, ensemble.timescales_kwargs["Mstar"],coulomb_log,ensemble.timescales_kwargs['Mcollisions'],prof.alpha,prof.r0,prof.rho0, cv=prof.get_veldisp_constant())
            # print(rtest)
            rmin = rtest
            rmin = r_no_relax(prof.rho0, prof.r0, ensemble.timescales_kwargs["Mstar"], coulomb_log, cv, prof.alpha)
            # rmin = 2* stellar_radius_approximation(ensemble.timescales_kwargs["Mstar"])
            # rmin = 1e-2 * u.pc
            out['N_collisions_df'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_df,
                                        rmin =rmin,
                                        e = ensemble.timescales_kwargs["e"])
            out['mass_df_rate'][sys_id] = mass_accretion_ratio* Mdot_pl_no_bh_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        reduced_mass, 
                                        coulomb_log,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_df,
                                        rmin =rmin,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
            out['mass_binaries_rate'][sys_id]= Mdot_binaries_pl_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        reduced_mass, 
                                        coulomb_log,
                                        prof.enclosed_mass(r_df),
                                        r_df,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_df,
                                        rmin =rmin,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
            out['mass_depletion_rate'][sys_id] = Mdot_deplete_noBH_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        reduced_mass, 
                                        coulomb_log,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_df,
                                        rmin =rmin,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
            # if out['mass_depletion_rate'][sys_id]<((ensemble.timescales_kwargs["Mstar"]/(ensemble.timescales_kwargs["Mstar"]+ensemble.timescales_kwargs['Mcollisions'])).cgs*out['mass_df_rate'][sys_id]):
            #     out['mass_depletion_rate'][sys_id]= ((ensemble.timescales_kwargs["Mstar"]/(ensemble.timescales_kwargs["Mstar"]+ensemble.timescales_kwargs['Mcollisions'])).cgs*out['mass_df_rate'][sys_id])
            out['mass_accretion_rate'][sys_id]= (1-f_vms)*(out['mass_df_rate'][sys_id]-out['mass_depletion_rate'][sys_id]-out['mass_binaries_rate'][sys_id])
            if out['mass_accretion_rate'][sys_id]==0:
                out['mass_accretion_rate'][sys_id]= 0 *u.Msun/u.yr 
            Z = ensemble.imf_kwargs['Z']
            out['M_VMS'][sys_id]= (out['mass_accretion_rate'][sys_id].to_value(u.Msun/u.yr)/(10**(-9.13)*Z**0.74))**(1./2.1)* u.Msun

    return out


def _get_t_merger(all_possible_kwargs, sys_id, ensemble, verbose = True):
        halomass = max([ensemble.grid['M'][sys_id] /10, 1e4 * u.Msun])
        cosmo = all_possible_kwargs['cosmology']
        z = all_possible_kwargs['redshift']
        if all_possible_kwargs['int_type'] =="neighbor":
            int_kwargs, missing = filter_kwargs_for(neighbor_merger_timescale, all_possible_kwargs)
            return neighbor_merger_timescale(ensemble.grid['M'][sys_id], 
                                            ensemble.grid['R'][sys_id], 
                                            z, 
                                            halomass,
                                            cosmo, 
                                            **int_kwargs)
        elif all_possible_kwargs['int_type']== "local":
            int_kwargs, missing = filter_kwargs_for(local_merger_timescale, all_possible_kwargs)
            return local_merger_timescale(ensemble.grid['M'][sys_id], 
                                            ensemble.grid['R'][sys_id], 
                                            z, 
                                            halomass,
                                            cosmo, 
                                            **int_kwargs)
        elif all_possible_kwargs['int_type']=="general":
            int_kwargs, missing = filter_kwargs_for(interaction_timescale, all_possible_kwargs)
            return interaction_timescale(ensemble.grid['M'][sys_id], 
                                            ensemble.grid['R'][sys_id], 
                                            z, 
                                            halomass,
                                            cosmo, 
                                            **int_kwargs)
