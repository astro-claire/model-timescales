"""
Model for each system! 
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import warnings
from .tables import structural_table, timescale_table
from .tools import get_system, condition_test
from .integrals import Ncoll_pl_no_bh,Ncoll_pl_no_bh_limits,N_coll_bh_limits
from .recipes import per_system_comparison, destructive_colllision_criterion
from ..physics.stars import main_sequence_lifetime_approximation, stellar_radius_approximation  
from ..physics.halo_environment import local_merger_timescale, neighbor_merger_timescale, interaction_timescale
from ..physics.collisions import collision_timescale
from ..utils.energy import escape_velocity
from ..utils.filtering import filter_kwargs_for


def create_dynamical_model_integral(ensemble,*,
                        deltamstar = 0.5*u.Msun,
                        as_: Literal["dict", "pandas"] = "dict",
                        verbose = True, 
                        z_final = 0,
                        mass_fraction_retained = .01
                            ):
    """ 
    Create dynamical model using exact integral
    """
    #get per radius information
    timescales_by_radius = timescale_table(ensemble, include=("t_relax","t_coll","t_df"))
    denclosedmass_byradius = structural_table(ensemble, fields = ("Menc","dMencdR","sigma"))
    massloss_byradius = destructive_colllision_criterion(ensemble)
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
    if 'redshift' not in all_possible_kwargs.keys():
        all_possible_kwargs['redshift']=12
        if verbose == True: 
            print("Using default z = 12 since no redshift provided")
    if 'cosmology' not in all_possible_kwargs.keys():
        all_possible_kwargs['cosmology']=FlatLambdaCDM(71,0.27,Ob0=0.044, Tcmb0=2.726 *u.K)
        if verbose == True:
            print("No cosmology provided. Initializing flat LCDM with H0 = 71,Om = 0.27, Ob0 = 0.044,Tcmb0=2.726 ")
    if 'int_type' not in all_possible_kwargs.keys():
        all_possible_kwargs['int_type'] = 'neighbor'
        if verbose ==True:
            print("Using nearest neighbor for interaction type.")

    #quantities that only need to be calculated once
    t_universe = all_possible_kwargs['cosmology'].age(z_final).to('yr')
    f_IMF_m = ensemble.imf.mass_fraction(ensemble.Mstar,ensemble.Mstar + deltamstar )

    #now, iterate through all the systems to create the minimum disruption timescales
    minimum_disruption_time = []
    which_disruption_time = []
    for sys_id in range(ensemble.Nsystems):
        #calculate disruptive timescales:
        #First, main sequence
        t_ms = main_sequence_lifetime_approximation(ensemble.timescales_kwargs['Mstar']).to('yr')
        out['t_ms'].append(t_ms)
        #next, interaction timescale:
        t_merger = _get_t_merger(all_possible_kwargs,sys_id, ensemble, verbose = verbose).to('yr')
        out['t_merger'].append(t_merger)
        #find out what's the limiting time for the system:
        minimum_disruption_time.append(min([t_merger,t_ms,t_universe]))
        which_disruption_time.append([t_merger,t_ms,t_universe].index(min([t_merger,t_ms,t_universe])))
    out['which_disruption_time']=which_disruption_time
    out['minimum_disruption_time']= minimum_disruption_time
    comparison =per_system_comparison(timescales_by_radius, 't_coll', 'lt', value = minimum_disruption_time)
    out['coll_occur_within_tmin'] = comparison['condition']

    coll_sys_id = np.where(out['coll_occur_within_tmin'])[0]
    print("collisions occur in "+str(len(coll_sys_id))+" systems")
    #output empty lists
    out['N_collisions'] = [0]* ensemble.Nsystems
    out['N_collisions_massloss']= [0]* ensemble.Nsystems
    out['fraction_sticky'] =[0]* ensemble.Nsystems
    out['N_collisions_df'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction region
    out['N_collisions_df_massloss'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction + massloss region

    if 'Mcollisions' not in ensemble.timescales_kwargs.keys():
        ensemble.timescales_kwargs['Mcollisions'] = 1.0*u.Msun

    #setup rmin
    rmin = 2* stellar_radius_approximation(ensemble.timescales_kwargs["Mstar"])

    for sys_id in range(ensemble.Nsystems):
        prof = ensemble.profiles[sys_id]
        cv = prof.get_veldisp_constant()
        ts = minimum_disruption_time[sys_id]
        if "BH" in ensemble.densityModel:
            out['N_collisions'][sys_id] = N_coll_bh_limits(prof.r0, #TODO FIX THE BH FUNCTION
                                    ts, 
                                    prof.alpha, 
                                    cv,
                                    prof.rho0,
                                    f_IMF_m,
                                    ensemble.profile_kwargs['M_bh'],
                                    Mstar =  ensemble.timescales_kwargs["Mstar"],
                                    Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                    rmin =rmin,
                                    e = ensemble.timescales_kwargs["e"])                  
        else:
            out['N_collisions'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0,
                                    ts, 
                                    prof.alpha, 
                                    cv,
                                    prof.rho0,
                                    f_IMF_m,
                                    Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                    Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                    rmin =rmin,
                                    e =  ensemble.timescales_kwargs["e"])
        sys_massloss = get_system(massloss_byradius, sys_id)
        massloss = np.array(sys_massloss['massloss'])
        ml_idx = np.where(massloss==1)[0]
        if len(ml_idx)>1:
            where_ml_cutoff = ml_idx[-1]
            radiusml = sys_massloss['r'][where_ml_cutoff] 
            out['N_collisions_massloss'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0, #TODO you need to add in the no bh here
                                    ts, 
                                    prof.alpha, 
                                    cv,
                                    prof.rho0,
                                    f_IMF_m,
                                    Mstar = ensemble.timescales_kwargs["Mstar"],
                                    Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                    e = ensemble.timescales_kwargs["e"],
                                    rmin =rmin,
                                    rmax = radiusml)
        #That was total ncol, let's consider where dynamical friction will bring sticky spheres to the middle
        sys_df = get_system(timescales_by_radius,sys_id)
        sticky_tdf= sys_df['stickytdf'] * u.yr
        where_stickydf = np.where(sticky_tdf<ts)[0]
        if len(where_stickydf>1):
            out['fraction_sticky'] = float(where_stickydf[-1])/ensemble.Nsampling
            r_stickydf = ensemble.radii[sys_id][where_stickydf[-1]] #rmax of dynamical friction region
            if "BH" in ensemble.densityModel:
                out['N_collisions_df'][sys_id] = N_coll_bh_limits(prof.r0,
                                        ts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        ensemble.profile_kwargs['M_bh'],
                                        Mstar =  ensemble.timescales_kwargs["Mstar"],
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmin =rmin,
                                        rmax =r_stickydf,
                                        e = ensemble.timescales_kwargs["e"])
            else:
                out['N_collisions_df'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0,
                                        ts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_stickydf,
                                        rmin =rmin,
                                        e = ensemble.timescales_kwargs["e"])        
            if len(ml_idx)>1:
                where_ml_cutoff = ml_idx[-1]
                radiusml = sys_massloss['r'][where_ml_cutoff] 
                out['N_collisions_df_massloss'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0, #TODO you need to add in the no bh here
                                        ts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        e = ensemble.timescales_kwargs["e"],
                                        rmin =rmin,
                                        rmax = min(r_stickydf,radiusml))
    whereml, = np.where(np.array(out['N_collisions_massloss'])>1)
    print("mass loss occurs in "+str(len(whereml))+" systems")
    print(out['N_collisions'])
    out['N_collisions_constructive'] = np.array(out['N_collisions_df'])-np.array(out['N_collisions_df_massloss'])
    superstar= np.array([out['N_collisions_constructive'][i] * mass_fraction_retained for i in range(len(out['N_collisions_constructive']))])
    out['M_superstar'] = superstar *u.Msun
    gasmass = [out['N_collisions_constructive'][i] * (1-mass_fraction_retained) + out['N_collisions_df_massloss'][i] for i in range(len(out['N_collisions_constructive']))]
    out['Mgas'] = gasmass * u.Msun
    return out



def create_dynamical_model(ensemble,*,
                        deltamstar = 0.5*u.Msun,
                        as_: Literal["dict", "pandas"] = "dict",
                        verbose = True
                            ):
    """ 
    Create the dynamical model for the systems 
        sum_1^N N(r) * 1/M * fIMF_M * dMdr* delta r
    """
    #get per radius information
    timescales_by_radius = timescale_table(ensemble, include=("t_relax","t_coll","t_df"))
    denclosedmass_byradius = structural_table(ensemble, fields = ("Menc","dMencdR","sigma"))
    massloss_byradius = destructive_colllision_criterion(ensemble)

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
    if 'redshift' not in all_possible_kwargs.keys():
        all_possible_kwargs['redshift']=12
        if verbose == True: 
            print("Using default z = 12 since no redshift provided")
    if 'cosmology' not in all_possible_kwargs.keys():
        all_possible_kwargs['cosmology']=FlatLambdaCDM(71,0.27,Ob0=0.044, Tcmb0=2.726 *u.K)
        if verbose == True:
            print("No cosmology provided. Initializing flat LCDM with H0 = 71,Om = 0.27, Ob0 = 0.044,Tcmb0=2.726 ")
    if 'int_type' not in all_possible_kwargs.keys():
        all_possible_kwargs['int_type'] = 'neighbor'
        if verbose ==True:
            print("Using nearest neighbor for interaction type.")

    #quantities that only need to be calculated once
    t_universe = all_possible_kwargs['cosmology'].age(0).to('yr')
    f_IMF_m = ensemble.imf.mass_fraction(ensemble.Mstar,ensemble.Mstar + deltamstar )

    #now, iterate through all the systems to create the minimum disruption timescales
    minimum_disruption_time = []
    for sys_id in range(ensemble.Nsystems):
        #calculate disruptive timescales:
        #First, main sequence
        t_ms = main_sequence_lifetime_approximation(ensemble.timescales_kwargs['Mstar']).to('yr')
        out['t_ms'].append(t_ms)
        #next, interaction timescale:
        t_merger = _get_t_merger(all_possible_kwargs,sys_id, ensemble, verbose = verbose).to('yr')
        out['t_merger'].append(t_merger)
        #find out what's the limiting time for the system:
        minimum_disruption_time.append(min([t_merger,t_ms,t_universe]))
    comparison =per_system_comparison(timescales_by_radius, 't_coll', 'lt', value = minimum_disruption_time)
    out['coll_occur_within_tmin'] = comparison['condition']

    coll_sys_id = np.where(out['coll_occur_within_tmin'])[0]
    print("collisions occur in "+str(len(coll_sys_id))+" systems")
    out['N_collisions'] = [0]* ensemble.Nsystems
    out['N_collisions_massloss']= [0]* ensemble.Nsystems

    #now we need to integrate how many  collisions occured in each system
    if len(coll_sys_id)>1:
        for sys_id in coll_sys_id:
            #get all the necessary data for this system
            sys_data = get_system(timescales_by_radius,sys_id)
            sys_dmdr = get_system(denclosedmass_byradius, sys_id)
            sys_massloss = get_system(massloss_byradius, sys_id)
            t_disrupt = minimum_disruption_time[sys_id].to_value('yr')
            #the next line calculates the number of collisions per star at each radius
            Ncol_M_r= [t_disrupt/i.to_value(u.yr) for i in sys_data['t_coll']]
            #assemble the summation
            Ncol_M_r = np.array(Ncol_M_r) 
            col_idx = np.where(Ncol_M_r>1)[0]
            dMdr = sys_dmdr['dMencdR'] * (u.Msun / u.pc)
            summation_component = Ncol_M_r * dMdr * 1.0/ensemble.Mstar * f_IMF_m
            summation_component = summation_component[col_idx]
            Ncol_M_r = Ncol_M_r[col_idx]
            radii = sys_data['r'] * u.pc
            radii = radii[col_idx]
            dr = np.diff(radii)
            # perform the summation
            total_N_collisions = np.sum(summation_component[:-1]*dr)
            out['N_collisions'][sys_id] = total_N_collisions.value
            #Now look at the mass loss 
            massloss = np.array(sys_massloss['massloss'])
            ml_idx = np.where(massloss[col_idx]==1)[0]
            if len(ml_idx)>1:
                summation_component = summation_component[ml_idx]
                radii = radii[ml_idx]
                dr = np.diff(radii)
                num_ML_collisions = np.sum(summation_component[:-1]*dr)
                out['N_collisions_massloss'][sys_id] = num_ML_collisions.value

    #now, let's investigate the dynamical friction
    #The minimum disruption time is different because we need to use the massive star's timescale instead
    if 'M_obj' in ensemble.timescales_kwargs.keys():
        M_obj = ensemble.timescales_kwargs['M_obj']
    else:
        print("using default M_obj: 10 Msun")
        M_obj= 10.0* u.Msun
    f_IMF_Mobj = ensemble.imf.mass_fraction(M_obj-deltamstar,M_obj + deltamstar )

    #Now we have the mass & mass fraction of the massive stars. Let's do the minimum calculation
    minimum_disruption_time = []
    for sys_id in range(ensemble.Nsystems):
        #calculate disruptive timescales:
        #First, main sequence
        t_ms = main_sequence_lifetime_approximation(M_obj).to('yr')
        #next, interaction timescale (don't need to recalculate)
        t_merger = out["t_merger"][sys_id]
        #find out what's the limiting time for the system:
        minimum_disruption_time.append(min([t_merger,t_ms,t_universe]))
    comparison =per_system_comparison(timescales_by_radius, 't_df', 'lt', value = minimum_disruption_time, return_where =True)
    out['df_occur_within_tmin'] = comparison['condition']

    df_sys_id = np.where(out['df_occur_within_tmin'])[0]
    out['t_coll_massive'] = [0]*ensemble.Nsystems
    out['total_N_coll_massive'] = [0]*ensemble.Nsystems
    radius_wheredf = comparison["where_true"]
    print("mass segregation occurs in "+str(len(df_sys_id))+" systems")
    if len(df_sys_id)>1:
        for sys_id in df_sys_id:
            sys_dmdr = get_system(denclosedmass_byradius, sys_id)
            final_idx = radius_wheredf[sys_id][-1]
            Menc_id = sys_dmdr["Menc"][final_idx]
            # the next line counts the number of 
            Nms = Menc_id * f_IMF_Mobj/M_obj
            core_radius_id = int(len(sys_dmdr['r'])/10.)
            new_core_radius = sys_dmdr['r'][core_radius_id]
            core_volume = 4./3. * np.pi * new_core_radius**3
            core_velocity = sys_dmdr['sigma'][core_radius_id]
            # t_coll_core_MS = collision_timescale(Nms/core_volume,core_velocity, M_obj, Mcollisions=M_obj).to('yr')
            t_coll_core_MS = collision_timescale(Nms/core_volume,core_velocity, ensemble.timescales_kwargs['Mstar'], Mcollisions=M_obj).to('yr')
            out['t_coll_massive'][sys_id] = t_coll_core_MS

            N_coll= minimum_disruption_time[sys_id]/t_coll_core_MS
            out['total_N_coll_massive'][sys_id]= N_coll * Nms

    if as_ == "dict":
        return out

    # Optional pandas return
    try:
        import pandas as pd  # local import to keep pandas optional
    except Exception as e:
        raise ImportError('pandas is required when as_="pandas". Install with `pip install pandas`.') from e

    # Build a DataFrame without stripping units (object dtype for Quantity columns)
    return pd.DataFrame(out)

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
