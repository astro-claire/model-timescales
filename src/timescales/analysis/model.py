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
from .recipes import per_system_comparison, destructive_colllision_criterion
from ..physics.stars import main_sequence_lifetime_approximation, stellar_radius_approximation  
from ..physics.halo_environment import local_merger_timescale, neighbor_merger_timescale, interaction_timescale
from ..utils.energy import escape_velocity
from ..utils.filtering import filter_kwargs_for

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
    timescales_by_radius = timescale_table(ensemble, verbose = verbose)
    denclosedmass_byradius = structural_table(ensemble, fields = ("dMencdR"))
    massloss_byradius =  masslosstable = destructive_colllision_criterion(ensemble)

    #initialize a table of outputs. One row for each system (bulk)
    out = Table = {
        "system_id": [],
        "mass": list(ensemble.grid['M']),
        "radius": list(ensemble.grid['R']),
        "velocity": list(ensemble.grid['V']),
        "kinetic": list(ensemble.grid['K']),
        "potential": list(ensemble.grid['U']),
        "t_ms": [],
        "t_merger":[], 
        "coll_occur_within_tmin": [],
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
    minimum_disruption_time = []
    #now, iterate through all the systems to create the minimum disruption timescales
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
    f_IMF_m = ensemble.imf.mass_fraction(ensemble.Mstar,ensemble.Mstar + deltamstar )

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
    print(out['N_collisions_massloss'])


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
        halomass = ensemble.grid['M'][sys_id] /10
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
