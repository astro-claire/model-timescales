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
from ..utils.energy import escape_velocity
from ..physics.dynamical_friction import stellar_df_radius, stellar_df_time, bh_df_radius, bh_df_time
from ..physics.blackhole import sphere_of_influence
from ..utils.filtering import filter_kwargs_for
from .tools import select_coulomb_calculator

def create_dynamical_model_integral(ensemble,*,
                        deltamstar = 0.5*u.Msun,
                        as_: Literal["dict", "pandas"] = "dict",
                        verbose = True, 
                        z_final = 0,
                        mass_fraction_retained = .01, 
                        mass_accretion_ratio = 0.5,
                        f_vms = 2e-2, 
                        t_dist_cc = None,
                        timescale_override = None
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
        if t_dist_cc is not None: 
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
    print("collisions occur in "+str(len(coll_sys_id))+" systems")
    #output empty lists
    out['N_collisions'] = [0]* ensemble.Nsystems
    out['rho0'] = [0]* ensemble.Nsystems
    out['M_VMS'] = [0]* ensemble.Nsystems #following Paccuci et al for this
    out['mass_fraction_retained'] = [0]* ensemble.Nsystems
    out['N_collisions_massloss']= [0]* ensemble.Nsystems
    out['fraction_sticky'] =[0]* ensemble.Nsystems
    out['N_collisions_df'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction region
    out['N_collisions_df_massloss'] =[0]* ensemble.Nsystems # number of collisions in the dynamical friction + massloss region
    out['mass_accretion_rate']=[0] * ensemble.Nsystems
    out['mass_df_rate']=[0] * ensemble.Nsystems
    out['mass_depletion_rate']=[0] * ensemble.Nsystems
    out['mass_binaries_rate']=[0] * ensemble.Nsystems
    if 'Mcollisions' not in ensemble.timescales_kwargs.keys():
        ensemble.timescales_kwargs['Mcollisions'] = 1.0*u.Msun




    if "BH" in ensemble.densityModel:
        pass
    else:
        #first we need to calculate the relevant radii