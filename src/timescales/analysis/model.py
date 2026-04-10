"""
Model for each system! 
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.units import Quantity
from astropy.cosmology import FlatLambdaCDM
from typing import Literal, Optional
import warnings
from .tables import structural_table, timescale_table
from .tools import get_system
from .integrals import Ncoll_pl_no_bh_limits,N_coll_bh_limits,Mdot_pl_no_bh_limits, Mdot_binaries_pl_limits, Mdot_deplete_noBH_limits,Mdot_df_withbh
from .recipes import per_system_comparison, destructive_collision_criterion
from ..physics.stars import main_sequence_lifetime_approximation, stellar_radius_approximation
from ..physics.halo_environment import local_merger_timescale, neighbor_merger_timescale, interaction_timescale
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
    massloss_byradius = destructive_collision_criterion(ensemble)
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
            t_relax = tsc['t_relax'][-1] #should I change this to the half mass radius? 
            disrupt_list.append(t_relax*0.2)
        if timescale_override is not None: 
            disrupt_list.append(timescale_override)
        minimum_disruption_time.append(min(disrupt_list))
        which_disruption_time.append([t_merger,t_ms,t_universe].index(min([t_merger,t_ms,t_universe])))
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

    #setup rmin
    rmin = 2* stellar_radius_approximation(ensemble.timescales_kwargs["Mstar"])
    Rcollisions = stellar_radius_approximation(ensemble.timescales_kwargs["Mcollisions"])
    for sys_id in range(ensemble.Nsystems):
        # denclosedmass_byradius[sys_id]
        # print(get_system(denclosedmass_byradius, sys_id)['sigma'])
        sigma_avg = np.mean(get_system(denclosedmass_byradius, sys_id)['sigma']*(u.km/u.s))
        reduced_mass = (ensemble.timescales_kwargs['Mcollisions']*ensemble.timescales_kwargs['Mstar'])/(ensemble.timescales_kwargs['Mcollisions']+ensemble.timescales_kwargs['Mstar'])
        GMR1 = c.G*ensemble.timescales_kwargs['Mstar']**2/(rmin/2)
        GMR2 = c.G*ensemble.timescales_kwargs['Mcollisions']**2/(Rcollisions)
        out['mass_fraction_retained'][sys_id] = (reduced_mass *sigma_avg**2/(GMR1+GMR2)).cgs.value
        prof = ensemble.profiles[sys_id]
        out['rho0'][sys_id]= prof.rho0
        cv = prof.get_veldisp_constant()
        ts = minimum_disruption_time[sys_id]
        #------------
        #Calculate the base number of collisions in the whole object
        #------------
        if "BH" in ensemble.densityModel:
            out['N_collisions'][sys_id] = N_coll_bh_limits(prof.r0, 
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
        #------------
        #Calculate whether the system has mass loss - if so calculate the number of collisions that contribute to mass loss
        #------------
        sys_massloss = get_system(massloss_byradius, sys_id) #TODO change this to the analytic formula?
        massloss = np.array(sys_massloss['massloss'])
        ml_idx = np.where(massloss==1)[0]
        if len(ml_idx)>1:
            where_ml_cutoff = ml_idx[-1]
            radiusml = sys_massloss['r'][where_ml_cutoff] 
            if "BH" in ensemble.densityModel:
                out['N_collisions_massloss'][sys_id] = N_coll_bh_limits(prof.r0, 
                                        ts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        ensemble.profile_kwargs['M_bh'],
                                        Mstar =  ensemble.timescales_kwargs["Mstar"],
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmin =rmin,
                                        e = ensemble.timescales_kwargs["e"],
                                        rmax = radiusml)                  
            else:
                out['N_collisions_massloss'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0, 
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
        #------------
        #Calculate the dynamical friction radius 
        #------------        
        sys_df = get_system(timescales_by_radius,sys_id)
        sticky_tdf= sys_df['stickytdf'] * u.yr
        newts = min(ts, main_sequence_lifetime_approximation(2*ensemble.Mstar))
        where_stickydf = np.where(sticky_tdf<newts)[0]
        relaxt= sys_df['t_relax'] * u.yr
        where_relax = np.where(relaxt>10*u.yr)[0]
        r_relax = ensemble.radii[sys_id][where_relax[0]]
        # print("relaxr", r_relax)
        if len(where_stickydf>1):
            out['fraction_sticky'][sys_id] = float(where_stickydf[-1])/ensemble.Nsampling
            r_stickydf = ensemble.radii[sys_id][where_stickydf[-1]] #rmax of dynamical friction region
            if len(ml_idx)>1:
                print((r_stickydf/radiusml).cgs)
                r_bhsphere = 1*u.pc
                if (r_stickydf/radiusml).cgs>1:
                    print("OUTSIDE OF ML RADIUS")
                if (r_stickydf/r_bhsphere).cgs>1:
                    print("DF OUTSIDE OF SPHERE ")
                else:
                    print("DF NOT OUT OF SPHERE")
            sys_data = get_system(denclosedmass_byradius, sys_id)
            coulomb_log = coulomb_func(sys_data['Menc'][-1], sys_data['r'][-1], sys_data["sigma"][-1], Mstar=ensemble.timescales_kwargs["Mstar"])
            if "BH" in ensemble.densityModel:
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
                                        rmax =r_stickydf,
                                        e = ensemble.timescales_kwargs["e"])
                out['mass_df_rate'][sys_id] = Mdot_df_withbh(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        ensemble.profile_kwargs['M_bh'],
                                        coulomb_log,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_stickydf,
                                        rmin =r_relax,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
                out['mass_binaries_rate'][sys_id]= Mdot_binaries_pl_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        reduced_mass, 
                                        coulomb_log,
                                        prof.enclosed_mass(r_stickydf),
                                        r_stickydf,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_stickydf,
                                        rmin =r_relax,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
                out['mass_depletion_rate'][sys_id] = Mdot_df_withbh(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        ensemble.profile_kwargs['M_bh'],
                                        coulomb_log,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_stickydf,
                                        rmin =r_relax,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
            else:
                out['N_collisions_df'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_stickydf,
                                        rmin =rmin,
                                        e = ensemble.timescales_kwargs["e"])
                #------------
                #Calculate mass migration/accretion rates
                #------------
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
                                        rmax =r_stickydf,
                                        rmin =r_relax,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
                out['mass_binaries_rate'][sys_id]= Mdot_binaries_pl_limits(prof.r0,
                                        newts, 
                                        prof.alpha, 
                                        cv,
                                        prof.rho0,
                                        f_IMF_m,
                                        reduced_mass, 
                                        coulomb_log,
                                        prof.enclosed_mass(r_stickydf),
                                        r_stickydf,
                                        Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                        Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                        rmax =r_stickydf,
                                        rmin =r_relax,
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
                                        rmax =r_stickydf,
                                        rmin =r_relax,
                                        e = ensemble.timescales_kwargs["e"]).to(u.Msun/u.yr)
                if out['mass_depletion_rate'][sys_id]<((ensemble.timescales_kwargs["Mstar"]/(ensemble.timescales_kwargs["Mstar"]+ensemble.timescales_kwargs['Mcollisions'])).cgs*out['mass_df_rate'][sys_id]):
                    out['mass_depletion_rate'][sys_id]= ((ensemble.timescales_kwargs["Mstar"]/(ensemble.timescales_kwargs["Mstar"]+ensemble.timescales_kwargs['Mcollisions'])).cgs*out['mass_df_rate'][sys_id])
                out['mass_accretion_rate'][sys_id]= (1-f_vms)*(out['mass_df_rate'][sys_id]-out['mass_depletion_rate'][sys_id]-out['mass_binaries_rate'][sys_id])
                # print(Mdottest.to(u.Msun/u.yr))
            if len(ml_idx)>1:
                where_ml_cutoff = ml_idx[-1]
                radiusml = sys_massloss['r'][where_ml_cutoff] 
                if "BH" in ensemble.densityModel:
                    out['N_collisions_df_massloss'][sys_id] = N_coll_bh_limits(prof.r0, 
                                            newts, 
                                            prof.alpha, 
                                            cv,
                                            prof.rho0,
                                            f_IMF_m,
                                            ensemble.profile_kwargs['M_bh'],
                                            Mstar =  ensemble.timescales_kwargs["Mstar"],
                                            Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                            rmin =rmin,
                                            e = ensemble.timescales_kwargs["e"],
                                            rmax = min(r_stickydf,radiusml))                  
                else:
                    out['N_collisions_df_massloss'][sys_id] = Ncoll_pl_no_bh_limits(prof.r0, 
                                            newts, 
                                            prof.alpha, 
                                            cv,
                                            prof.rho0,
                                            f_IMF_m,
                                            Mstar = ensemble.timescales_kwargs["Mstar"],#1.0*u.Msun,
                                            Mcollisions=ensemble.timescales_kwargs['Mcollisions'], 
                                            e = ensemble.timescales_kwargs["e"],
                                            rmin =rmin,
                                            rmax = min(r_stickydf,radiusml))
            if out['mass_accretion_rate'][sys_id]==0:
                out['mass_accretion_rate'][sys_id]= 0 *u.Msun/u.yr 
            Z = ensemble.imf_kwargs['Z']
            out['M_VMS'][sys_id]= (out['mass_accretion_rate'][sys_id].to_value(u.Msun/u.yr)/(10**(-9.13)*Z**0.74))**(1./2.1)* u.Msun
    whereml, = np.where(np.array(out['N_collisions_massloss'])>1)
    print("mass loss occurs in "+str(len(whereml))+" systems")
    print(len(np.where(np.array(out['N_collisions_df_massloss'])>1)[0]))
    print(out['M_VMS'])
    print(out['mass_accretion_rate'])
    out['fraction_collisions_df']= np.array(out['N_collisions_df'])/np.array(out['N_collisions'])
    out['N_collisions_constructive'] = np.array(out['N_collisions_df'])-np.array(out['N_collisions_df_massloss'])
    superstar= np.array([out['N_collisions_constructive'][i] * out['mass_fraction_retained'][i] for i in range(len(out['N_collisions_constructive']))])
    out['M_superstar'] = superstar *ensemble.timescales_kwargs["Mstar"]
    gasmass = [out['N_collisions_constructive'][i] * (1-out['mass_fraction_retained'][i]) + out['N_collisions_df_massloss'][i] for i in range(len(out['N_collisions_constructive']))]
    out['Mgas'] = gasmass * u.Msun
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
