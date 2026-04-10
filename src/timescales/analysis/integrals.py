import numpy as np
import scipy
from astropy.constants import G
from astropy.constants import c as c_sl
import astropy.units as u
from ..physics.stars import stellar_radius_approximation

def get_ecc_functions(e,alpha):
    #functions from Rose et al 2020
    f1 = (1.-e)**(0.5-alpha)*0.5*scipy.special.hyp2f1(
        0.5,alpha -0.5, 1., 2.*e/(e-1.)
        ) + (1.+e)**(0.5-alpha)*0.5*scipy.special.hyp2f1(
            0.5,alpha -0.5, 1., 2.*e/(e+1.)
            )

    f2 = (1.-e)**(1.5-alpha)*0.5*scipy.special.hyp2f1(
        0.5,alpha -1.5, 1., 2.*e/(e-1.)) + (1.+e)**(1.5-alpha)*0.5*scipy.special.hyp2f1(
            0.5,alpha -1.5, 1., 2.*e/(e+1.))
    return f1,f2


# DEPRECATED: not currently used (superseded by Ncoll_pl_no_bh_limits)
def Ncoll_pl_no_bh(r0,ts, alpha, cv,rho0,fimf,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0, rmin = 1000*u.Rsun):
    """
    Number of collisions for a no black hole power law system
    """
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 

    #the solution: alpha <1.33
    if alpha<1.33:
        #-------------
        prefactor = np.pi *ts * fimf * (3-alpha) / (Mstar**2)
        # eccentricity functions
        F1 = f1 * rc**2
        F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
        #term prefactors
        sqrt_term = cv * G / (1+alpha)
        rhoterm1 = (4.* np.pi)**(3./2.) * (rho0**(5./2.)) * (3.- alpha)**(-3./2.)
        rhoterm2 = (4.* np.pi)**(1./2.) * (rho0**(3./2.)) * (3.- alpha)**(-1./2.)
        # The main r0 terms
        r0term1 = (r0**(4)) / (4.-(5.*alpha/2))
        r0term2 = (r0**(2.)) /(2.-(3.*alpha/2))
        #put it together
        #------
        result = prefactor * ( (F1* np.sqrt(sqrt_term)*rhoterm1*r0term1)+(F2 *(sqrt_term)**(-1./2.)*rhoterm2 *r0term2))
        return result.cgs.value
    else:
        def integral(r):
            prefactor = np.pi *ts * fimf * (3-alpha) / (Mstar**2)
            # eccentricity functions
            F1 = f1 * rc**2
            F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
            #term prefactors
            sqrt_term = cv * G / (1+alpha)
            rhoterm1 = (4.* np.pi)**(3./2.) * (rho0**(5./2.)) / r0**(-5.*alpha/2.) * (3.- alpha)**(-3./2.)
            rhoterm2 = (4.* np.pi)**(1./2.) * (rho0**(3./2.)) / r0**(-3.*alpha/2.) * (3.- alpha)**(-1./2.)
            # The main r terms
            r0term1 = (r**(4.-(5.*alpha/2))) / (4.-(5.*alpha/2))
            r0term2 = (r**(2.-(3.*alpha/2))) /(2.-(3.*alpha/2))
            result = prefactor * ( (F1* np.sqrt(sqrt_term)*rhoterm1*r0term1).cgs+(F2 *(sqrt_term)**(-1./2.)*rhoterm2 *r0term2).cgs)
            return result.cgs.value

        return integral(r0)-integral(rmin.to(u.pc))



def Ncoll_pl_no_bh_limits(r0,ts, alpha, cv,rho0,fimf,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """
    Number of collisions for a no black hole power law system
    """
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 

    def integral(r):
        prefactor = np.pi *ts * fimf * (3-alpha) / (Mstar**2)
        # eccentricity functions
        F1 = f1 * rc**2
        F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
        #term prefactors
        sqrt_term = cv * G / (1+alpha)
        rhoterm1 = (4.* np.pi)**(3./2.) * (rho0**(5./2.)) / r0**(-5.*alpha/2.) * (3.- alpha)**(-3./2.)
        rhoterm2 = (4.* np.pi)**(1./2.) * (rho0**(3./2.)) / r0**(-3.*alpha/2.) * (3.- alpha)**(-1./2.)
        # The main r terms
        r0term1 = (r**(4.-(5.*alpha/2))) / (4.-(5.*alpha/2))
        r0term2 = (r**(2.-(3.*alpha/2))) /(2.-(3.*alpha/2))
        result = prefactor * ( (F1* np.sqrt(sqrt_term)*rhoterm1*r0term1).cgs+(F2 *(sqrt_term)**(-1./2.)*rhoterm2 *r0term2).cgs)
        return result.cgs.value
    def integral58(r):
        """alpha = 5/8. Special case integrals to handle discontinuities in the previous function's integral form"""
        prefactor = np.pi *ts * fimf * (3-alpha) / (Mstar**2)
        # eccentricity functions
        F1 = f1 * rc**2
        F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
        #term prefactors
        sqrt_term = cv * G / (1+alpha)
        rhoterm1 = (4.* np.pi)**(3./2.) * (rho0**(5./2.)) / r0**(-5.*alpha/2.) * (3.- alpha)**(-3./2.)
        rhoterm2 = (4.* np.pi)**(1./2.) * (rho0**(3./2.)) / r0**(-3.*alpha/2.) * (3.- alpha)**(-1./2.)
        # The main r terms
        r0term1 = np.log(r.to("pc").value)
        r0term2 = (r**(2.-(3.*alpha/2))) /(2.-(3.*alpha/2))
        result = prefactor * ( (F1* np.sqrt(sqrt_term)*rhoterm1*r0term1).cgs+(F2 *(sqrt_term)**(-1./2.)*rhoterm2 *r0term2).cgs)
        return result.cgs.value
    def integral43(r):
        """alphs = 4/3. Special case integrals to handle discontinuities in the previous function's integral form"""
        prefactor = np.pi *ts * fimf * (3-alpha) / (Mstar**2)
        # eccentricity functions
        F1 = f1 * rc**2
        F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
        #term prefactors
        sqrt_term = cv * G / (1+alpha)
        rhoterm1 = (4.* np.pi)**(3./2.) * (rho0**(5./2.)) / r0**(-5.*alpha/2.) * (3.- alpha)**(-3./2.)
        rhoterm2 = (4.* np.pi)**(1./2.) * (rho0**(3./2.)) / r0**(-3.*alpha/2.) * (3.- alpha)**(-1./2.)
        # The main r terms
        r0term1 = (r**(4.-(5.*alpha/2))) / (4.-(5.*alpha/2))
        r0term2 = np.log(r.to("pc").value)
        result = prefactor * ( (F1* np.sqrt(sqrt_term)*rhoterm1*r0term1).cgs+(F2 *(sqrt_term)**(-1./2.)*rhoterm2 *r0term2).cgs)
        return result.cgs.value
    if alpha == 1.6:
        return integral58(rmax.to("pc"))-integral58(rmin.to("pc"))
    elif np.isclose(alpha, (4./3.),1e-6) :
        return integral43(rmax.to("pc"))-integral43(rmin.to("pc"))
    else:
        return integral(rmax.to("pc"))-integral(rmin.to("pc")) 
    # return integral(rmax.to(u.pc))-integral(rmin.to(u.pc))

# DEPRECATED: not currently used
def N_coll_no_bh_integrand(r0,ts, alpha, cv,rho0,fimf,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """
    Number of collisions for a no black hole power law system INTEGRAND PORTION
    """
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 

    def integral(r):
        prefactor = np.pi *ts * fimf * (3-alpha) / (Mstar**2)
        # eccentricity functions
        F1 = f1 * rc**2
        F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
        #term prefactors
        sqrt_term = cv * G / (1+alpha)
        rhoterm1 = (4.* np.pi)**(3./2.) * (rho0**(5./2.)) / r0**(-5.*alpha/2.) * (3.- alpha)**(-3./2.)
        rhoterm2 = (4.* np.pi)**(1./2.) * (rho0**(3./2.)) / r0**(-3.*alpha/2.) * (3.- alpha)**(-1./2.)
        # The main r terms
        r0term1 = (r**(3.-(5.*alpha/2)))
        r0term2 = (r**(1.-(3.*alpha/2))) 
        result = prefactor * ( (F1* np.sqrt(sqrt_term)*rhoterm1*r0term1).cgs+(F2 *(sqrt_term)**(-1./2.)*rhoterm2 *r0term2).cgs)
        return result.cgs

    # if alpha == 1.6:
    #     return integral58(rmax.to("pc"))-integral58(rmin.to("pc"))
    # elif np.isclose(alpha, (4./3.),1e-6) :
    #     return integral43(rmax.to("pc"))-integral43(rmin.to("pc"))
    # else:
    return integral(rmax.to("pc"))
    # return integral(rmax.to(u.pc))-integral(rmin.to(u.pc))


# DEPRECATED: not currently used
def N_coll_r_perM(r0,ts, alpha, cv,rho0,fimf,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """
    Number of collisions for a no black hole power law system INTEGRAND PORTION
    """
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 
    sqrt_term = cv * G / (1+alpha)

    def integral(r):
        prefactor = np.pi *ts 
        M_r = 4 * np.pi * rho0 *r **(3.-alpha) /(3.-alpha)/r0**(-alpha)
        sigma_r = np.sqrt(sqrt_term*M_r/r )
        n_r  = rho0/Mstar * (r/r0)**(-alpha)
        # eccentricity functions
        F1 = f1 * rc**2
        F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
        
        result = prefactor * n_r * sigma_r * (F1+(F2/sigma_r**2))
        return result.cgs

    # if alpha == 1.6:
    #     return integral58(rmax.to("pc"))-integral58(rmin.to("pc"))
    # elif np.isclose(alpha, (4./3.),1e-6) :
    #     return integral43(rmax.to("pc"))-integral43(rmin.to("pc"))
    # else:
    return integral(rmax.to("pc"))
    # return integral(rmax.to(u.pc))-integral(rmin.to(u.pc))


def N_coll_bh_limits(r0,ts, alpha, cv,rho0,fimf,MBH,*, 
                            Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """ 
    Number of collisions with a black hole and a power law rho profile
    """
    #calc collisional radii and eccentricity functions
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 

    #First prefactor
    prefactor = np.pi *ts * fimf * (3-alpha) / (Mstar**2)
    # eccentricity functions
    F1 = f1 * rc**2
    F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
    #term prefactors
    sqrt_term = cv * G / (1+alpha)
    cm = 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    # more prefactors and the integral need the limits
    def integrate_func(r):
        first_pref = np.sqrt(MBH) * r**(5./2.-2*alpha)/(5./2.-2*alpha)
        second_pref = np.sqrt(1./MBH) * r**(7./2.-2*alpha)/(7./2.-2*alpha)

        #put it all together
        result = prefactor * (
            (F1 * np.sqrt(sqrt_term) *cm * crho * first_pref*
                scipy.special.hyp2f1(
                    0.5, 
                    (5./2.-2*alpha)/(3.-alpha),
                    (11./2.-3*alpha)/(3.-alpha),
                    (-cm * r**(3.-alpha)/(MBH)).cgs.value
                )
                 ) 
            +
            (F2 * (sqrt_term)**(-1./2.)*cm * crho * second_pref* 
                scipy.special.hyp2f1(
                    0.5, 
                    (7./2.-2*alpha)/(3.-alpha),
                    (13./2.-3*alpha)/(3.-alpha),
                    (-cm * r**(3.-alpha)/(MBH)).value
                )
             )
        )
        return result.cgs.value
    def integrate_54(r):
        """Exact form for alpha = 5/4 for which hypergeometric integral has zero in denominator"""
        first_pref = 8./7.
        second_pref = np.sqrt(1./MBH) * r**(7./2.-2*alpha)/(7./2.-2*alpha)

        #put it all together
        result = prefactor * (
            (F1 * np.sqrt(sqrt_term) *cm * crho * first_pref*
                (
                    np.sqrt(cm * r**(7./4.)+MBH)-
                    np.sqrt(MBH)*np.arcsinh((np.sqrt(MBH)/np.sqrt(cm)/r**(7./8.)).cgs.value)
                )
                 ) 
            +
            (F2 * (sqrt_term)**(-1./2.)*cm * crho * second_pref* 
                scipy.special.hyp2f1(
                    0.5, 
                    (7./2.-2*alpha)/(3.-alpha),
                    (13./2.-3*alpha)/(3.-alpha),
                    (-cm * r**(3.-alpha)/(MBH)).value
                )
             )
        )
        return result.cgs.value
    def integrate_74(r):
        """Exact form for alpha = 7/4 for which hypergeometric integral has zero in denominator"""

        first_pref = np.sqrt(MBH) * r**(5./2.-2*alpha)/(5./2.-2*alpha)
        second_pref =-8./(5.*np.sqrt(MBH))

        #put it all together
        result = prefactor * (
            (F1 * np.sqrt(sqrt_term) *cm * crho * first_pref*
                scipy.special.hyp2f1(
                    0.5, 
                    (5./2.-2*alpha)/(3.-alpha),
                    (11./2.-3*alpha)/(3.-alpha),
                    (-cm * r**(3.-alpha)/(MBH)).cgs.value
                )
                 ) 
            +
            (F2 * (sqrt_term)**(-1./2.)*cm * crho * second_pref* 
                (
                    np.arcsinh((np.sqrt(MBH)/np.sqrt(cm)/(r**(5/8))).cgs.value)
                )
             )
        )
        return result.cgs.value
    if alpha == 1.25:
        return integrate_54(rmax.to("pc"))-integrate_54(rmin.to("pc"))
    elif alpha == 1.75:
        return integrate_74(rmax.to("pc"))-integrate_74(rmin.to("pc"))
    else:
        return integrate_func(rmax.to("pc"))-integrate_func(rmin.to("pc")) 



def Mdot_pl_no_bh_limits(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """
    Number of collisions for a no black hole power law system
    """ 
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 
    Massratio = Mcollisions/Mstar
    # eccentricity functions
    F1 = f1 * rc**2
    F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
    cm = 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    #outer prefactor
    pref_num1 = np.pi * G**2 * ts * fimf * (Mstar+Mcollisions) *Massratio * coulomb
    pref_denom1 = 0.34 * Mstar
    pref_num2 = np.pi * ts * fimf * reduced_mass* Massratio *G * coulomb * (3-alpha) * (Mstar+Mcollisions)
    pref_denom2 = 0.34 *( (Mstar**2/rstar)+(Mcollisions**2/rcollisions))*Mstar
    def integrate_func1(r):
        result  = pref_num1 / pref_denom1 * F1 * (3-alpha)*(1+alpha)**2/cv/G *crho**2 /(1-2*alpha) * r**(1-2*alpha)
        return result.cgs
    def integrate_func2(r):
        result = pref_num1 / pref_denom1 * F2 * (3-alpha)*(1+alpha)/cv**2/G**2 * crho**2/cm/(-alpha-1)* r**(-1-alpha)
        return result.cgs
    def integrate_func3(r):
        result = pref_num2 / pref_denom2 * F1 * (3-alpha)*crho**2 * cm /(3-3*alpha)*r**(3-3*alpha)
        return result.cgs
    def integrate_func4(r):
        result = pref_num2 / pref_denom2 * F2 *(3-alpha)*(1+alpha)/cv/G *crho**2 /(1-2*alpha) * r**(1-2*alpha)
        return result.cgs
    def integrate_func(r):
        result = integrate_func1(r)+integrate_func2(r)-integrate_func3(r)-integrate_func4(r)
        return result.cgs
    return integrate_func(rmax.to("pc"))-integrate_func(rmin.to("pc")) 


# DEPRECATED: not currently used
def dMdotdr_pl_no_bh_limits(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """
    Number of collisions for a no black hole power law system
    """ 
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 
    Massratio = Mcollisions/Mstar
    # eccentricity functions
    F1 = f1 * rc**2
    F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
    cm = 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    #outer prefactor
    pref_num1 = np.pi * G**2 * ts * fimf * (Mstar+Mcollisions) *Massratio * coulomb
    pref_denom1 = 0.34 * Mstar
    pref_num2 = np.pi * ts * fimf * reduced_mass* Massratio *G * coulomb * (3-alpha) * (Mstar+Mcollisions)
    pref_denom2 = 0.34 *( (Mstar**2/rstar)+(Mcollisions**2/rcollisions))*Mstar
    def integrate_func1(r):
        result  = pref_num1 / pref_denom1 * F1 * (3-alpha)*(1+alpha)/cv/G *crho**2 * r**(-2*alpha)
        return result.cgs
    def integrate_func2(r):
        result = pref_num1 / pref_denom1 * F2 * (3-alpha)*(1+alpha)/cv**2/G**2 * crho**2/cm* r**(-2-alpha)
        return result.cgs
    def integrate_func3(r):
        result = pref_num2 / pref_denom2 * F1 * (3-alpha)*crho**2 * cm *r**(2-3*alpha)
        return result.cgs
    def integrate_func4(r):
        result = pref_num2 / pref_denom2 * F2 *(3-alpha)*(1+alpha)/cv/G *crho**2  * r**(-2*alpha)
        return result.cgs
    def integrate_func(r):
        result = integrate_func1(r)+integrate_func2(r)-integrate_func3(r)-integrate_func4(r)
        return result.cgs
    return integrate_func(rmax.to("pc"))-integrate_func(rmin.to("pc")) 




# DEPRECATED: not currently used (first-order approximation, superseded by Ncoll_pl_no_bh_limits)
def Ncoll_pl_no_bh_limits_firstorder(r0,ts, alpha, cv,rho0,fimf,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """
    Number of collisions for a no black hole power law system to first order + sticky spheres
    """
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    cross_section = np.pi * rc**2.
    cm = 4. * np.pi * rho0/(3.-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    N0 = 4. * np.pi * cross_section *ts/ (Mstar**2.) *np.sqrt(G* cv/(1.+alpha)) *crho**2.*np.sqrt(cm)*1./(4.-(5.*alpha/2.))
    def integral(r):
        return N0 * r**(4.-(5.*alpha/2.))
    result =  integral(rmax).cgs-integral(rmin).cgs
    return result.cgs

# DEPRECATED: not currently used (first-order approximation, superseded by Mdot_pl_no_bh_limits)
def Mdot_pl_no_bh_limits_firstorder(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """ Mass rate first order and sticky spheres""" 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    cross_section = np.pi * rc**2.
    cm = 4. * np.pi * rho0/(3.-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    N0 = 4. * np.pi * cross_section *ts/ (Mstar**2.) *np.sqrt(G* cv/(1.+alpha)) *crho**2*np.sqrt(cm)*1./(4.-(5.*alpha/2.))
    tdf_avg = average_tdf(rmin,rmax,coulomb,cv, Mstar,Mcollisions,cm,crho,alpha)
    # print(tdf_avg.to('yr'))
    def integral(r):
        result = N0 * 2 *Mstar/tdf_avg * r**(4-(5*alpha/2))
        return result.cgs 
    result = integral(rmax)-integral(rmin)
    return result.cgs

# DEPRECATED: not currently used (first-order approximation)
def Mdot_deplete_pl_no_bh_limits_firstorder(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """ Mass rate first order and sticky spheres""" 
    Mdf = Mdot_pl_no_bh_limits_firstorder(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,Mstar = Mstar,Mcollisions=Mcollisions, e =e,rmax = rmax, rmin = rmin)
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    cross_section = np.pi * rc**2.
    cm = 4. * np.pi * rho0/(3.-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    tdf_avg = average_tdf(rmin,rmax,coulomb,cv, Mstar,Mcollisions,cm,crho,alpha)
    result = Mdf * Mcollisions/(2*Mstar) * tdf_avg/ts
    return result.cgs

# DEPRECATED: not currently used
def average_tdf(rmin,rmax,coulomb, cv,Mstar,Mcollisions,cm,crho, alpha):
    q = Mstar/Mcollisions
    box = 1/(rmax-rmin) * 0.34*q / G**2 /Mstar/coulomb
    triangle = box * (cv * G*(1+alpha))**(3./2.)
    def integral(r):
        final = triangle *(cm**1.5/crho)*(1./(4.-(alpha/2.)))*(r**(4.-(alpha/2.)))
        return final.to('s') 
    return integral(rmax)-integral(rmin)


def Mdot_deplete_noBH_limits(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,*, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """
    The full depletion rate using the collisions timescale (equation 108 in my document)
    """ 
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 
    Massratio = Mcollisions/Mstar
    # eccentricity functions
    F1 = f1 * rc**2
    F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
    cm = 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    D=(G*Mstar**2/rstar)+(G* Mcollisions**2/rcollisions)
    #outer prefactor
    pref = 4. * np.pi**2 * (Mstar+Mcollisions) *fimf *crho**2/(Mstar**2)
    
    def integrate_func1(r):
        pref_1 = (F1 -(F2 * reduced_mass/D))*(cv * G /(1.+alpha))**0.5 *cm**0.5 * 1./(4.-(5.*alpha/2.))
        result = pref_1 * r**(4.-(5.*alpha/2.))
        return result.cgs
    def integrate_func2(r):
        pref_2 = F1 *reduced_mass/D *(cv * G /(1.+alpha))**1.5*cm**1.5 * 1./(6.-(7.*alpha/2.))
        result = pref_2 * r**(6.-(7.*alpha/2.))
        return result.cgs
    def integrate_func3(r):
        pref_3 = F2*(cv * G /(1.+alpha))**(-0.5)*cm**(-0.5) *1./(2.-(3.*alpha/2.))
        result = pref_3 * r**(2.-(3.*alpha/2.))
        return result.cgs
    def integrate_func(r):
        result = integrate_func1(r)-integrate_func2(r)+integrate_func3(r)
        return pref * result
    result = integrate_func(rmax)-integrate_func(rmin)
    return result.cgs


# def Mdot_binaries_pl_limits(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,Mtot, Rtot,*,
#                                 mubb= 0.17507,mubs =0.153619, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
#     """ Mass outflow/inflow rate due to binary heating. My equation 114"""
#     #Calculations 
#     rstar = stellar_radius_approximation(Mstar)
#     rcollisions = stellar_radius_approximation(Mcollisions)
#     rc = (rstar+rcollisions).to(u.Rsun)
#     f1,f2 = get_ecc_functions(e,alpha) 
#     Massratio = Mcollisions/Mstar
#     cm = 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
#     crho = rho0/(r0**(-alpha))
#     pref = 4 * np.pi * G * Mstar**2 * Rtot / ( Mtot**2 ) * (mubs+mubb) *((1+alpha)/cv/G)**0.5 * cm**(-0.5) * crho **2 / (2.-(3.*alpha/2.))
#     def integrate_func(r):
#         result  = pref * r**(2.-(3.*alpha/2.))
#         return result.cgs
#     result  = integrate_func(rmax)-integrate_func(rmin)
#     return result.cgs

def Mdot_binaries_pl_limits(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,Mtot, Rtot,*,
                                mubb= 0.17507,mubs =0.153619,sigma = 20*u.km/u.s, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
    """ delta phi versionversion Mass outflow/inflow rate due to binary heating. My equation 114"""

    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 
    Massratio = Mcollisions/Mstar
    cm = 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
    crho = rho0/(r0**(-alpha))
    pref = 4 * np.pi  * Mstar *G**2/((3*sigma)**2) * (mubs+mubb) *((1+alpha)/cv/G)**0.5 * cm**(-0.5) * crho **2 / (2.-(3.*alpha/2.))
    def integrate_func(r):
        result  = pref * r**(2.-(3.*alpha/2.))
        return result.cgs
    result  = integrate_func(rmax)-integrate_func(rmin)
    return result.cgs


def Mdot_gas_no_bh_limits(r0, ts, alpha, cv, rho0, fimf, Mstar, Mcollisions, *,
                           e=0, rmax=1e10*u.pc, rmin=1000*u.Rsun):
    """
    Integrated gas production rate from stellar collisions in a power-law profile (no BH).

    Computes int_{rmin}^{rmax} M_g,coll(r) / t_coll(r) * dN/dr dr, where
        M_g,coll(r) = (M_* + M_c) * mu * sigma(r)^2 / E_bind
        1/t_coll(r) = pi * n(r) * sigma(r) * (f1*rc^2 + 2*f2*rc*G*(M_*+M_c)/sigma^2)
        dN/dr       = (rho_*(r)/M_*) * 4*pi*r^2

    With sigma(r) = c_sig * r^((2-alpha)/2), the integrand has two analytic power-law terms:

        Term A (geometric, ~f1): prefactor_A * r^(5 - 7*alpha/2)
        Term B (grav. focusing, ~f2): prefactor_B * r^(3 - 5*alpha/2)

    Note: alpha = 12/7 (Term A) or alpha = 8/5 (Term B) produces a log singularity
    in the antiderivative; these values are unlikely for typical power-law profiles
    but are not currently handled as special cases.

    Parameters
    ----------
    r0 : Quantity [length]
        Characteristic (half-mass) radius of the profile.
    ts : Quantity [time]
        Disruption timescale (used only to match the signature pattern; the rate
        itself is independent of ts — multiply the result by ts to get total mass).
    alpha : float
        Power-law slope of the density profile.
    cv : float
        Virial velocity dispersion normalization constant.
    rho0 : Quantity [mass/volume]
        Central density at r0.
    fimf : float
        IMF mass fraction in the relevant mass bin.
    Mstar : Quantity [mass]
        Stellar mass M_*.
    Mcollisions : Quantity [mass]
        Collider mass M_c.
    e : float, optional
        Orbital eccentricity (default 0).
    rmax : Quantity [length], optional
        Upper integration limit.
    rmin : Quantity [length], optional
        Lower integration limit.

    Returns
    -------
    Quantity [mass/time]
        Total gas production rate in CGS units (g/s). Convert with .to(u.Msun/u.yr).
    """
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar + rcollisions).to(u.Rsun)
    f1, f2 = get_ecc_functions(e, alpha)

    mu = Mstar * Mcollisions / (Mstar + Mcollisions)
    E_bind = G * (Mcollisions**2 / rcollisions + Mstar**2 / rstar)

    # Profile density and enclosed-mass coefficients (matching existing convention)
    crho = rho0 / (r0**(-alpha))
    cm   = 4 * np.pi * crho / (3 - alpha)

    # Velocity dispersion coefficient: sigma(r) = c_sig * r^((2-alpha)/2)
    c_sig = np.sqrt(cv * G * cm / (1 + alpha))

    # Prefactor for Term A (geometric collision cross-section, ~f1):
    #   P_A * r^(6 - 7*alpha/2) / (6 - 7*alpha/2)
    pref_A = (4 * np.pi**2 * (Mstar + Mcollisions) * mu * c_sig**3
              * crho**2 * fimf * f1 * rc**2 / (E_bind * Mstar**2))

    # Prefactor for Term B (gravitational focusing, ~f2):
    #   P_B * r^(4 - 5*alpha/2) / (4 - 5*alpha/2)
    pref_B = (8 * np.pi**2 * (Mstar + Mcollisions)**2 * mu * c_sig
              * crho**2 * fimf * G * f2 * rc / (E_bind * Mstar**2))

    def integrate_A(r):
        return (pref_A / (6 - 7 * alpha / 2) * r**(6 - 7 * alpha / 2)).cgs

    def integrate_B(r):
        return (pref_B / (4 - 5 * alpha / 2) * r**(4 - 5 * alpha / 2)).cgs

    def integrate_func(r):
        return integrate_A(r) + integrate_B(r)

    return integrate_func(rmax.to("pc")) - integrate_func(rmin.to("pc"))


# def Mdot_binaries_pl_limits(r0,ts, alpha, cv,rho0,fimf,reduced_mass, coulomb,Mtot, Rtot,*,
#                                 mubb= 0.17507,mubs =0.153619, Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 0,rmax = 1e10*u.pc, rmin = 1000*u.Rsun):
#     """c ^2 version Mass outflow/inflow rate due to binary heating. My equation 114"""

#     #Calculations 
#     rstar = stellar_radius_approximation(Mstar)
#     rcollisions = stellar_radius_approximation(Mcollisions)
#     rc = (rstar+rcollisions).to(u.Rsun)
#     f1,f2 = get_ecc_functions(e,alpha) 
#     Massratio = Mcollisions/Mstar
#     cm = 4 * np.pi * rho0/(3-alpha)/(r0**(-alpha))
#     crho = rho0/(r0**(-alpha))
#     pref = 4 * np.pi  * Mstar *G**2/(c_sl**2) * (mubs+mubb) *((1+alpha)/cv/G)**0.5 * cm**(-0.5) * crho **2 / (2.-(3.*alpha/2.))
#     def integrate_func(r):
#         result  = pref * r**(2.-(3.*alpha/2.))
#         return result.cgs
#     result  = integrate_func(rmax)-integrate_func(rmin)
#     return result.cgs


import numpy as np
import astropy.units as u
from astropy.constants import G
from scipy.special import hyp2f1

def Mdot_df_withbh(r0, td, alpha, cv, rho0, fimf, MBH, lnLambda, *,
                   Mstar=1.0*u.Msun,
                   Mcollisions=1.0*u.Msun,
                   e=0.0,
                   rmax=1e10*u.pc,
                   rmin=1000*u.Rsun,
                   q=None):
    """
    Mass dynamical-friction rate with a central BH, using the closed-form
    hypergeometric expressions.

    Parameters
    ----------
    r0 : Quantity [length]
        Reference radius for the density normalization (rho = rho0 (r/r0)^(-alpha)).
    td : Quantity [time]
        Disruption time.
    alpha : float
        Density slope (rho ∝ r^{-alpha}).
    cv : float
        Dimensionless coefficient in sigma^2 = (cv G/(1+alpha)) (M(r)+MBH)/r.
    rho0 : Quantity [mass/length^3]
        Density at r0.
    fimf : float
        IMF fraction f^{IMF}_{M_*}.
    MBH : Quantity [mass]
        Central black hole mass.
    lnLambda : float
        Coulomb logarithm ln Λ.
    Mstar : Quantity [mass]
        Stellar mass M_*.
    Mcollisions : Quantity [mass]
        Mass of the inspiraling/colliding object M_i.
    e : float
        Eccentricity parameter passed to get_ecc_functions.
    rmax : Quantity [length]
        Upper limit r_df (your r_df).
    rmin : Quantity [length]
        Lower limit r_min.
    q : float, optional
        Dimensionless factor multiplying t_relax to get t_df. If None,
        defaults to Mcollisions/Mstar.

    Returns
    -------
    mdot : Quantity [mass/time]
        Evaluated \dot{M} from rmin to rmax.
    """

    # ---------- sanity / units ----------
    r0 = r0.to(u.cm)
    rmin = rmin.to(u.cm)
    rdf = rmax.to(u.cm)
    td = td.to(u.s)
    rho0 = rho0.to(u.g/u.cm**3)
    Mstar = Mstar.to(u.g)
    Mcollisions = Mcollisions.to(u.g)
    MBH = MBH.to(u.g)

    if q is None:
        q = (Mcollisions / Mstar).decompose().value
    else:
        q = float(q)

    # ---------- collision radius + eccentricity functions ----------
    # you already have these helpers in your codebase
    rstar = stellar_radius_approximation(Mstar.to(u.Msun))
    rcoll = stellar_radius_approximation(Mcollisions.to(u.Msun))
    rc = (rstar + rcoll).to(u.cm)

    f1, f2 = get_ecc_functions(e, alpha)

    # eccentricity functions as in your earlier code
    F1 = (f1 * rc**2).to(u.cm**2)
    F2 = (2.0 * G * f2 * rc * (Mstar + Mcollisions)).to(u.cm**4/u.s**2)  # (G M L) has units L^4/T^2

    # ---------- constants c_rho, c_M ----------
    # rho(r) = c_rho r^{-alpha} with c_rho = rho0 * r0^alpha
    c_rho = (rho0 * r0**alpha).to(u.g * u.cm**(alpha - 3))

    # M(r) = c_M r^{3-alpha} with c_M = 4π c_rho/(3-alpha)
    c_M = (4.0 * np.pi * c_rho / (3.0 - alpha)).to(u.g * u.cm**(alpha - 3))

    # convenient sigma^2 prefactor: sigma^2 = (cv G/(1+alpha)) (M(r)+MBH)/r
    # note: keep as Quantity
    sigma2_pref = (cv * G / (1.0 + alpha)).to(u.cm**3/(u.g*u.s**2))

    # reduced mass mu (has units of mass, needed for your energy ratio term)
    mu = (Mstar * Mcollisions / (Mstar + Mcollisions)).to(u.g)

    # A ≡ G Mi^2/Ri + G M_*^2/R_* (you used this combo repeatedly)
    # Here I assume Ri ~ R_coll object radius and R_* ~ stellar radius from your approximation
    Ri = rcoll.to(u.cm)
    Rstar = rstar.to(u.cm)
    A = (G * Mcollisions**2 / Ri + G * Mstar**2 / Rstar).to(u.cm**2 * u.g / u.s**2)  # energy units

    # ---------- hypergeometric building blocks ----------
    beta = 3.0 - alpha

    def z_of_r(r):
        # z = - c_M r^{3-alpha} / MBH (dimensionless)
        return (-(c_M * r**beta / MBH).decompose().value)

    # I1: ∫ r^{3-3α} (c_M r^{3-α} + MBH)^{-1} dr, as used in your mdot expression
    # antiderivative: r^{4-3α}/(MBH(4-3α)) * 2F1(1, (4-3α)/β; (7-4α)/β; z)
    def antideriv_I1(r):
        denom = (4.0 - 3.0*alpha)
        a = 1.0
        b = (4.0 - 3.0*alpha)/beta
        c = (7.0 - 4.0*alpha)/beta
        zz = z_of_r(r)
        pref = (r**(4.0 - 3.0*alpha) / (MBH * denom)).to(u.cm**(4.0 - 3.0*alpha) / u.g)
        return pref * hyp2f1(a, b, c, zz)

    # I2: ∫ r^{4-3α} (c_M r^{3-α} + MBH)^{-2} dr
    # antiderivative: r^{5-3α}/(MBH^2(5-3α)) * 2F1(2, (5-3α)/β; (8-4α)/β; z)
    def antideriv_I2(r):
        denom = (5.0 - 3.0*alpha)
        a = 2.0
        b = (5.0 - 3.0*alpha)/beta
        c = (8.0 - 4.0*alpha)/beta
        zz = z_of_r(r)
        pref = (r**(5.0 - 3.0*alpha) / (MBH**2 * denom)).to(u.cm**(5.0 - 3.0*alpha) / u.g**2)
        return pref * hyp2f1(a, b, c, zz)

    # I3: ∫ r^{2-3α} dr = r^{3-3α}/(3-3α)
    def antideriv_I3(r):
        denom = (3.0 - 3.0*alpha)
        return (r**(3.0 - 3.0*alpha) / denom).to(u.cm**(3.0 - 3.0*alpha))

    # Evaluate definite integrals
    I1 = antideriv_I1(rdf) - antideriv_I1(rmin)
    I2 = antideriv_I2(rdf) - antideriv_I2(rmin)
    I3 = antideriv_I3(rdf) - antideriv_I3(rmin)

    # ---------- assemble prefactors exactly as in your latex ----------
    common1 = (np.pi * G**2 * td * fimf * (Mstar + Mcollisions) * q * lnLambda * (3.0 - alpha) / (0.34 * Mstar)).to(
        u.cm**6 / (u.g**2 * u.s**3) * u.s * u.g / u.g
    )
    # We’ll trust astropy to simplify; final mdot will be coerced to mass/time at end.

    # (cv G/(1+alpha))^{-1} and ^{-2}
    sigfac_m1 = (sigma2_pref)**(-1)  # units g s^2 / cm^3
    sigfac_m2 = (sigma2_pref)**(-2)

    # term 1 (F1 * ... * I1)
    term1 = common1 * F1 * sigfac_m1 * (c_rho**2) * c_M * I1

    # term 2 (F2 * ... * I2)
    term2 = common1 * F2 * sigfac_m2 * (c_rho**2) * c_M * I2

    # mu-terms prefactor
    common_mu = (np.pi * G * td * fimf * mu * (Mstar + Mcollisions) * q * lnLambda * (3.0 - alpha) /
                 (0.34 * A * Mstar)).to(
        1/u.s
    )  # this coercion may be too aggressive; we’ll handle at end.

    # term 3: - common_mu * F1 * c_rho^2 c_M * I3
    term3 = -common_mu * F1 * (c_rho**2) * c_M * I3

    # term 4: - common_mu * F2 * (cvG/(1+alpha))^{-1} * c_rho^2 c_M * I1
    term4 = -common_mu * F2 * sigfac_m1 * (c_rho**2) * c_M * I1

    mdot = (term1 + term2 + term3 + term4)

    # Try to coerce to a clean mass/time unit for output
    return mdot.to(u.Msun/u.yr)



# DEPRECATED: not currently used
def Mdot_dep_withbh(r0, td, alpha, cv, rho0, fimf, MBH, lnLambda, *, 
                    Mstar=1.0*u.Msun,
                    Mcollisions=1.0*u.Msun,
                    e=0.0,
                    rmax=1e10*u.pc,
                    rmin=1000*u.Rsun):
    """
    Depletion/encounter-driven mass rate with a central BH, using the closed-form
    hypergeometric expressions you derived:

        dot{M}_dep = (4π^2 (M_*+M_i) f_IMF / M_*^2) * [ ... ] |_{rmin}^{rdf}

    Notes / Assumptions
    -------------------
    - rho(r) = rho0 * (r/r0)^(-alpha) = c_rho * r^(-alpha), with c_rho = rho0 * r0^alpha
    - M(r) = c_M r^(3-alpha), with c_M = 4π c_rho/(3-alpha)
    - sigma^2(r) = (cv G/((1+alpha) r)) * (M(r) + MBH)
    - F1(e) and F2(e) are built from get_ecc_functions(e, alpha) exactly as in your earlier code:
        F1 = f1 * r_c^2
        F2 = 2 G f2 r_c (M_* + M_i)
      where r_c is the sum of radii of the interacting bodies.

    Parameters
    ----------
    r0 : Quantity[length]
    td : Quantity[time]
    alpha : float
    cv : float
    rho0 : Quantity[mass/length^3]
    fimf : float
    MBH : Quantity[mass]
    lnLambda : float, optional
        Not used in this dot{M}_dep expression (kept only for signature similarity).
    Mstar : Quantity[mass]
    Mcollisions : Quantity[mass]  (this is your M_i)
    e : float
    rmax : Quantity[length]   (this is your r_df / r_df-like upper limit; called rdf in latex)
    rmin : Quantity[length]

    Returns
    -------
    mdot_dep : Quantity[mass/time]
    """

    # --- normalize units ---
    r0 = r0.to(u.cm)
    rmin = rmin.to(u.cm)
    rdf = rmax.to(u.cm)
    td = td.to(u.s)
    rho0 = rho0.to(u.g/u.cm**3)
    Mstar = Mstar.to(u.g)
    Mi = Mcollisions.to(u.g)
    MBH = MBH.to(u.g)

    # --- radii + eccentricity functions ---
    rstar = stellar_radius_approximation(Mstar.to(u.Msun)).to(u.cm)
    ri = stellar_radius_approximation(Mcollisions.to(u.Msun)).to(u.cm)
    rc = (rstar + ri).to(u.cm)

    f1, f2 = get_ecc_functions(e, alpha)

    F1 = (f1 * rc**2).to(u.cm**2)
    F2 = (2.0 * G * f2 * rc * (Mstar + Mi)).to(u.cm**4/u.s**2)

    # --- c_rho, c_M ---
    c_rho = (rho0 * r0**alpha).to(u.g * u.cm**(alpha - 3))
    c_M = (4.0 * np.pi * c_rho / (3.0 - alpha)).to(u.g * u.cm**(alpha - 3))

    beta = 3.0 - alpha

    # sigma^2 prefactor (Quantity):
    sigma2_pref = (cv * G / (1.0 + alpha)).to(u.cm**3/(u.g*u.s**2))

    # reduced mass mu:
    mu = (Mstar * Mi / (Mstar + Mi)).to(u.g)

    # A ≡ G Mi^2/Ri + G M_*^2/R_* (energy-like)
    A = (G * Mi**2 / ri + G * Mstar**2 / rstar).to(u.cm**2 * u.g / u.s**2)

    # dimensionless argument z(r) = -c_M r^(3-alpha)/MBH
    def z_of_r(r):
        return (-(c_M * r**beta / MBH).decompose().value)

    # --- antiderivatives for the three needed integrals ---
    # Generic identity used:
    # ∫ r^(m-1) (1 + k r^beta)^p dr = r^m/m * 2F1(-p, m/beta; 1+m/beta; -k r^beta)
    # Here (c_M r^beta + MBH)^p = MBH^p (1 + (c_M/MBH) r^beta)^p

    def antideriv_J1(r):
        # ∫ r^(3/2-2α) (c_M r^β + MBH)^(+1/2) dr
        m = 5.0/2.0 - 2.0*alpha
        p = +0.5
        a = -p                       # -1/2
        b = m / beta
        c = 1.0 + b                  # (11/2 - 3α)/(3-α)
        zz = z_of_r(r)
        pref = (MBH**p * r**m / m).to(u.g**0.5 * u.cm**m)
        return pref * hyp2f1(a, b, c, zz)

    def antideriv_J2(r):
        # ∫ r^(1/2-2α) (c_M r^β + MBH)^(+3/2) dr
        m = 3.0/2.0 - 2.0*alpha
        p = +1.5
        a = -p                       # -3/2
        b = m / beta
        c = 1.0 + b                  # (9/2 - 3α)/(3-α)
        zz = z_of_r(r)
        pref = (MBH**p * r**m / m).to(u.g**1.5 * u.cm**m)
        return pref * hyp2f1(a, b, c, zz)

    def antideriv_J3(r):
        # ∫ r^(5/2-2α) (c_M r^β + MBH)^(-1/2) dr
        m = 7.0/2.0 - 2.0*alpha
        p = -0.5
        a = -p                       # +1/2
        b = m / beta
        c = 1.0 + b                  # (13/2 - 3α)/(3-α)
        zz = z_of_r(r)
        pref = (MBH**p * r**m / m).to(u.g**(-0.5) * u.cm**m)
        return pref * hyp2f1(a, b, c, zz)

    # definite integrals
    J1 = antideriv_J1(rdf) - antideriv_J1(rmin)
    J2 = antideriv_J2(rdf) - antideriv_J2(rmin)
    J3 = antideriv_J3(rdf) - antideriv_J3(rmin)

    # --- assemble the bracket exactly like your latex ---
    # factors of (cv G/(1+alpha))^(±1/2, ±3/2)
    sig_half_p1 = (sigma2_pref)**(0.5)
    sig_half_m1 = (sigma2_pref)**(-0.5)
    sig_3half_p = (sigma2_pref)**(1.5)

    # bracket pieces
    coeff_mix = (F1 - F2 * (mu / A))  # has units cm^2 (since F2*(mu/A) -> cm^2)
    termA = coeff_mix * (c_rho**2) * sig_half_p1 * J1
    termB = -F1 * (mu / A) * (c_rho**2) * sig_3half_p * J2
    termC = +F2 * (c_rho**2) * sig_half_m1 * J3

    bracket = termA + termB + termC

    prefactor = (4.0 * np.pi**2 * (Mstar + Mi) * fimf / (Mstar**2)).to(1/u.g)

    mdot_dep = (prefactor * bracket ).to(u.g/u.s)  # /td gives mass/time

    return mdot_dep.to(u.Msun/u.yr)