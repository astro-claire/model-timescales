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
