import numpy as np
import scipy
from astropy.constants import G
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

    return integral(rmax.to(u.pc))-integral(rmin.to(u.pc))


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
        print(scipy.special.hyp2f1(
                    0.5, 
                    (5./2.-2*alpha)/(3.-alpha),
                    (11./2.-3*alpha)/(3.-alpha),
                    (-cm * r**(3.-alpha)/(MBH)).cgs.value
                ),scipy.special.hyp2f1(
                    0.5, 
                    (7./2.-2*alpha)/(3.-alpha),
                    (13./2.-3*alpha)/(3.-alpha),
                    (-cm * r**(3.-alpha)/(MBH)).value
                ))
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