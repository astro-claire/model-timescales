import numpy as np
import scipy
from astropy.constant import G
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


def Ncoll_pl_no_bh(r,alpha, cv,rho0,r0,fimf,rc,Mstar = 1.0*u.Msun,Mcollisions=1.*u.Msun, e = 1):
    """
    Number of collisions for a no black hole power law system
    """
    #Calculations 
    rstar = stellar_radius_approximation(Mstar)
    rcollisions = stellar_radius_approximation(Mcollisions)
    rc = (rstar+rcollisions).to(u.Rsun)
    f1,f2 = get_ecc_functions(e,alpha) 

    #the solution:
    #-------------
    prefactor = np.pi *ts * fimf / (Mstar**2)
    # eccentricity functions
    F1 = f1 * rc**2
    F2 = 2. * G * f2 * rc * (Mstar + Mcollisions)
    #term prefactors
    sqrt_term = cv * G / (1+alpha)
    rhoterm1 = (4.* np.pi)**(3./2.) * (rho0**(5./2.))*(3.-alpha)**(-3./2.)
    rhoterm2 = (4.* np.pi)**(1./2.) * (rho0**(3./2.))*(3.-alpha)**(-1./2.)
    # The main r0 terms
    r0term1 = r0**(5.-(5.*alpha/2)) /(5.-(5.*alpha/2))
    r0term2 = r0**(3.-(3.*alpha/2)) /(3.-(3.*alpha/2))
    #put it together
    #------
    result = prefactor * ( (F1* np.sqrt(sqrt_term)*rhoterm1*r0term1)+(F2 *(sqrt_term)**(-1./2.)*rhoterm2 *r0term2))

    return result