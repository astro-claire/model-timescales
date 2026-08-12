import numpy as np
import astropy.units as u
import astropy.constants as c



def stellar_bondi_accretion_rate(rhogas, Mstar, cs, v):
    return (4 * np.pi *  rhogas * c.G**2 * Mstar**2 / (cs**2+ v**2)**(3./2)).to(u.Msun/u.yr)

def stellar_2phase_bondi_accretion_rate(rho1, rho2, Mstar, cs1,cs2, v,f):
    M_dot_BoHo1 = 4. * np.pi *rho1* c.G**2 * Mstar **2 / (cs1**2 + v**2)**(3/2)
    M_dot_BoHo2 = 4. * np.pi *rho2* c.G**2 * Mstar **2 / (cs2**2 + v**2)**(3/2)
    M_dot_BoHo = (M_dot_BoHo1) + ((1.-f) * M_dot_BoHo2)
    return M_dot_BoHo

def smbh_bondi_accretion_rate(rhogas, MBH, cs):
    return (4 * np.pi * rhogas * c.G**2 * MBH**2 / (cs**3)).to(u.Msun/u.yr)


def bondi_radius(M,cs):
    return (c.G * M/ (cs**2)).to('pc')

