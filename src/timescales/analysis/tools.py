#src/timescales/analysis/tools.py 
""" 
Tools for analysis scripts
"""
from ..physics import coulomb

def select_coulomb_calculator(ensemble):
    """ 
    Pick which coulomb logarithm calculator to use based on the presence of a BH or not
    """
    if "BH" in ensemble.densityModel:
        return coulomb.coulomb_log_BH
    else:
        return coulomb.coulomb_log