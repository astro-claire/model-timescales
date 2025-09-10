#src/timescales/analysis/tools.py 
""" 
Tools for analysis scripts
"""
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from typing import Optional

from ..utils.units import as_quantity
from ..physics import coulomb

def select_coulomb_calculator(ensemble):
    """ 
    Pick which coulomb logarithm calculator to use based on the presence of a BH or not
    """
    if "BH" in ensemble.densityModel:
        return coulomb.coulomb_log_BH
    else:
        return coulomb.coulomb_log


def condition_test(ts1, 
                            operation: str,*, 
                            ts2: Optional[Quantity]= None, 
                            value: Optional[Quantity]=None, 
                            verbose: Optional[bool] = True
                            ):
    """ 
    Check if a condition is met

    Parameters
    ----------
    ts1 (astropy.Quantity): timescale for comparison
    operation (str): what comparative operation to do
        Options: 'eq' (equal to), 'lt' (less than), 'gt' (greater than), or 'true'
    
    Returns
    -------
    Bool - True if condition is met
    """
    if isinstance(ts1[0], Quantity):
        units = ts1[0].unit
        ts1 = u.Quantity(ts1,units)
    else: 
        ts1 = np.array(ts1)

    if (ts2 != None and value != None):
        raise TypeError("Two comparison values provided.")

    if ts2 != None:
        if isinstance(ts2[0], Quantity):
            units = ts2[0].unit
            ts2 = u.Quantity(ts2,units)
        else:
            ts2 = np.array(ts2)
        comparison = ts2
    elif value != None: 
        comparison = value
    elif operation != 'true':
        raise TypeError("Either timescale or value for comparison must be provided if equal, greater, or less than is selected.")


    result = False
    if operation == 'lt':
        values = np.where(ts1 < comparison, 1, 0)
    elif operation == 'gt':
        values = np.where(ts1 > comparison, 1, 0)
    elif operation == 'eq':
        values = np.where(ts1 == comparison, 1, 0)
    elif operation == 'true':
        values = ts1
    else: 
        raise NotImplementedError("Unknown operation. Options are 'eq' (equal to), 'lt' (less than), 'gt' (greater than), or 'true'.")
    if np.sum(values)>1: 
        result  = True
    
    return result


def get_system(table, system_id, *, as_df=False):
    """
    Extract all rows for a specific system ID from a table.
    """
    if isinstance(table, dict):
        return {
            key: [val for sid, val in zip(table["system_id"], values) if sid == system_id]
            for key, values in table.items()
        }
    elif as_df:
        return table[table["system_id"] == system_id]
    else:
        raise TypeError("Table must be a dict or pandas.DataFrame")

