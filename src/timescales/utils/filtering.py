#src/timescales/utils/filtering.py
import inspect
from astropy import units as u
from astropy.units import Quantity

def filter_kwargs_for(func, kwargs):
    sig = inspect.signature(func)
    valid_keys = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_keys}
