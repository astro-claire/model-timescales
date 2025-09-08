#src/timescales/utils/filtering.py
import inspect
from astropy import units as u
from astropy.units import Quantity

# def filter_kwargs_for(func, kwargs):
#     sig = inspect.signature(func)
#     valid_keys = set(sig.parameters.keys())
#     return {k: v for k, v in kwargs.items() if k in valid_keys}
def filter_kwargs_for(func, kwargs):
    """
    Split kwargs into:
      - provided: keys in kwargs that match func's parameters
      - missing: keyword-only params of func that have defaults, but weren't in kwargs
    """
    sig = inspect.signature(func)

    # Collect provided
    provided = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Collect missing (only keyword args with defaults, not supplied)
    missing = {}
    for name, param in sig.parameters.items():
        if (param.default is not inspect._empty    # has a default
            and name not in kwargs                # user didn't provide
            and param.kind in (inspect.Parameter.KEYWORD_ONLY,
                               inspect.Parameter.POSITIONAL_OR_KEYWORD)):
            missing[name] = param.default

    return provided, missing