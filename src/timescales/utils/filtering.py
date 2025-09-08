#src/timescales/utils/filtering.py

def filter_kwargs_for(func, kwargs):
    sig = inspect.signature(func)
    valid_keys = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_keys}
