from pathlib import Path
try:
    # Prefer a real module in utils/ if present later
    from ._image_utils_impl import *  # type: ignore
except Exception:
    # Fallback to root image_utils.py
    from importlib import import_module as _im
    _m = _im("image_utils")
    globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith("_")})
