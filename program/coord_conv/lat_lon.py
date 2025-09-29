import importlib, inspect, re

_core = importlib.import_module(__package__ + '.coord_conv')

if hasattr(_core, 'lat_lon'):
    lat_lon_mod = _core.lat_lon
    if hasattr(lat_lon_mod, 'caldistco'):
        caldistco = lat_lon_mod.caldistco
        __all__ = ['caldistco']
    else:
        names = [n for n in dir(lat_lon_mod) if callable(getattr(lat_lon_mod, n))]
        cand = None
        for n in names:
            l = n.lower()
            if 'dist' in l and ('co' in l or 'course' in l):
                cand = n; break
        if cand:
            caldistco = getattr(lat_lon_mod, cand)
            __all__ = ['caldistco']
        else:
            raise ImportError("coord_conv.lat_lon: suitable function not found inside embedded lat_lon")
else:
    names = [n for n in dir(_core) if callable(getattr(_core, n))]
    preferred = ['caldistco','calcdistco','distco','calc_dist_course','calcdist_course']
    cand = None
    for p in preferred:
        for n in names:
            if n.lower() == p:
                cand = n; break
        if cand: break
    if not cand:
        for n in names:
            l = n.lower()
            if 'dist' in l and ('co' in l or 'course' in l):
                cand = n; break
    if not cand:
        raise ImportError(f"coord_conv.lat_lon: could not find distance/course function in {_core.__file__}. "
                          f"Available: {names[:20]}...")
    caldistco = getattr(_core, cand)
    __all__ = ['caldistco']
