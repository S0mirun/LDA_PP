import importlib

_core = importlib.import_module(__package__ + '.coord_conv')


if hasattr(_core, 'lat_lon') and hasattr(_core.lat_lon, 'caldistco'):
    caldistco = _core.lat_lon.caldistco
else:
    candidates = ['caldistco','calcdistco','distco','calc_dist_course','calcdist_course']
    found = None
    for name in dir(_core):
        if name.startswith('_'):
            continue
        if name.lower() in candidates:
            found = name
            break
    if not found:
        for name in dir(_core):
            if name.startswith('_'):
                continue
            low = name.lower()
            if 'dist' in low and ('co' in low or 'course' in low):
                found = name
                break
    if not found:
        raise ImportError(f"[coord_conv.lat_lon] distance/course function is not found in {_core.__file__}")
    caldistco = getattr(_core, found)

__all__ = ['caldistco']