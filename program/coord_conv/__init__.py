from coord_conv import *
from . import lat_lon
from pathlib import Path

p = Path("coord_conv/__init__.py")
with p.open("r", encoding="utf-8") as f:
    s = f.read()
if "from . import lat_lon" not in s:
    s = s.rstrip() + "\nfrom . import lat_lon\n"
with p.open("w", encoding="utf-8") as f:
    f.write(s)
print("updated:", p.resolve())
