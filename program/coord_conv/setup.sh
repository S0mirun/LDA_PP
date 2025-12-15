set -e
PYTHON=${PYTHON:-python}
"$PYTHON" -m numpy.f2py -c -m coord_conv latlonconv.f90