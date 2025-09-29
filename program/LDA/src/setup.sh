gfortran -c ./utils/latlonconv.f90
f2py --fcompiler=gnu95 -m coord_conv -c --f90flags='-O3' ./utils/latlonconv.f90