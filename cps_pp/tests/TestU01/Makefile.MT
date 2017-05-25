
#BUILDDIR = /home/chulwoo/CPS/branches/sitmo/local
#DFLAGS +=  -DUSE_OMP -DUSE_C11  -DUSE_C11_RNG -DUSE_C11_SITMO  -DUSE_GMP -DUSE_QMP -DUSE_QIO
#BIN=cps_SITMO.x
BUILDDIR = /home/chulwoo/CPS/branches/sitmo/mt
DFLAGS +=  -DUSE_OMP -DUSE_C11  -DUSE_C11_RNG -DUSE_C11_MT  -DUSE_GMP -DUSE_QMP -DUSE_QIO
BIN=cps_MT.x
#BUILDDIR = /home/chulwoo/CPS/branches/sitmo/ranlux
#DFLAGS +=  -DUSE_OMP -DUSE_C11  -DUSE_C11_RNG -DUSE_C11_RANLUX  -DUSE_GMP -DUSE_QMP -DUSE_QIO
#BIN=cps_RANLUX.x
include Makefile.common
