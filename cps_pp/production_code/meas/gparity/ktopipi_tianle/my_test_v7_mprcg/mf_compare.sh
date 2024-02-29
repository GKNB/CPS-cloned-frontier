#!/bin/bash

mpirun -n 1 /home/ckelly/qcdserver12/software/bld/cps_mpi_grid/tests/mesonfield_compare/NOARCH.x 4 3 0 $1 $2 -qmp-geom 1 1 1 1


#~/CPS/testing/a2a_test/full_test/gp3/ck_full_mobius_noxyzstep/traj_0_pion_mf_mom_2_2_2_hyd1s_rad2.dat ~/CPS/testing/a2a_test/full_test/gp3/ck_full_mobius_noxyzstep/traj_0_pion_mf_mom_2_2_2_hyd1s_rad2.dat -qmp-geom 1 1 1 1
