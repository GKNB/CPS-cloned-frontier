<<<<<<< HEAD
SRCDIR = /project/projectdirs/m2078/chulwoo/CPS_Gparity/src/cps2/cps_pp
BUILDDIR = /project/projectdirs/m2078/chulwoo/CPS_Gparity/build2/cps
=======
SRCDIR = /global/u2/c/ckelly/CPS/src/cps_pp_git/cps_pp
BUILDDIR = /global/homes/c/ckelly/CPS/bld/cps_pp_avx512
>>>>>>> 7c08a24df10cdde130a3cddc95db233175f5e66d
VPATH:=${SRCDIR}/include/input
RPCGEN=${SRCDIR}/misc/vml_encode/rpcgenvml/rpcgenvml
RHFLAGS=-h -C -x -u util/vml/ -v alg/
RCFLAGS=-c -C -x -u util/vml/ -v alg/

all: vml.done enum_int.stmp2 cg_arg.stmp do_arg.stmp w_spect_arg.stmp hmc_arg.stmp hmd_arg.stmp pbp_arg.stmp eig_arg.stmp pot_arg.stmp fix_gauge_arg.stmp meas_arg.stmp int_arg.stmp dens_arg.stmp array_arg.stmp ghb_arg.stmp noise_arg.stmp mom_arg.stmp hyp_smear_arg.stmp kinetic_smear_arg.stmp s_spect_arg.stmp bfm_arg.stmp 
all: a2a_arg.stmp

all: qpropw_arg.stmp nuc3pt_arg.stmp ape_smear_arg.stmp eigcg_arg.stmp  lanczos_arg.stmp  
all: quda_arg.stmp lanc_arg.stmp mobius_arg.stmp

vml.done: Makefile.in
	- cp ${BUILDDIR}/config.h .
	- cp ${BUILDDIR}/conf.h .
	- cp ${BUILDDIR}/precision.h .
	- cp ${SRCDIR}/include/input/*.x .
	touch vml.done
enum_int.stmp2 : enum.x vml.done
#	${RPCGEN} $(RHFLAGS) enum.x >${SRCDIR}/include/util/enum_int.h
	${RPCGEN} $(RHFLAGS) enum.x >${SRCDIR}/include/alg/enum_int.h
	${RPCGEN} $(RCFLAGS) enum.x >${SRCDIR}/src/alg/alg_arg/enum_int.C
	touch enum_int.stmp2
%.stmp : %.x vml.done
	${RPCGEN} $(RHFLAGS) $< >${SRCDIR}/include/alg/$*.h
	${RPCGEN} $(RCFLAGS) $< >${SRCDIR}/src/alg/alg_arg/$*.C
	touch $@

clean: 
	rm -f *stmp *stmp2 vml.done
