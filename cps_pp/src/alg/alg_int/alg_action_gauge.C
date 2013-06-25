#include<config.h>
CPS_START_NAMESPACE 
//------------------------------------------------------------------
//
// alg_action_gauge.C
//
// AlgActionGauge is represents the pure gauge contribution to the QCD
// action
//
//------------------------------------------------------------------

CPS_END_NAMESPACE
#include<alg/alg_hmd.h>
#include<util/lattice.h>
#include<util/vector.h>
#include<util/gjp.h>
#include<util/smalloc.h>
#include<util/verbose.h>
#include<util/error.h>
#include<util/time_cps.h>
#include<alg/alg_int.h>
CPS_START_NAMESPACE

AlgActionGauge::AlgActionGauge(AlgMomentum &mom, ActionGaugeArg &g_arg)
    : AlgAction(mom, g_arg.action_arg), cname("AlgActionGauge")
{
  int_type = INT_GAUGE;
  gauge_arg = &g_arg;
  gluon = g_arg.gluon;

}

AlgActionGauge::~AlgActionGauge() {

}

//!< Heat Bath for the gauge action (i.e., does nothing)
void AlgActionGauge::heatbath() {
}

//!< Calculate gauge contribution to the Hamiltonian
Float AlgActionGauge::energy() {
  Float dtime = -dclock();

  char *fname = "energy()";

  Lattice &lat = 
    LatticeFactory::Create(F_CLASS_NONE, gluon);
  Float h = lat.GhamiltonNode();

  {
    Float gsum_h(h);
    glb_sum(&gsum_h);
    if(UniqueID()==0)   printf("AlgActionGauge::energy() %e\n",gsum_h);
  }

  LatticeFactory::Destroy();

  dtime += dclock();
  print_flops(cname, fname, 0, dtime);
  return h;
}

void AlgActionGauge::prepare_fg(Matrix * force, Float dt_ratio)
{
  Float dtime = -dclock();
  const char *fname = "prepare_fg(M*,F)";

  Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, gluon);
  Fdt = lat.EvolveMomGforce(force, dt_ratio);
  if (force_measure == FORCE_MEASURE_YES) {
    char label[200];
    sprintf(label, "%s:", force_label);
    Fdt.print(dt_ratio, label);
  }
  LatticeFactory::Destroy();

  dtime += dclock();
  print_flops(cname, fname, 0, dtime);
}

//!< evolve method evolves the momentum due to the gauge force
void AlgActionGauge::evolve(Float dt, int steps) 
{
  Float dtime = -dclock();
  const char *fname = "evolve()";
  //!< Create an appropriate lattice
  Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, gluon);

  if(!UniqueID()){
    Float pvals[4];
    for(int ii=0;ii<4;ii++){
      int off = 18 * ii + 2;
      pvals[ii] = ((Float*)mom)[off];
    }

    printf("AlgActionGauge::evolve() start dt = %f, nsteps = %d, conj mom Px(0) = %.9e, Py(0) = %.9e, Pz(0) = %.9e, Pt(0) = %.9e\n",dt,steps,pvals[0],pvals[1],pvals[2],pvals[3]);
  }

  for (int i=0; i<steps; i++) {
    Fdt = lat.EvolveMomGforce(mom, dt);

    if (force_measure == FORCE_MEASURE_YES) {
      char label[200];
      sprintf(label, "%s:", force_label);
      Fdt.print(dt, label);
    }
  }

  LatticeFactory::Destroy();
  dtime += dclock();
  print_flops(cname, fname, 0, dtime);

  if(!UniqueID()){
    Float pvals[4];
    for(int ii=0;ii<4;ii++){
      int off = 18 * ii + 2;
      pvals[ii] = ((Float*)mom)[off];
    }

    printf("AlgActionGauge::evolve() end conj mom Px(0) = %.9e, Py(0) = %.9e, Pz(0) = %.9e, Pt(0) = %.9e\n",pvals[0],pvals[1],pvals[2],pvals[3]);
  }

}

//!< Dummy methods
void AlgActionGauge::cost(CgStats *cg_stats_global) {

}

void AlgActionGauge::init() {

}

void AlgActionGauge::copyConjLattice(){
  Lattice &lat = LatticeFactory::Create(F_CLASS_NONE, G_CLASS_NONE);
  lat.CopyConjGaugeField();
  LatticeFactory::Destroy();
}


CPS_END_NAMESPACE
