#include<config.h>
CPS_START_NAMESPACE
/*!\file
  \brief  Implementation of Gwilson class.

  $Id: g_wilson.C,v 1.11 2008-09-18 15:23:17 chulwoo Exp $
*/
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: chulwoo $
//  $Date: 2008-09-18 15:23:17 $
//  $Header: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/util/lattice/g_wilson/g_wilson.C,v 1.11 2008-09-18 15:23:17 chulwoo Exp $
//  $Id: g_wilson.C,v 1.11 2008-09-18 15:23:17 chulwoo Exp $
//  $Name: not supported by cvs2svn $
//  $Locker:  $
//  $Revision: 1.11 $
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/src/util/lattice/g_wilson/g_wilson.C,v $
//  $State: Exp $
//
//--------------------------------------------------------------------
//------------------------------------------------------------------
//
// g_wilson.C
//
// Gwilson is derived from Lattice and is relevant to the 
// standard Wilson single plaquette action.
//
//------------------------------------------------------------------

CPS_END_NAMESPACE
#include <util/lattice.h>
#include <util/verbose.h>
#include <util/vector.h>
#include <util/gjp.h>
#include <util/gw_hb.h>
#include <util/time_cps.h>
#include <comms/nga_reg.h>
#include <comms/glb.h>
#include <comms/cbuf.h>
#include <cassert>
CPS_START_NAMESPACE


//------------------------------------------------------------------
enum { MATRIX_SIZE = 18 };
//------------------------------------------------------------------

//------------------------------------------------------------------
// static variables used only inside this file
//------------------------------------------------------------------
static IFloat invs3 = -1./3.;




//  CRAM temp buffer
#ifdef _TARTAN
static Matrix *mp0 = (Matrix *)CRAM_SCRATCH_ADDR;	// ihdot
static Matrix *mp1 = mp0 + 1;
static Matrix *mp2 = mp1 + 1;
//static Matrix *mp3 = mp2 + 1;
#else
static Matrix mt0;
static Matrix mt1;
static Matrix mt2;
//static Matrix mt3;
static Matrix *mp0 = &mt0;		// ihdot
static Matrix *mp1 = &mt1;
static Matrix *mp2 = &mt2;
//static Matrix *mp3 = &mt3;
#endif 


//------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------
Gwilson::Gwilson()
{
  cname = "Gwilson";
  char *fname = "Gwilson()";
  VRB.Func(cname,fname);
}


//------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------
Gwilson::~Gwilson()
{
  char *fname = "~Gwilson()";
  VRB.Func(cname,fname);
}


//------------------------------------------------------------------
// GclassType Gclass(void):
// It returns the type of gauge class
//------------------------------------------------------------------
GclassType Gwilson::Gclass(void){
  return G_CLASS_WILSON;
}

const unsigned CBUF_MODE4 = 0xcca52112;

void Gwilson::SigmaHeatbath() 
{
  const char* fname = "SigmaHeatBath()";
  VRB.Result(cname, fname, "Entering SigmaHeatBath()\n");

  int x[4];

  //floats because glb_sum works on floats
  Float n_zero = 0;
  Float n_one = 0;

  Float max_plaq = -999.0;
  Float min_plaq = +999.0;
  
  for(x[0] = 0; x[0] < node_sites[0]; ++x[0]) {
    for(x[1] = 0; x[1] < node_sites[1]; ++x[1]) {
      for(x[2] = 0; x[2] < node_sites[2]; ++x[2]) {
	for(x[3] = 0; x[3] < node_sites[3]; ++x[3]) {
	  for (int mu = 0; mu < 3; ++mu) {
	    for(int nu = mu+1; nu < 4; ++nu) {
              Float re_tr_plaq = ReTrPlaqNonlocal(x, mu, nu);
              //printf("sigmaheatbath re_tr_plaq = %e\n", re_tr_plaq);
              if(re_tr_plaq > max_plaq) max_plaq = re_tr_plaq;
              if(re_tr_plaq < min_plaq) min_plaq = re_tr_plaq;
              
              Float exponent = -DeltaS(re_tr_plaq);
              if(exponent >= 0) printf("re_tr_plaq = %e\n", re_tr_plaq);
              assert(exponent < 0);
              Float probability_zero = exp(exponent);

              LRG.AssignGenerator(x);
              IFloat rand = LRG.Urand(0.0, 1.0);
              if(!(rand >= 0 && rand <= 1)) printf("rand = %e\n", rand);
              assert(rand >= 0 && rand <= 1);
              if(rand < probability_zero) {
                *(SigmaField() + SigmaOffset(x, mu, nu)) = 0;
                n_zero += 1.0;
              } else {
                *(SigmaField() + SigmaOffset(x, mu, nu)) = 1;
                n_one += 1.0;
              }
            }
          }
        }
      }
    }
  }

  glb_sum(&n_zero);
  glb_sum(&n_one);

  glb_max(&max_plaq);
  glb_min(&min_plaq);

  VRB.Result(cname, fname, "Finished: n_zero = %f, n_one = %f; max_plaq = %f, min_plaq = %f\n", n_zero, n_one, max_plaq, min_plaq);
}

//------------------------------------------------------------------
/*!
  \param force The computed force from the gauge action.
  \param x the lattice site coordinates.
  \param mu The direction mu.
  \todo Could this not be be a virtual Lattice method?
*/
//------------------------------------------------------------------
void Gwilson::GforceSite(Matrix& force, int *x, int mu)
{
  char *fname = "GforceSite(M&,i*,i)";
//  VRB.Func(cname,fname);

  setCbufCntrlReg(4, CBUF_MODE4);

  Matrix *u_off = GaugeField()+GsiteOffset(x)+mu;

  Float plaq_multiplier = GJP.Beta()*invs3;

  //----------------------------------------
  //  get staple
  //     mp1 = staple
  //----------------------------------------
  //Staple(*mp1, x, mu);	
  StapleWithSigmaCorrections(*mp1, x, mu);
  ForceFlops += 198*3*3+12+216*3;
  

  //----------------------------------------
  // mp2 = U_mu(x)
  //----------------------------------------
  moveMem((IFloat *)mp2, (IFloat *)u_off,
  	MATRIX_SIZE * sizeof(IFloat));

  
  //----------------------------------------
  // force = -beta/3*U_mu(x)*stap
  // "-" because no staggered phase
  //----------------------------------------
  mDotMEqual((IFloat *)&force, (const IFloat *)mp2, (const IFloat *)mp1);

  vecTimesEquFloat((IFloat *)&force, plaq_multiplier, MATRIX_SIZE);

  
  // mp1 and mp2 free

  mp1->Dagger((IFloat *)&force);
  force.TrLessAntiHermMatrix(*mp1);

  for(int i = 0; i < 18; i++) {
    Float a = *(((Float*)&force) + i);
    assert(!(a != a));
  }
  ForceFlops += 198+18+24;
}

//------------------------------------------------------------------
// Float GhamiltonNode(void):
// The pure gauge Hamiltonian of the node sublattice.
//------------------------------------------------------------------
Float Gwilson::GhamiltonNode(void){
  char *fname = "GhamiltonNode()";
  VRB.Func(cname,fname);

  Float plaq_multiplier = GJP.Beta()*invs3;
  Float sum = SumReTrPlaqNode();
  sum *= plaq_multiplier;

  Float sigma_energy = SumSigmaEnergyNode();

  Float glb_normal_energy = sum;
  Float glb_sigma_energy = sigma_energy;
  glb_sum(&glb_normal_energy);
  glb_sum(&glb_sigma_energy);

  sum += sigma_energy;

  VRB.Result(cname, fname, "glb_normal_energy = %f, glb_sigma_energy = %f\n", glb_normal_energy, glb_sigma_energy);

  int x[] = {0, -1, 0, 0};
  VRB.Result(cname, fname, "after hamilton re_tr_plaq = %e\n", ReTrPlaqNonlocal(x, 0, 1));

  return sum;

}

//------------------------------------------------------------------------------
// void GactionGradient(Matrix &grad, int *x, int mu)
//   Calculates the partial derivative of the gauge action
//   w.r.t. the link U_mu(x).
//------------------------------------------------------------------------------
void Gwilson::GactionGradient(Matrix &grad, int *x, int mu)
{
  char *fname = "GactionGradient(M&,I*,I)" ;
  VRB.Func(cname, fname) ;

  ERR.NotImplemented(cname, fname);

  //----------------------------------------------------------------------------
  // get staple
  //----------------------------------------------------------------------------
  Staple(grad, x, mu) ;

  //----------------------------------------------------------------------------
  // grad = - (beta/3) * staple
  //   N.B. invs3 should be -1/3
  //----------------------------------------------------------------------------
  Float tmp = GJP.Beta() * invs3 ;
  vecTimesEquFloat((IFloat *)&grad, tmp, MATRIX_SIZE) ;
}

/*!
  The staple sum around the link \f$ U_\mu(x) \f$ is
\f[
  \sum_{\nu \neq \mu} [
           U_\nu(x+\mu) U^\dagger_\mu(x+\nu) U^\dagger_\nu(x)                  
        +  U^\dagger_\nu(x+\mu-\nu) U^\dagger_\mu(x-\nu) U_\nu(x-\nu)  ]
\f]
  \param x The coordinates of the lattice site 
  \param mu The link direction
  \param stap The computed staple sum.
*/
void Gwilson::AllStaple(Matrix & stap, const int *x, int mu){
  char * fname = "AllStaple()";
  VRB.Func(cname, fname);
  ERR.NotImplemented(cname, fname);
  BufferedStaple(stap, x, mu); 
}

CPS_END_NAMESPACE
