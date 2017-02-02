#ifndef _BFM_CGNE_M_HIGH_H
#define _BFM_CGNE_M_HIGH_H

#if defined(USE_BFM) && defined(USE_BFM_A2A) && defined(USE_NEW_BFM_GPARITY)
#include<util/lattice/bfm_evo.h>

CPS_START_NAMESPACE

template <class Float>
int CGNE_M_high(bfmbase<Float> &bfm, Fermion_t solution[2], Fermion_t source[2], multi1d<bfm_fermion> &evecs, multi1d<double> &evals, int nLowMode, bool isMIXprecision)
{
  int me = bfm.thread_barrier();
  Fermion_t src = bfm.threadedAllocFermion(); 
  Fermion_t hsrc = bfm.threadedAllocFermion(); 
  Fermion_t tmp = bfm.threadedAllocFermion(); 
  Fermion_t Mtmp= bfm.threadedAllocFermion(); 
  Fermion_t lsol= bfm.threadedAllocFermion(); 

  if (bfm.isBoss() && !me ) printf("CGNE_M_high: Solver type is %d\n",bfm.solver);

  double f;
  f = bfm.norm(source[0]);
  f+= bfm.norm(source[1]);

  if (bfm.isBoss() && !me ) printf("CGNE_M_high: Source norm is %le\n",f);
  f = bfm.norm(solution[0]);
  f+= bfm.norm(solution[1]);
  if (bfm.isBoss() && !me ) printf("CGNE_M_high: Guess norm is %le\n",f);

  // src_o = Mdag * (source_o - Moe MeeInv source_e)
  //
  // When using the CGdiagonalMee, we left multiply system of equations by 
  // diag(MeeInv,MooInv), and must multiply by an extra MooInv
  //

  if ( bfm.CGdiagonalMee ) { 
    bfm.MooeeInv(source[Even],tmp,DaggerNo);
    bfm.Meo     (tmp,src,Odd,DaggerNo);
    bfm.axpy    (src,src,source[Odd],-1.0);
    bfm.MooeeInv(src,tmp,DaggerNo);
    bfm.Mprec(tmp,src,Mtmp,DaggerYes);  
  } else { 
    bfm.MooeeInv(source[Even],tmp,DaggerNo);
    bfm.Meo     (tmp,src,Odd,DaggerNo);
    bfm.axpy    (tmp,src,source[Odd],-1.0);
    bfm.Mprec(tmp,src,Mtmp,DaggerYes);  
  }

  int Nev = evals.size();
  bfm.axpby   (lsol,src,src,0.0,0.0);
  bfm.axpby   (Mtmp,src,src,0.0,0.0);

  if(Nev < nLowMode) {
    if ( bfm.isBoss() && !me ) printf("Number of low eigen modes to do deflation is smaller than number of low modes to be substracted!\n");
    exit(4);
  }

  if(Nev > 0){
    if ( bfm.isBoss() && !me ) printf("bfmbase::CGNE: deflating with %d evecs. Evecs are %s precision\n",Nev, isMIXprecision ? "single" : "double");
  
    if(!isMIXprecision) {
      for(int n = 0; n < Nev; n++){
	std::complex<double> cn = bfm.dot(evecs[n][1], src);
	bfm.caxpy(lsol, evecs[n][1], lsol, cn.real() / double(evals[n]), cn.imag() / double(evals[n])  );

	if(n == nLowMode - 1) bfm.axpy(Mtmp, lsol, lsol, 0.);
      }
    }
    else { // evecs is stored in single precision
      for(int n = 0; n < Nev; n++){
	bfm.precisionChange(evecs[n][1], tmp, SingleToDouble, 1);
	std::complex<double> cn = bfm.dot(tmp, src);
	bfm.caxpy(lsol, tmp, lsol, cn.real() / double(evals[n]),  cn.imag() / double(evals[n]) );
	
	if(n == nLowMode - 1) bfm.axpy(Mtmp, lsol, lsol, 0.);
      }
    }
    bfm.axpy(solution[Odd], lsol, lsol, 0.0);      
  }


  f = bfm.norm(src);
  if (bfm.isBoss() && !me ) printf("CGNE_M_high: CGNE_prec_MdagM src norm %le\n",f);
  f = bfm.norm(solution[Odd]);
  if (bfm.isBoss() && !me ) printf("CGNE_M_high: CGNE_prec_MdagM guess norm %le\n",f);

  int iter = bfm.CGNE_prec_MdagM(solution[Odd],src);

  f = bfm.norm(solution[Odd]);
  if (bfm.isBoss() && !me ) printf("CGNE_M_high: CGNE_prec_MdagM sol norm %le\n",f);


  bfm.axpy(solution[Odd], Mtmp, solution[Odd], -1.0);

  f = bfm.norm(solution[Odd]);
  if (bfm.isBoss() && !me ) printf("CGNE_M_high: CGNE_prec_MdagM sol norm after subtracting low-mode part %le\n",f);
  
  bfm.threadedFreeFermion(Mtmp);
  bfm.threadedFreeFermion(lsol);
  bfm.threadedFreeFermion(hsrc);

  // sol_e = M_ee^-1 * ( src_e - Meo sol_o )...
  bfm.Meo(solution[Odd],tmp,Even,DaggerNo);    
  bfm.axpy(src,tmp,source[Even],-1.0);
  bfm.MooeeInv(src,solution[Even],DaggerNo);

  f = bfm.norm(solution[Even]);
  if (bfm.isBoss() && !me ) printf("CGNE_M_high: CGNE_prec_MdagM even checkerboard of sol %le\n",f);
  
  f =bfm.norm(solution[0]);
  f+=bfm.norm(solution[1]);
  if (bfm.isBoss() && !me ) printf("CGNE_M_high: unprec sol norm is %le\n",f);
  
  bfm.threadedFreeFermion(tmp);
  bfm.threadedFreeFermion(src);

  return iter;
}



CPS_END_NAMESPACE
#endif
#endif
