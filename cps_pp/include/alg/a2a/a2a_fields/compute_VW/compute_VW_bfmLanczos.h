#ifndef A2A_COMPUTE_VW_BFMLANCZOS_H_
#define A2A_COMPUTE_VW_BFMLANCZOS_H_

#include <alg/a2a/a2a_fields.h>

#ifdef USE_BFM_LANCZOS

#ifndef USE_BFM
#error "Must be using BFM"
#endif

#include<util/lattice/bfm_mixed_solver.h>
#include<util/lattice/bfm_evo.h>
#include<alg/eigen/Krylov_5d.h>

CPS_START_NAMESPACE

  //BFM for Lanczos and either BFM or Grid for A2A
  
  //In the Lanczos class you can choose to store the vectors in single precision (despite the overall precision, which is fixed to double here)
  //Set 'singleprec_evecs' if this has been done
  void computeVWlow(A2AvectorV<Policies> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bfm_evo<double> &dwf, bool singleprec_evecs);

  //singleprec_evecs specifies whether the input eigenvectors are stored in single precision
  //You can optionally pass a single precision bfm instance, which if given will cause the underlying CG to be performed in mixed precision.
  //WARNING: if using the mixed precision solve, the eigenvectors *MUST* be in single precision (there is a runtime check)
  void computeVWhigh(A2AvectorV<Policies> &V, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, Lattice &lat, const CGcontrols &cg_controls, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp = NULL);

  void computeVW(A2AvectorV<Policies> &V, Lattice &lat, BFM_Krylov::Lanczos_5d<double> &eig, bool singleprec_evecs, const CGcontrols &cg_controls, bfm_evo<double> &dwf_d, bfm_evo<float> *dwf_fp = NULL){
    computeVWlow(V,lat,eig,dwf_d,singleprec_evecs);
    computeVWhigh(V,eig,singleprec_evecs,lat,cg_controls,dwf_d,dwf_fp);
  }

#include "implementation/compute_VW_bfmLanczos.tcc"


CPS_END_NAMESPACE

#endif //USE_BFM_LANCZOS

#endif
