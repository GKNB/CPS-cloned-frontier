#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <util/qcdio.h>
#ifdef PARALLEL
#include <comms/sysfunc_cps.h>
#endif
#include <comms/scu.h>
#include <comms/glb.h>

#include <util/lattice.h>
#include <util/time_cps.h>
#include <util/smalloc.h>

#include <util/command_line.h>

#include<unistd.h>
#include<config.h>

#include <util/spincolorflavormatrix.h>
#include <alg/propmanager.h>
#include <alg/fix_gauge_arg.h>
#include <alg/alg_gparitycontract.h>

#include <string>
#include <sstream>
#include <bitset>

#ifdef USE_OMP
#include <omp.h>
#endif

CPS_START_NAMESPACE

void QuarkMomCombination::reset(){ quark_mom.clear(); }
void QuarkMomCombination::add_prop(const PropagatorContainer &prop, const bool &conj){
  std::vector<std::vector<int> > qmom = prop.get_allowed_momenta();
  if(qmom.size()==0){ ERR.General(cname,"add_prop","Propagator has no allowed momenta!\n"); }
  if(conj) //complex conjugation of prop inverts the allowed sink momenta
    for(int i=0;i<qmom.size();i++)
      for(int j=0;j<3;j++) qmom[i][j]*=-1;
  quark_mom.push_back(qmom);
}
std::vector<std::vector<int> > QuarkMomCombination::get_total_momenta() const{
  std::vector<std::vector<int> > out;
  std::vector<int> start(3,0);
  mom_comb_recurse(out,start,0);
  if(out.size()==0){ ERR.General(cname,"get_total_momenta","Found no momentum combinations!\n"); }
  return out;
}
void QuarkMomCombination::mom_comb_recurse(std::vector<std::vector<int> > &into, std::vector<int> &cur_branch, const int &depth) const{
  if(depth == quark_mom.size()){
    into.push_back(cur_branch);
  }else{
    for(int poss=0;poss<quark_mom[depth].size();poss++){ //loop over possible momenta for this quark
      const std::vector<int> &choice = quark_mom[depth][poss];
      std::vector<int> new_branch(cur_branch);
      for(int i=0;i<3;i++) new_branch[i]+=choice[i];
      mom_comb_recurse(into,new_branch,depth+1);
    }
  }
}
bool QuarkMomCombination::contains(const std::vector<int> &what, const std::vector<std::vector<int> > &in){
  for(int i=0;i<in.size();i++){
    if(what == in[i]) return true;
  }
  return false;
}
bool QuarkMomCombination::contains(const std::vector<int> &what) const{
  std::vector<std::vector<int> > in = get_total_momenta();
  return contains(what,in);
}
bool QuarkMomCombination::contains(const int *what) const{
  std::vector<int> w(3); for(int i=0;i<3;i++) w[i] = what[i];
  return contains(w);
}
  



const Float AlgGparityContract::Pi_const(3.141592654);

AlgGparityContract::AlgGparityContract(Lattice & latt, CommonArg& c_arg, GparityContractArg& arg): Alg(latt,&c_arg), args(&arg){ 
  cname = "AlgGparityContract"; 
  
  shift_x = GJP.XnodeCoor()*GJP.XnodeSites();
  shift_y = GJP.YnodeCoor()*GJP.YnodeSites();
  shift_z = GJP.ZnodeCoor()*GJP.ZnodeSites();
  shift_t = GJP.TnodeCoor()*GJP.TnodeSites();
  //Local lattice dimensions:
  size_x = GJP.XnodeSites();
  size_y = GJP.YnodeSites();
  size_z = GJP.ZnodeSites();
  size_t = GJP.TnodeSites();
  size_xy = size_x*size_y;
  spatial_vol = (GJP.VolNodeSites()/GJP.TnodeSites()); // =size_x*size_y_size_z
}

void AlgGparityContract::global_coord(const int &site, int *into_vec){
  into_vec[3] = site/spatial_vol + shift_t;
  into_vec[2] = (site%spatial_vol)/size_xy + shift_z;
  into_vec[1] = (site%size_xy)/size_x + shift_y;
  into_vec[0] = site%size_x + shift_x;
}
Rcomplex AlgGparityContract::sink_phasefac(const int *momphase, const int *pos,const bool &is_cconj){
  //momphase is the sum of the phase factors from the propagators forming the contraction
  //NOTE: In G-parity directions, momentum is discretised in odd units of \pi/2L rather than even/odd units of 2\pi/L (periodic/antiperiodic).
  Float pdotx = 0.0;
  Float sgn = 1.0;
  if(is_cconj) sgn = -1.0;

  for(int d=0;d<3;d++){
    Float mom_unit;
    if(GJP.Bc(d) == BND_CND_GPARITY) mom_unit = Pi_const/( (Float) 2*GJP.Nodes(d)*GJP.NodeSites(d));
    else if(GJP.Bc(d) == BND_CND_PRD) mom_unit = 2.0*Pi_const/( (Float) GJP.Nodes(d)*GJP.NodeSites(d));
    else if(GJP.Bc(d) == BND_CND_APRD) mom_unit = Pi_const/( (Float) GJP.Nodes(d)*GJP.NodeSites(d));
    else ERR.General(cname,"sink_phasefac(int *,const int &)","Unknown boundary condition\n");
    
    pdotx += sgn*momphase[d]*pos[d]*mom_unit;
  }
  return Rcomplex(cos(pdotx),sin(pdotx));
}
Rcomplex AlgGparityContract::sink_phasefac(const int *momphase,const int &site,const bool &is_cconj){
  int pos[4]; global_coord(site,pos);
  return sink_phasefac(momphase,pos,is_cconj);
}
void AlgGparityContract::sum_momphase(int *into, PropagatorContainer &prop){
  int propmom[3]; prop.momentum(propmom);
  for(int i=0;i<3;i++) into[i]+=propmom[i];
}

void AlgGparityContract::run(const int &conf_idx){
  //Calculate propagators first. When contracting on only a single thread
  //this is not strictly necessary as the PropagatorContainer will calculate
  //the prop if it has not already been done. However in a multi-threaded
  //inversion, all the threads try to calculate the prop independently, and it will crash.
  PropManager::calcProps(AlgLattice());

  for(int i=0;i<args->meas.meas_len;i++){
    spectrum(args->meas.meas_val[i],conf_idx);   
  }
}

//Left multiply by a gamma matrix structure in QDP-style conventions:
//\Gamma(n) = \gamma_1^n1 \gamma_2^n2  \gamma_3^n3 \gamma_4^n4    where ni are bit fields: n4 n3 n2 n1
void AlgGparityContract::qdp_gl(WilsonMatrix &wmat, const int &gidx) const{
  std::bitset<4> gmask(gidx);
  if(gmask == std::bitset<4>(15)){ wmat.gl(-5); return; }

  //CPS conventions \gamma^4 = gl(3) etc
  for(int i=3;i>=0;i--) if(gmask[i]) wmat.gl(i);
}
//Right multiply by a gamma matrix structure in QDP-style conventions:
//\Gamma(n) = \gamma_1^n1 \gamma_2^n2  \gamma_3^n3 \gamma_4^n4    where ni are bit fields: n4 n3 n2 n1 
void AlgGparityContract::qdp_gr(WilsonMatrix &wmat, const int &gidx) const{
  std::bitset<4> gmask(gidx);
  if(gmask == std::bitset<4>(15)){ wmat.gr(-5); return; }

  //CPS conventions \gamma^4 = gl(3) etc
  for(int i=0;i<4;i++) if(gmask[i]) wmat.gr(i);
}

void AlgGparityContract::qdp_gl(SpinColorFlavorMatrix &wmat, const int &gidx) const{
  std::bitset<4> gmask(gidx);
  if(gmask == std::bitset<4>(15)){ wmat.gl(-5); return; }

  //CPS conventions \gamma^4 = gl(3) etc
  for(int i=3;i>=0;i--) if(gmask[i]) wmat.gl(i);
}
void AlgGparityContract::qdp_gr(SpinColorFlavorMatrix &wmat, const int &gidx) const{
  std::bitset<4> gmask(gidx);
  if(gmask == std::bitset<4>(15)){ wmat.gr(-5); return; }

  //CPS conventions \gamma^4 = gl(3) etc
  for(int i=0;i<4;i++) if(gmask[i]) wmat.gr(i);
}
//Coefficient when matrix is transposed or conjugated
Float AlgGparityContract::qdp_gcoeff(const int &gidx, const bool &transpose, const bool &conj) const{
  if(gidx==2 || gidx==8 || gidx==15){ return 1.0; } //gamma^2, gamma^4 and gamma^5 are hermitian and real
  else if(gidx==1 || gidx==4){ //gamma^1 and gamma^3 are hermitian and imaginary
    if(transpose && conj) return 1.0;
    else return -1.0;
  } 
  std::bitset<4> gmask(gidx);
  Float out(1);
  if(transpose) out *= -1.0; //- sign for reordering 2 or 3 gamma matrices
    
  std::bitset<4> mask(1);
  for(int i=0;i<4;i++, mask<<1){
    if( (mask &gmask).any() ) out *= qdp_gcoeff((int)mask.to_ulong(),transpose,conj);
  }
  return out;
}


static void MomCombError(const char* cname, const char* fname, const int* desired_mom, const QuarkMomCombination &momcomb){
  if(!UniqueID()){
    printf("%s::%s Desired momentum (%d,%d,%d) not available. Available combinations are:\n",cname,fname,desired_mom[0],desired_mom[1],desired_mom[2]);
    std::vector<std::vector<int> > mc = momcomb.get_total_momenta();
    for(int i=0;i<mc.size();i++) printf("%d: (%d,%d,%d)\n",i,mc[i][0],mc[i][1],mc[i][2]);
  }
  fflush(stdout);
  sync();
  exit(-1);
}


void AlgGparityContract::meson_LL_std(PropagatorContainer &prop, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp){  
  /*Mesons comprising $ \bar u $ and $ d$*/
  std::ostringstream os; os << "LL_MESON " << gamma_idx_1 << " " << gamma_idx_2;
  CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

  if(UniqueID()==0) printf("Doing LL_MESON %d %d contraction\n",gamma_idx_1,gamma_idx_2);

  /*Mesons comprising $ \bar u $ and $ d$*/
  /*Require a "CorrelationFunction &corrfunc"*/
  /*Require propagator "PropagatorContainer &prop_src_y_0_pcon corresponding to \mathcal{G}^{(0)}_{x,y}*/
  PropagatorContainer &prop_src_y_0_pcon = prop;

  /*Fourier transform on sink index x*/
  /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
  const int *desired_mom_x = sink_mom;
  {
    /*[-1 ]*[{\rm tr}_{sc,0}\left\{S_2 \gamma^5 \mathcal{G}^{(0) \dagger}_{x,y} \gamma^5 S_1 \mathcal{G}^{(0)}_{x,y}\right\}_{0}  ]*/
    QuarkMomCombination momcomb;
    momcomb.add_prop(prop_src_y_0_pcon, false);
    momcomb.add_prop(prop_src_y_0_pcon, true);
    bool desired_mom_available(momcomb.contains(desired_mom_x));
    /*Create an appropriate error message if !desired_mom_available*/
    if(!desired_mom_available) MomCombError("AlgGparityContract","meson_LL_std",sink_mom,momcomb);
  }


  corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
  for(int x=0;x<GJP.VolNodeSites();x++){
    int x_pos_vec[4];
    global_coord(x,x_pos_vec);
  
    /*Get all WilsonMatrices needed*/
    WilsonMatrix prop_snk_x_0_src_y_0_hconj_wmat(prop_src_y_0_pcon.getProp(AlgLattice()).SiteMatrix(x,0));
    prop_snk_x_0_src_y_0_hconj_wmat.hconj();
  
    WilsonMatrix& prop_snk_x_0_src_y_0_wmat = prop_src_y_0_pcon.getProp(AlgLattice()).SiteMatrix(x,0);
  
    /*Starting contraction 0*/
    /*[-1 ]*[{\rm tr}_{sc,0}\left\{S_2 \gamma^5 \mathcal{G}^{(0) \dagger}_{x,y} \gamma^5 S_1 \mathcal{G}^{(0)}_{x,y}\right\}_{0}  ]*/
  
    {
      Rcomplex contraction(-1 , 0);
      contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
      Rcomplex result_subdiag1(1.0);
      {
	WilsonMatrix sdiag1_trset0_wmat_prod(prop_snk_x_0_src_y_0_hconj_wmat);
	sdiag1_trset0_wmat_prod.gl(-5);
	//sdiag1_trset0_wmat_prod = S_2 * sdiag1_trset0_wmat_prod;
	qdp_gl(sdiag1_trset0_wmat_prod,gamma_idx_2);

	sdiag1_trset0_wmat_prod.gr(-5);
	//sdiag1_trset0_wmat_prod *= S_1;
	qdp_gr(sdiag1_trset0_wmat_prod,gamma_idx_1);

	sdiag1_trset0_wmat_prod *= prop_snk_x_0_src_y_0_wmat;
	Rcomplex sdiag1_trset0_cmplx;
	sdiag1_trset0_cmplx = sdiag1_trset0_wmat_prod.Trace();
	result_subdiag1 *= sdiag1_trset0_cmplx;
      }
      contraction *= result_subdiag1;
    
    
      corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
    }
  }
  corrfunc.write(fp);
}

void AlgGparityContract::meson_LL_gparity(PropagatorContainer &prop, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp){
  /*Mesons comprising $ \bar u $ and $ d$*/
  /*Require a "CorrelationFunction &corrfunc"*/
  std::ostringstream os; os << "LL_MESON " << gamma_idx_1 << " " << gamma_idx_2;
  CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

  if(UniqueID()==0) printf("Doing LL_MESON %d %d contraction\n",gamma_idx_1,gamma_idx_2);

  /*Mesons comprising $ \bar u $ and $ d$*/
  /*Require a "CorrelationFunction &corrfunc"*/
  /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
  PropagatorContainer &prop_src_y_u_d_eitherflav_pcon = prop;

  /*Fourier transform on sink index x*/
  /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
  const int *desired_mom_x = sink_mom;

  {
    QuarkMomCombination momcomb;
    momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, false);
    momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, false);
    bool desired_mom_available(momcomb.contains(desired_mom_x));
    /*Create an appropriate error message if !desired_mom_available*/
    if(!desired_mom_available) MomCombError("AlgGparityContract","meson_LL_gparity",sink_mom,momcomb);
  }

  corrfunc.setNcontractions(2);
#pragma omp parallel for default(shared)
  for(int x=0;x<GJP.VolNodeSites();x++){
    int x_pos_vec[4];
    global_coord(x,x_pos_vec);
  
    /*Get all SpinColorFlavorMatrices needed*/
    SpinColorFlavorMatrix prop_ud_snk_x_src_y_trans_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
    prop_ud_snk_x_src_y_trans_scfmat.transpose();
  
    SpinColorFlavorMatrix prop_ud_snk_x_src_y_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
  
    /*Starting contraction 0*/
    /*[1 ]*[{\rm tr}_{scf,0}\left\{C S_2^T F_1 F_\updownarrow \mathcal{G}^{[u/d] T}_{x,y} F_1 F_\updownarrow C S_1 \mathcal{G}^{[u/d] }_{x,y}\right\}_{0}  ]*/
  
    {
      Rcomplex contraction(1 , 0);
      contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
      Rcomplex result_subdiag1(1.0);
      {
	SpinColorFlavorMatrix sdiag1_trset0_scfmat_prod(prop_ud_snk_x_src_y_trans_scfmat);
	sdiag1_trset0_scfmat_prod.pl(Fud);
	sdiag1_trset0_scfmat_prod.pl(F1);
	//sdiag1_trset0_scfmat_prod = S_2.trans() * sdiag1_trset0_scfmat_prod;
	qdp_gl(sdiag1_trset0_scfmat_prod,gamma_idx_2);
	result_subdiag1 *= qdp_gcoeff(gamma_idx_2,true,false);

	sdiag1_trset0_scfmat_prod.ccl(-1);
	sdiag1_trset0_scfmat_prod.pr(F1);
	sdiag1_trset0_scfmat_prod.pr(Fud);
	sdiag1_trset0_scfmat_prod.ccr(1);
	//sdiag1_trset0_scfmat_prod *= S_1;
	qdp_gr(sdiag1_trset0_scfmat_prod,gamma_idx_1);

	sdiag1_trset0_scfmat_prod *= prop_ud_snk_x_src_y_scfmat;
	Rcomplex sdiag1_trset0_cmplx;
	sdiag1_trset0_cmplx = sdiag1_trset0_scfmat_prod.Trace();
	result_subdiag1 *= sdiag1_trset0_cmplx;
      }
      contraction *= result_subdiag1;
    
    
      corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
    }
    /*Starting contraction 1*/
    /*[{\rm tr}_{scf,0}\left\{S_2 C F_0 F_\updownarrow \mathcal{G}^{[u/d] T}_{x,y} F_1 F_\updownarrow C S_1 \mathcal{G}^{[u/d] }_{x,y}\right\}_{0}  ]*[1 ]*/
  
    {
      Rcomplex contraction(1 , 0);
      contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
      Rcomplex result_subdiag0(1.0);
      {
	SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_ud_snk_x_src_y_trans_scfmat);
	sdiag0_trset0_scfmat_prod.pl(Fud);
	sdiag0_trset0_scfmat_prod.pl(F0);
	sdiag0_trset0_scfmat_prod.ccl(-1);
	//sdiag0_trset0_scfmat_prod = S_2 * sdiag0_trset0_scfmat_prod;
	qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_2);

	sdiag0_trset0_scfmat_prod.pr(F1);
	sdiag0_trset0_scfmat_prod.pr(Fud);
	sdiag0_trset0_scfmat_prod.ccr(1);
	//sdiag0_trset0_scfmat_prod *= S_1;
	qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_1);

	sdiag0_trset0_scfmat_prod *= prop_ud_snk_x_src_y_scfmat;
	Rcomplex sdiag0_trset0_cmplx;
	sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	result_subdiag0 *= sdiag0_trset0_cmplx;
      }
      contraction *= result_subdiag0;
    
    
      corrfunc(omp_get_thread_num(),1,x_pos_vec[3]) += contraction;
    }
  }
  corrfunc.write(fp);
}

void AlgGparityContract::contract_LL_mesons(const ContractionTypeLLMesons &args, const int &conf_idx){
  std::ostringstream file; file << args.file << "." << conf_idx;

  FILE *fp;
  if ((fp = Fopen(file.str().c_str(), "w")) == NULL) {
    ERR.FileW("CorrelationFunction","write(const char *file)",file.str().c_str());
  }

  PropagatorContainer &prop = PropManager::getProp(args.prop_L);

  //loop through LL meson correlation functions
  if(GJP.Gparity()){
    for(int g1=0;g1<16;g1++){
      for(int g2=0;g2<16;g2++){
	meson_LL_gparity(prop, args.sink_mom, g1, g2, fp);
      }
    }
  }else{
    for(int g1=0;g1<16;g1++){
      for(int g2=0;g2<16;g2++){
	meson_LL_std(prop, args.sink_mom, g1, g2, fp);
      }
    }
  }
  Fclose(fp);
}


void AlgGparityContract::meson_HL_gparity(PropagatorContainer &prop_H,PropagatorContainer &prop_L, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp){
  PropagatorContainer &prop_src_y_u_d_eitherflav_pcon = prop_L;
  PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon = prop_H;
  const int *desired_mom_x = sink_mom;

  {
    /*<<(\bar s',s)*(\bar s,s')>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_SPRIME_S_S_SPRIME " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_SPRIME_S_S_SPRIME %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, false);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, false);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(2);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
  
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_trans_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
      prop_sprimes_snk_x_src_y_trans_scfmat.transpose();
  
      /*Starting contraction 0*/
      /*[{\rm tr}_{scf,0}\left\{C \Gamma[g2] F_1 F_\updownarrow \mathcal{G}^{[s^\prime/s] T}_{x,y} F_1 F_\updownarrow C \Gamma[g1] \mathcal{G}^{[s^\prime/s] }_{x,y}\right\}_{0}  ]*[f_\Gamma(g2,T) ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= qdp_gcoeff(gamma_idx_2,true,false);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_trans_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(Fud);
	  sdiag0_trset0_scfmat_prod.pl(F1);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.ccl(-1);
	  sdiag0_trset0_scfmat_prod.pr(F1);
	  sdiag0_trset0_scfmat_prod.pr(Fud);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod *= prop_sprimes_snk_x_src_y_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
      /*Starting contraction 1*/
      /*[{\rm tr}_{scf,0}\left\{\Gamma[g2] C F_0 F_\updownarrow \mathcal{G}^{[s^\prime/s] T}_{x,y} F_1 F_\updownarrow C \Gamma[g1] \mathcal{G}^{[s^\prime/s] }_{x,y}\right\}_{0}  ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_trans_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(Fud);
	  sdiag0_trset0_scfmat_prod.pl(F0);
	  sdiag0_trset0_scfmat_prod.ccl(-1);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.pr(F1);
	  sdiag0_trset0_scfmat_prod.pr(Fud);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod *= prop_sprimes_snk_x_src_y_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),1,x_pos_vec[3]) += contraction;
      }
    }

    corrfunc.write(fp);
  }


  {
    /*<<(\bar d,s)*(\bar s,d)>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_D_S_S_D " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_D_S_S_D %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, true);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, false);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_hconj_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
      prop_ud_snk_x_src_y_hconj_scfmat.hconj();
  
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
  
      /*Starting contraction 0*/
      /*[-1 ]*[{\rm tr}_{scf,0}\left\{\gamma^5 \Gamma[g1] F_0 \mathcal{G}^{[s^\prime/s] }_{x,y} F_0 \Gamma[g2] \gamma^5 \mathcal{G}^{[u/d] \dagger}_{x,y}\right\}_{0}  ]*/
  
      {
	Rcomplex contraction(-1 , 0);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag1(1.0);
	{
	  SpinColorFlavorMatrix sdiag1_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_scfmat);
	  sdiag1_trset0_scfmat_prod.pl(F0);
	  qdp_gl(sdiag1_trset0_scfmat_prod,gamma_idx_1);
	  sdiag1_trset0_scfmat_prod.gl(-5);
	  sdiag1_trset0_scfmat_prod.pr(F0);
	  qdp_gr(sdiag1_trset0_scfmat_prod,gamma_idx_2);
	  sdiag1_trset0_scfmat_prod.gr(-5);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_x_src_y_hconj_scfmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_scfmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	contraction *= result_subdiag1;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }
    corrfunc.write(fp);
  }

  {
    /*<<(\bar d,s)*(\bar u,s')>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_D_S_U_SPRIME " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_D_S_U_SPRIME %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, true);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, false);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_hconj_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
      prop_ud_snk_x_src_y_hconj_scfmat.hconj();
  
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
  
      /*Starting contraction 0*/
      /*[{\rm tr}_{scf,0}\left\{\gamma^5 \Gamma[g1] F_0 \mathcal{G}^{[s^\prime/s] }_{x,y} F_1 C \Gamma[g2] \gamma^5 C \mathcal{G}^{[u/d] \dagger}_{x,y}\right\}_{0}  ]*[f_\Gamma(g2,T) ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= qdp_gcoeff(gamma_idx_2,true,false);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(F0);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod.gl(-5);
	  sdiag0_trset0_scfmat_prod.pr(F1);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.gr(-5);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_x_src_y_hconj_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }
    corrfunc.write(fp);

  }

  {
    /*<<(\bar d,s')*(\bar s',d)>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_D_SPRIME_SPRIME_D " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_D_SPRIME_SPRIME_D %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, true);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, true);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_cconj_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
      prop_sprimes_snk_x_src_y_cconj_scfmat.cconj();
  
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_hconj_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
      prop_ud_snk_x_src_y_hconj_scfmat.hconj();
  
      /*Starting contraction 0*/
      /*[{\rm tr}_{scf,0}\left\{\gamma^5 \Gamma[g1] \gamma^5 C F_0 F_\updownarrow \mathcal{G}^{[s^\prime/s] *}_{x,y} F_1 F_\updownarrow \gamma^5 C \Gamma[g2] \gamma^5 \mathcal{G}^{[u/d] \dagger}_{x,y}\right\}_{0}  ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_cconj_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(Fud);
	  sdiag0_trset0_scfmat_prod.pl(F0);
	  sdiag0_trset0_scfmat_prod.ccl(-1);
	  sdiag0_trset0_scfmat_prod.gl(-5);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod.gl(-5);
	  sdiag0_trset0_scfmat_prod.pr(F1);
	  sdiag0_trset0_scfmat_prod.pr(Fud);
	  sdiag0_trset0_scfmat_prod.gr(-5);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.gr(-5);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_x_src_y_hconj_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }
    
    corrfunc.write(fp);
  }

  {
    /*<<(\bar d,s')*(\bar u,s)>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/    
    std::ostringstream os; os << "HL_MESON_D_SPRIME_U_S " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_D_SPRIME_U_S %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, true);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, true);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_hconj_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
      prop_ud_snk_x_src_y_hconj_scfmat.hconj();
  
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_cconj_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
      prop_sprimes_snk_x_src_y_cconj_scfmat.cconj();
  
      /*Starting contraction 0*/
      /*[{\rm tr}_{scf,0}\left\{\gamma^5 \Gamma[g1] \gamma^5 C F_0 F_\updownarrow \mathcal{G}^{[s^\prime/s] *}_{x,y} F_0 F_\updownarrow \gamma^5 \Gamma[g2] \gamma^5 C \mathcal{G}^{[u/d] \dagger}_{x,y}\right\}_{0}  ]*[f_\Gamma(g2,T) ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= qdp_gcoeff(gamma_idx_2,true,false);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_cconj_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(Fud);
	  sdiag0_trset0_scfmat_prod.pl(F0);
	  sdiag0_trset0_scfmat_prod.ccl(-1);
	  sdiag0_trset0_scfmat_prod.gl(-5);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod.gl(-5);
	  sdiag0_trset0_scfmat_prod.pr(F0);
	  sdiag0_trset0_scfmat_prod.pr(Fud);
	  sdiag0_trset0_scfmat_prod.gr(-5);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.gr(-5);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_x_src_y_hconj_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }
    corrfunc.write(fp);
  }

  {
    /*<<(\bar u,s)*(\bar s,u)>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_U_S_S_U " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_U_S_S_U %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, false);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, false);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
  
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_trans_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
      prop_sprimes_snk_x_src_y_trans_scfmat.transpose();
  
      /*Starting contraction 0*/
      /*[{\rm tr}_{scf,0}\left\{C \Gamma[g2] F_1 F_\updownarrow \mathcal{G}^{[s^\prime/s] T}_{x,y} F_0 F_\updownarrow \Gamma[g1] C \mathcal{G}^{[u/d] }_{x,y}\right\}_{0}  ]*[f_\Gamma(g2,T) ]*[f_\Gamma(g1,T) ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= qdp_gcoeff(gamma_idx_2,true,false)*qdp_gcoeff(gamma_idx_1,true,false);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_trans_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(Fud);
	  sdiag0_trset0_scfmat_prod.pl(F1);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.ccl(-1);
	  sdiag0_trset0_scfmat_prod.pr(F0);
	  sdiag0_trset0_scfmat_prod.pr(Fud);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_x_src_y_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }
    corrfunc.write(fp);
  }


  {
    /*<<(\bar u,s)*(\bar d,s')>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_U_S_D_SPRIME " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_U_S_D_SPRIME %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, false);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, false);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
  
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_trans_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
      prop_sprimes_snk_x_src_y_trans_scfmat.transpose();
  
      /*Starting contraction 0*/
      /*[{\rm tr}_{scf,0}\left\{\Gamma[g2] C F_0 F_\updownarrow \mathcal{G}^{[s^\prime/s] T}_{x,y} F_0 F_\updownarrow \Gamma[g1] C \mathcal{G}^{[u/d] }_{x,y}\right\}_{0}  ]*[f_\Gamma(g1,T) ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= qdp_gcoeff(gamma_idx_1,true,false);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_trans_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(Fud);
	  sdiag0_trset0_scfmat_prod.pl(F0);
	  sdiag0_trset0_scfmat_prod.ccl(-1);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.pr(F0);
	  sdiag0_trset0_scfmat_prod.pr(Fud);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_x_src_y_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }


    corrfunc.write(fp);
  }


  {
    /*<<(\bar u,s')*(\bar s',u)>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_U_SPRIME_SPRIME_U " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_U_SPRIME_SPRIME_U %d %d contraction\n",gamma_idx_1,gamma_idx_2);

    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, false);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, true);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
  
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_hconj_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
      prop_sprimes_snk_x_src_y_hconj_scfmat.hconj();
  
      /*Starting contraction 0*/
      /*[-1 ]*[{\rm tr}_{scf,0}\left\{C \Gamma[g2] \gamma^5 C F_1 \mathcal{G}^{[s^\prime/s] \dagger}_{x,y} F_1 \gamma^5 C \Gamma[g1] C \mathcal{G}^{[u/d] }_{x,y}\right\}_{0}  ]*[f_\Gamma(g2,T) ]*[f_\Gamma(g1,T) ]*/
  
      {
	Rcomplex contraction(-1 , 0);
	contraction *= qdp_gcoeff(gamma_idx_2,true,false)*qdp_gcoeff(gamma_idx_1,true,false);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag1(1.0);
	{
	  SpinColorFlavorMatrix sdiag1_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_hconj_scfmat);
	  sdiag1_trset0_scfmat_prod.pl(F1);
	  sdiag1_trset0_scfmat_prod.ccl(-1);
	  sdiag1_trset0_scfmat_prod.gl(-5);
	  qdp_gl(sdiag1_trset0_scfmat_prod,gamma_idx_2);
	  sdiag1_trset0_scfmat_prod.ccl(-1);
	  sdiag1_trset0_scfmat_prod.pr(F1);
	  sdiag1_trset0_scfmat_prod.gr(-5);
	  sdiag1_trset0_scfmat_prod.ccr(1);
	  qdp_gr(sdiag1_trset0_scfmat_prod,gamma_idx_1);
	  sdiag1_trset0_scfmat_prod.ccr(1);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_x_src_y_scfmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_scfmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	contraction *= result_subdiag1;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }

    corrfunc.write(fp);
  }



  {
    /*<<(\bar u,s')*(\bar d,s)>>*/
    /*Require a "CorrelationFunction &corrfunc" with option "CorrelationFunction::THREADED"*/
    std::ostringstream os; os << "HL_MESON_U_SPRIME_D_S " << gamma_idx_1 << " " << gamma_idx_2;
    CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

    if(UniqueID()==0) printf("Doing HL_MESON_U_SPRIME_D_S %d %d contraction\n",gamma_idx_1,gamma_idx_2);
    /*Require propagator "PropagatorContainer &prop_src_y_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
    /*Require propagator "PropagatorContainer &prop_src_y_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{x,y} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/

    /*Fourier transform on sink index x*/
    /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
    {
      QuarkMomCombination momcomb;
      momcomb.add_prop(prop_src_y_u_d_eitherflav_pcon, false);
      momcomb.add_prop(prop_src_y_sprime_s_eitherflav_pcon, true);
      bool desired_mom_available(momcomb.contains(desired_mom_x));
      /*Create an appropriate error message if !desired_mom_available*/
      if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_gparity",desired_mom_x,momcomb);
    }

    corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
    for(int x=0;x<GJP.VolNodeSites();x++){
      int x_pos_vec[4];
      global_coord(x,x_pos_vec);
  
      /*Get all SpinColorFlavorMatrices needed*/
      SpinColorFlavorMatrix prop_sprimes_snk_x_src_y_hconj_scfmat(prop_src_y_sprime_s_eitherflav_pcon , AlgLattice(), x);
      prop_sprimes_snk_x_src_y_hconj_scfmat.hconj();
  
      SpinColorFlavorMatrix prop_ud_snk_x_src_y_scfmat(prop_src_y_u_d_eitherflav_pcon , AlgLattice(), x);
  
      /*Starting contraction 0*/
      /*[{\rm tr}_{scf,0}\left\{\Gamma[g2] \gamma^5 F_0 \mathcal{G}^{[s^\prime/s] \dagger}_{x,y} F_1 \gamma^5 C \Gamma[g1] C \mathcal{G}^{[u/d] }_{x,y}\right\}_{0}  ]*[f_\Gamma(g1,T) ]*/
  
      {
	Rcomplex contraction(1 , 0);
	contraction *= qdp_gcoeff(gamma_idx_1,true,false);
	contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_x_src_y_hconj_scfmat);
	  sdiag0_trset0_scfmat_prod.pl(F0);
	  sdiag0_trset0_scfmat_prod.gl(-5);
	  qdp_gl(sdiag0_trset0_scfmat_prod,gamma_idx_2);
	  sdiag0_trset0_scfmat_prod.pr(F1);
	  sdiag0_trset0_scfmat_prod.gr(-5);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  qdp_gr(sdiag0_trset0_scfmat_prod,gamma_idx_1);
	  sdiag0_trset0_scfmat_prod.ccr(1);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_x_src_y_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	contraction *= result_subdiag0;
    
    
	corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
      }
    }


    corrfunc.write(fp);
  }
}


void AlgGparityContract::meson_HL_std(PropagatorContainer &prop_H,PropagatorContainer &prop_L, const int* sink_mom, const int &gamma_idx_1, const int &gamma_idx_2, FILE *fp){
  /*Mesons comprising $ \bar u $ and $ s$*/
  /*Require a "CorrelationFunction &corrfunc"*/
  std::ostringstream os; os << "HL_MESON_L_H " << gamma_idx_1 << " " << gamma_idx_2;
  CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

  if(UniqueID()==0) printf("Doing HL_MESON_L_H %d %d contraction\n",gamma_idx_1,gamma_idx_2);

  /*Require propagator "PropagatorContainer &prop_src_y_0_pcon corresponding to \mathcal{G}^{(0)}_{x,y}*/
  /*Require propagator "PropagatorContainer &prop_src_y_s_pcon corresponding to \mathcal{G}^{(s)}_{x,y}*/
  PropagatorContainer &prop_src_y_0_pcon = prop_L;
  PropagatorContainer &prop_src_y_s_pcon = prop_H;

  /*Fourier transform on sink index x*/
  /*Require a 3-component array 'desired_mom_x' representing the required momentum at this sink position*/
  const int *desired_mom_x = sink_mom;
  {
    QuarkMomCombination momcomb;
    momcomb.add_prop(prop_src_y_0_pcon, true);
    momcomb.add_prop(prop_src_y_s_pcon, false);
    bool desired_mom_available(momcomb.contains(desired_mom_x));
    /*Create an appropriate error message if !desired_mom_available*/
    if(!desired_mom_available) MomCombError("AlgGparityContract","meson_HL_std",desired_mom_x,momcomb);
  }


  corrfunc.setNcontractions(1);
#pragma omp parallel for default(shared)
  for(int x=0;x<GJP.VolNodeSites();x++){
    int x_pos_vec[4];
    global_coord(x,x_pos_vec);
  
    /*Get all WilsonMatrices needed*/
    WilsonMatrix& prop_snk_x_s_src_y_s_wmat = prop_src_y_s_pcon.getProp(AlgLattice()).SiteMatrix(x,0);
  
    WilsonMatrix prop_snk_x_0_src_y_0_hconj_wmat(prop_src_y_0_pcon.getProp(AlgLattice()).SiteMatrix(x,0));
    prop_snk_x_0_src_y_0_hconj_wmat.hconj();
  
    /*Starting contraction 0*/
    /*[-1 ]*[{\rm tr}_{sc,0}\left\{\gamma^5 S_1 \mathcal{G}^{(s)}_{x,y} S_2 \gamma^5 \mathcal{G}^{(0) \dagger}_{x,y}\right\}_{0}  ]*/
  
    {
      Rcomplex contraction(-1 , 0);
      contraction *= sink_phasefac(desired_mom_x,x_pos_vec,false);
    
      Rcomplex result_subdiag1(1.0);
      {
	WilsonMatrix sdiag1_trset0_wmat_prod(prop_snk_x_s_src_y_s_wmat);
	//sdiag1_trset0_wmat_prod = S_1 * sdiag1_trset0_wmat_prod;
	qdp_gl(sdiag1_trset0_wmat_prod,gamma_idx_1);

	sdiag1_trset0_wmat_prod.gl(-5);
	//sdiag1_trset0_wmat_prod *= S_2;
	qdp_gr(sdiag1_trset0_wmat_prod,gamma_idx_2);

	sdiag1_trset0_wmat_prod.gr(-5);
	sdiag1_trset0_wmat_prod *= prop_snk_x_0_src_y_0_hconj_wmat;
	Rcomplex sdiag1_trset0_cmplx;
	sdiag1_trset0_cmplx = sdiag1_trset0_wmat_prod.Trace();
	result_subdiag1 *= sdiag1_trset0_cmplx;
      }
      contraction *= result_subdiag1;
    
    
      corrfunc(omp_get_thread_num(),0,x_pos_vec[3]) += contraction;
    }
  }
  corrfunc.write(fp);
}

void AlgGparityContract::contract_HL_mesons(const ContractionTypeHLMesons &args, const int &conf_idx){
  std::ostringstream file; file << args.file << "." << conf_idx;

  FILE *fp;
  if ((fp = Fopen(file.str().c_str(), "w")) == NULL) {
    ERR.FileW("CorrelationFunction","write(const char *file)",file.str().c_str());
  }

  PropagatorContainer &prop_H = PropManager::getProp(args.prop_H);
  PropagatorContainer &prop_L = PropManager::getProp(args.prop_L);

  //loop through HL meson correlation functions
  if(GJP.Gparity()){
    for(int g1=0;g1<16;g1++){
      for(int g2=0;g2<16;g2++){
	meson_HL_gparity(prop_H,prop_L, args.sink_mom, g1, g2, fp);
      }
    }
  }else{
    for(int g1=0;g1<16;g1++){
      for(int g2=0;g2<16;g2++){
  	meson_HL_std(prop_H,prop_L, args.sink_mom, g1, g2, fp);
      }
    }
  }
  Fclose(fp);
}

void AlgGparityContract::contract_OVVpAA_gparity(const ContractionTypeOVVpAA &args, const int &conf_idx){
  /*Require a "CorrelationFunction &corrfunc"*/
  /*Require propagator "PropagatorContainer &prop_src_z_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{y,z} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
  /*Require propagator "PropagatorContainer &prop_src_z_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{y,z} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
  /*Require propagator "PropagatorContainer &prop_src_x_u_d_eitherflav_pcon corresponding to \mathcal{G}^{[u/d] }_{y,x} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
  /*Require propagator "PropagatorContainer &prop_src_x_sprime_s_eitherflav_pcon corresponding to \mathcal{G}^{[s^\prime/s] }_{y,x} with source of either flavour (full prop matrix is generated using single flavour source). Source must be real.*/
  std::ostringstream os; os << "O_VV_P_AA";
  CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

  if(UniqueID()==0) printf("Doing OVVpAA contractions with G-parity BCs\n");

  PropagatorContainer &prop_src_z_u_d_eitherflav_pcon = PropManager::getProp(args.prop_L_t1);
  PropagatorContainer &prop_src_z_sprime_s_eitherflav_pcon = PropManager::getProp(args.prop_H_t1);
  PropagatorContainer &prop_src_x_u_d_eitherflav_pcon = PropManager::getProp(args.prop_L_t0);
  PropagatorContainer &prop_src_x_sprime_s_eitherflav_pcon = PropManager::getProp(args.prop_H_t0);

  /*Fourier transform on sink index y*/
  /*Require a 3-component array 'desired_mom_y' representing the required momentum at this sink position*/
  int desired_mom_y[] = {0,0,0};

  {
    QuarkMomCombination momcomb;
    momcomb.add_prop(prop_src_z_u_d_eitherflav_pcon, false);
    momcomb.add_prop(prop_src_z_sprime_s_eitherflav_pcon, true);
    momcomb.add_prop(prop_src_x_u_d_eitherflav_pcon, false);
    momcomb.add_prop(prop_src_x_sprime_s_eitherflav_pcon, true);
    bool desired_mom_available(momcomb.contains(desired_mom_y));
    /*Create an appropriate error message if !desired_mom_available*/
    if(!desired_mom_available) MomCombError("AlgGparityContract","contract_OVVpAA_gparity",desired_mom_y,momcomb);
  }

  corrfunc.setNcontractions(4);

#pragma omp parallel for default(shared)
  for(int y=0;y<GJP.VolNodeSites();y++){
    int y_pos_vec[4];
    global_coord(y,y_pos_vec);
  
    /*Get all SpinColorFlavorMatrices needed*/
    SpinColorFlavorMatrix prop_sprimes_snk_y_src_x_hconj_scfmat(prop_src_x_sprime_s_eitherflav_pcon , AlgLattice(), y);
    prop_sprimes_snk_y_src_x_hconj_scfmat.hconj();
  
    SpinColorFlavorMatrix prop_sprimes_snk_y_src_z_hconj_scfmat(prop_src_z_sprime_s_eitherflav_pcon , AlgLattice(), y);
    prop_sprimes_snk_y_src_z_hconj_scfmat.hconj();
  
    SpinColorFlavorMatrix prop_ud_snk_y_src_z_scfmat(prop_src_z_u_d_eitherflav_pcon , AlgLattice(), y);
  
    SpinColorFlavorMatrix prop_ud_snk_y_src_x_scfmat(prop_src_x_u_d_eitherflav_pcon , AlgLattice(), y);
  
    /*Starting contraction 0*/
    /*[{\rm tr}_{scf,0}\left\{\mathcal{G}^{[s^\prime/s] \dagger}_{y,z} F_0 \gamma^\mu \gamma^5 \mathcal{G}^{[u/d] }_{y,z}\right\}_{0}  ]*[{\rm tr}_{scf,0}\left\{\mathcal{G}^{[s^\prime/s] \dagger}_{y,x} F_0 \gamma^\mu \gamma^5 \mathcal{G}^{[u/d] }_{y,x}\right\}_{0}  ]*[0.5 ]*/
  
    {
      Rcomplex contraction(0.5 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_y_src_z_hconj_scfmat);
	  sdiag0_trset0_scfmat_prod.pr(F0);
	  sdiag0_trset0_scfmat_prod.gr(gidx0);
	  sdiag0_trset0_scfmat_prod.gr(-5);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_y_src_z_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	g_prod *= result_subdiag0;
      
	Rcomplex result_subdiag1(1.0);
	{
	  SpinColorFlavorMatrix sdiag1_trset0_scfmat_prod(prop_sprimes_snk_y_src_x_hconj_scfmat);
	  sdiag1_trset0_scfmat_prod.pr(F0);
	  sdiag1_trset0_scfmat_prod.gr(gidx0);
	  sdiag1_trset0_scfmat_prod.gr(-5);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_y_src_x_scfmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_scfmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),0,y_pos_vec[3]) += contraction;
    }
    /*Starting contraction 1*/
    /*[-0.5 ]*[{\rm tr}_{scf,0}\left\{\mathcal{G}^{[s^\prime/s] \dagger}_{y,x} F_0 \gamma^\mu \gamma^5 \mathcal{G}^{[u/d] }_{y,z} \mathcal{G}^{[s^\prime/s] \dagger}_{y,z} F_0 \gamma^\mu \gamma^5 \mathcal{G}^{[u/d] }_{y,x}\right\}_{0}  ]*/
  
    {
      Rcomplex contraction(-0.5 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag1(1.0);
	{
	  SpinColorFlavorMatrix sdiag1_trset0_scfmat_prod(prop_sprimes_snk_y_src_x_hconj_scfmat);
	  sdiag1_trset0_scfmat_prod.pr(F0);
	  sdiag1_trset0_scfmat_prod.gr(gidx0);
	  sdiag1_trset0_scfmat_prod.gr(-5);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_y_src_z_scfmat;
	  sdiag1_trset0_scfmat_prod *= prop_sprimes_snk_y_src_z_hconj_scfmat;
	  sdiag1_trset0_scfmat_prod.pr(F0);
	  sdiag1_trset0_scfmat_prod.gr(gidx0);
	  sdiag1_trset0_scfmat_prod.gr(-5);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_y_src_x_scfmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_scfmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),1,y_pos_vec[3]) += contraction;
    }
    /*Starting contraction 2*/
    /*[{\rm tr}_{scf,0}\left\{\mathcal{G}^{[s^\prime/s] \dagger}_{y,z} F_0 \gamma^\mu \mathcal{G}^{[u/d] }_{y,z}\right\}_{0}  ]*[{\rm tr}_{scf,0}\left\{\mathcal{G}^{[s^\prime/s] \dagger}_{y,x} F_0 \gamma^\mu \mathcal{G}^{[u/d] }_{y,x}\right\}_{0}  ]*[0.5 ]*/
  
    {
      Rcomplex contraction(0.5 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag0(1.0);
	{
	  SpinColorFlavorMatrix sdiag0_trset0_scfmat_prod(prop_sprimes_snk_y_src_z_hconj_scfmat);
	  sdiag0_trset0_scfmat_prod.pr(F0);
	  sdiag0_trset0_scfmat_prod.gr(gidx0);
	  sdiag0_trset0_scfmat_prod *= prop_ud_snk_y_src_z_scfmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_scfmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	g_prod *= result_subdiag0;
      
	Rcomplex result_subdiag1(1.0);
	{
	  SpinColorFlavorMatrix sdiag1_trset0_scfmat_prod(prop_sprimes_snk_y_src_x_hconj_scfmat);
	  sdiag1_trset0_scfmat_prod.pr(F0);
	  sdiag1_trset0_scfmat_prod.gr(gidx0);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_y_src_x_scfmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_scfmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),2,y_pos_vec[3]) += contraction;
    }
    /*Starting contraction 3*/
    /*[-0.5 ]*[{\rm tr}_{scf,0}\left\{\mathcal{G}^{[s^\prime/s] \dagger}_{y,x} F_0 \gamma^\mu \mathcal{G}^{[u/d] }_{y,z} \mathcal{G}^{[s^\prime/s] \dagger}_{y,z} F_0 \gamma^\mu \mathcal{G}^{[u/d] }_{y,x}\right\}_{0}  ]*/
  
    {
      Rcomplex contraction(-0.5 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag1(1.0);
	{
	  SpinColorFlavorMatrix sdiag1_trset0_scfmat_prod(prop_sprimes_snk_y_src_x_hconj_scfmat);
	  sdiag1_trset0_scfmat_prod.pr(F0);
	  sdiag1_trset0_scfmat_prod.gr(gidx0);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_y_src_z_scfmat;
	  sdiag1_trset0_scfmat_prod *= prop_sprimes_snk_y_src_z_hconj_scfmat;
	  sdiag1_trset0_scfmat_prod.pr(F0);
	  sdiag1_trset0_scfmat_prod.gr(gidx0);
	  sdiag1_trset0_scfmat_prod *= prop_ud_snk_y_src_x_scfmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_scfmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),3,y_pos_vec[3]) += contraction;
    }
  }

  std::ostringstream file; file << args.file << "." << conf_idx;
  corrfunc.write(file.str().c_str());
}

// Sums a quantity over the whole lattice; 99 is a magic number.
void lat_sum(Float *float_p, int block) {slice_sum(float_p,block,99);}

// Assigns a pointer of type TYPE to PTR with enough memory allocated
// for SIZE objects of that type. Prints error on fail.
#define SMALLOC(PTR,TYPE,SIZE) PTR = (TYPE*)smalloc(SIZE*sizeof(TYPE)); \
  if (PTR == 0) ERR.Pointer("",fname,#PTR); \
  VRB.Smalloc("",fname,#PTR,PTR,SIZE);

enum TraceType {
  TR = 0,    // one full trace  (spin and color)
  TRTR,      // two full traces 
  TR_MX,     // one spin trace  (two color traces)
  TRTR_MX    // two spin traces (one color trace)
};
typedef enum TraceType TraceType;

// figure eight diagrams (labelled F8 (B_K style) or F8l (K->pi style))
//----------------------------------------------------------------
static void figure8(QPropW& q_str, QPropW& q_src,
		    QPropW& q_snk1, QPropW& q_snk2,
		    const int &conf_idx) {
  if(!UniqueID()){ printf("Doing figure 8 test\n"); }

  //strange then light, source then sink
  int is_light = 0;
  
  char *fname = "figure8()";
  VRB.Func("",fname);

  std::ostringstream file; file << "figure8_test." << conf_idx;

  FILE *fp;
  if ((fp = Fopen(file.str().c_str(), "w")) == NULL) {
    ERR.FileW("","figure8(...)",file.str().c_str());
  }

  int do_susy = 0;

  int gat, trt;
  int t_src = q_str.SourceTime();
  int t_snk = q_snk1.SourceTime();
  int t, time_size = GJP.Tnodes()*GJP.TnodeSites();
  Rcomplex *oo[3][4], *pp[3][4];
  for (gat=0;gat<3;gat++) for (trt=0;trt<4;trt++) {
	SMALLOC(oo[gat][trt],Rcomplex,time_size);
	SMALLOC(pp[gat][trt],Rcomplex,time_size);
  }
  WilsonMatrix tmp_str, tmp_snk; // strange, sink
  SpinMatrix spn_src, spn_snk;   // source, sink
  Matrix     col_src, col_snk;

  for (gat=0;gat<3;gat++) for (trt=0;trt<4;trt++) for (t=0;t<time_size;t++) 
	oo[gat][trt][t] = pp[gat][trt][t] = 0.0;

  int shift_t = GJP.TnodeCoor()*GJP.TnodeSites();
  int vol = GJP.VolNodeSites()/GJP.TnodeSites();
  for (int mu=-1; mu<4; mu++) {
	for (int nu=-1; nu<4; nu++) {
	  if (mu<3 && mu>-1 && nu<3 && nu>-1 && mu<=nu) continue;
	  gat = (nu<0?0:(mu<0?1:2)); //if nu<0 then gat = 0,  else if nu>=0 and mu<0 gat = 1, otherwise gat=2
	  //non-susy version uses gat=1, i.e. nu=0,1,2,3 and mu=-1
	  if (!do_susy && gat!=1) continue;
	  for (int i=0; i<GJP.VolNodeSites(); i++) {
		t = i/vol + shift_t;
		tmp_str = q_str[i];
		//tmp_str.hconj().gr(-5).gr(mu).gr(nu);  // pion-antiquark-gammas
		tmp_str.hconj();
		tmp_str.gr(-5).gr(mu).gr(nu);  // pion-antiquark-gammas
		tmp_snk = q_snk1[i];
		//tmp_snk.hconj().gr(-5).gr(mu).gr(nu); // pion-antiquark-gammas
		tmp_snk.hconj();
		tmp_snk.gr(-5).gr(mu).gr(nu); // pion-antiquark-gammas
		oo[gat][TR][t] += Trace(q_snk2[i]*tmp_snk,q_src[i]*tmp_str);
		oo[gat][TRTR][t] += Trace(q_snk2[i],tmp_snk)*Trace(q_src[i],tmp_str);
		spn_src = ColorTrace(q_src[i],tmp_str);
		spn_snk = ColorTrace(q_snk2[i],tmp_snk);
		oo[gat][TR_MX][t] += Tr(spn_snk,spn_src);
		col_src = SpinTrace(q_src[i],tmp_str);
		col_snk = SpinTrace(q_snk2[i],tmp_snk);
		oo[gat][TRTR_MX][t] += Tr(col_snk,col_src);
		tmp_str.gr(-5);		          // parity swap
		tmp_snk.gr(-5);
		pp[gat][TR][t] += Trace(q_snk2[i]*tmp_snk,q_src[i]*tmp_str);
		pp[gat][TRTR][t] += Trace(q_snk2[i],tmp_snk)*Trace(q_src[i],tmp_str);
		spn_src = ColorTrace(q_src[i],tmp_str);
		spn_snk = ColorTrace(q_snk2[i],tmp_snk);
		pp[gat][TR_MX][t] += Tr(spn_snk,spn_src);
		col_src = SpinTrace(q_src[i],tmp_str);
		col_snk = SpinTrace(q_snk2[i],tmp_snk);
		pp[gat][TRTR_MX][t] += Tr(col_snk,col_src);
	  }
    }
  }

  // Global sums and Output the correlators
  for (gat=0;gat<3;gat++) for (trt=0;trt<4;trt++) for (t=0;t<time_size;t++) {
    lat_sum((Float*)&oo[gat][trt][t], 2);
    lat_sum((Float*)&pp[gat][trt][t], 2);
  }
  // Print out results
  //----------------------------------------------------------------
  char gam [3][5];
  char gam2[3][5];  
  char tra [4][10];
  char tag[3][5] = {"SP","VA","TT"};
  for (int i=0;i<3;i++) strcpy(gam[i],tag[i]);
  char tag2[3][5] = {"SS","VV","TT"};
  for (int i=0;i<3;i++) strcpy(gam2[i],tag2[i]);
  char tag3[4][10] ={"TR_","TRTR_","TR_MX_","TRTR_MX_"};
  for (int i=0;i<4;i++) strcpy(tra[i],tag3[i]);

  for (gat=0;gat<3;gat++) {
	if (!do_susy && gat!=1) continue;
	for (trt=0;trt<4;trt++) for (t=0;t<time_size;t++){
	  Fprintf(fp,"F8%s%s%s %d %d %d  %.16e %.16e\t%.16e %.16e\n",
		  (is_light?"l":"\0"), tra[trt], gam2[gat],
		  t_src, t, t_snk,
		  oo[gat][trt][t].real(), oo[gat][trt][t].imag(),
		  pp[gat][trt][t].real(), pp[gat][trt][t].imag());
	  printf("F8%s%s%s %d %d %d  %.16e %.16e\t%.16e %.16e\n",
		 (is_light?"l":"\0"), tra[trt], gam2[gat],
		 t_src, t, t_snk,
		 oo[gat][trt][t].real(), oo[gat][trt][t].imag(),
		 pp[gat][trt][t].real(), pp[gat][trt][t].imag());  
	  }


  }
  for (gat=0;gat<3;gat++) for (trt=0;trt<4;trt++) {
	sfree(oo[gat][trt]);
	sfree(pp[gat][trt]);
  }
  Fclose(fp);
}




void AlgGparityContract::contract_OVVpAA_std(const ContractionTypeOVVpAA &args, const int &conf_idx){

  /*Require a "CorrelationFunction &corrfunc"*/
  std::ostringstream os; os << "O_VV_P_AA";
  CorrelationFunction corrfunc(os.str().c_str(),CorrelationFunction::THREADED);

  if(UniqueID()==0) printf("Doing OVVpAA contractions\n");

  /*Require propagator "PropagatorContainer &prop_src_z_0_pcon corresponding to \mathcal{G}^{(0)}_{y,z}*/
  /*Require propagator "PropagatorContainer &prop_src_z_s_pcon corresponding to \mathcal{G}^{(s)}_{y,z}*/
  /*Require propagator "PropagatorContainer &prop_src_x_0_pcon corresponding to \mathcal{G}^{(0)}_{y,x}*/
  /*Require propagator "PropagatorContainer &prop_src_x_s_pcon corresponding to \mathcal{G}^{(s)}_{y,x}*/

  PropagatorContainer &prop_src_z_0_pcon = PropManager::getProp(args.prop_L_t1);
  PropagatorContainer &prop_src_z_s_pcon = PropManager::getProp(args.prop_H_t1);
  PropagatorContainer &prop_src_x_0_pcon = PropManager::getProp(args.prop_L_t0);
  PropagatorContainer &prop_src_x_s_pcon = PropManager::getProp(args.prop_H_t0);

  /*Fourier transform on sink index y*/
  /*Require a 3-component array 'desired_mom_y' representing the required momentum at this sink position*/
  int desired_mom_y[] = {0,0,0};

  {
    QuarkMomCombination momcomb;
    momcomb.add_prop(prop_src_z_0_pcon, false);
    momcomb.add_prop(prop_src_z_s_pcon, true);
    momcomb.add_prop(prop_src_x_0_pcon, false);
    momcomb.add_prop(prop_src_x_s_pcon, true);
    bool desired_mom_available(momcomb.contains(desired_mom_y));
    /*Create an appropriate error message if !desired_mom_available*/
    if(!desired_mom_available) MomCombError("AlgGparityContract","contract_OVVpAA_std",desired_mom_y,momcomb);
  }

  corrfunc.setNcontractions(4);
#pragma omp parallel for default(shared)
  for(int y=0;y<GJP.VolNodeSites();y++){
    int y_pos_vec[4];
    global_coord(y,y_pos_vec);
  
    /*Get all WilsonMatrices needed*/
    WilsonMatrix prop_snk_y_s_src_z_s_hconj_wmat(prop_src_z_s_pcon.getProp(AlgLattice()).SiteMatrix(y,0));
    prop_snk_y_s_src_z_s_hconj_wmat.hconj();
  
    WilsonMatrix& prop_snk_y_0_src_z_0_wmat = prop_src_z_0_pcon.getProp(AlgLattice()).SiteMatrix(y,0);
  
    WilsonMatrix& prop_snk_y_0_src_x_0_wmat = prop_src_x_0_pcon.getProp(AlgLattice()).SiteMatrix(y,0);
  
    WilsonMatrix prop_snk_y_s_src_x_s_hconj_wmat(prop_src_x_s_pcon.getProp(AlgLattice()).SiteMatrix(y,0));
    prop_snk_y_s_src_x_s_hconj_wmat.hconj();
  
    /*Starting contraction 0*/
    /*[{\rm tr}_{sc,0}\left\{\mathcal{G}^{(s) \dagger}_{y,z} \gamma^\mu \gamma^5 \mathcal{G}^{(0)}_{y,z}\right\}_{0}  ]*[{\rm tr}_{sc,0}\left\{\mathcal{G}^{(s) \dagger}_{y,x} \gamma^\mu \gamma^5 \mathcal{G}^{(0)}_{y,x}\right\}_{0}  ]*[2 ]*/
  
    {
      Rcomplex contraction(2 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag0(1.0);
	{
	  WilsonMatrix sdiag0_trset0_wmat_prod(prop_snk_y_s_src_z_s_hconj_wmat);
	  sdiag0_trset0_wmat_prod.gr(gidx0);
	  sdiag0_trset0_wmat_prod.gr(-5);
	  sdiag0_trset0_wmat_prod *= prop_snk_y_0_src_z_0_wmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_wmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	g_prod *= result_subdiag0;
      
	Rcomplex result_subdiag1(1.0);
	{
	  WilsonMatrix sdiag1_trset0_wmat_prod(prop_snk_y_s_src_x_s_hconj_wmat);
	  sdiag1_trset0_wmat_prod.gr(gidx0);
	  sdiag1_trset0_wmat_prod.gr(-5);
	  sdiag1_trset0_wmat_prod *= prop_snk_y_0_src_x_0_wmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_wmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),0,y_pos_vec[3]) += contraction;
    }
    /*Starting contraction 1*/
    /*[-2 ]*[{\rm tr}_{sc,0}\left\{\mathcal{G}^{(s) \dagger}_{y,x} \gamma^\mu \gamma^5 \mathcal{G}^{(0)}_{y,z} \mathcal{G}^{(s) \dagger}_{y,z} \gamma^\mu \gamma^5 \mathcal{G}^{(0)}_{y,x}\right\}_{0}  ]*/
  
    {
      Rcomplex contraction(-2 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag1(1.0);
	{
	  WilsonMatrix sdiag1_trset0_wmat_prod(prop_snk_y_s_src_x_s_hconj_wmat);
	  sdiag1_trset0_wmat_prod.gr(gidx0);
	  sdiag1_trset0_wmat_prod.gr(-5);
	  sdiag1_trset0_wmat_prod *= prop_snk_y_0_src_z_0_wmat;
	  sdiag1_trset0_wmat_prod *= prop_snk_y_s_src_z_s_hconj_wmat;
	  sdiag1_trset0_wmat_prod.gr(gidx0);
	  sdiag1_trset0_wmat_prod.gr(-5);
	  sdiag1_trset0_wmat_prod *= prop_snk_y_0_src_x_0_wmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_wmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),1,y_pos_vec[3]) += contraction;
    }
    /*Starting contraction 2*/
    /*[{\rm tr}_{sc,0}\left\{\mathcal{G}^{(s) \dagger}_{y,z} \gamma^\mu \mathcal{G}^{(0)}_{y,z}\right\}_{0}  ]*[{\rm tr}_{sc,0}\left\{\mathcal{G}^{(s) \dagger}_{y,x} \gamma^\mu \mathcal{G}^{(0)}_{y,x}\right\}_{0}  ]*[2 ]*/
  
    {
      Rcomplex contraction(2 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag0(1.0);
	{
	  WilsonMatrix sdiag0_trset0_wmat_prod(prop_snk_y_s_src_z_s_hconj_wmat);
	  sdiag0_trset0_wmat_prod.gr(gidx0);
	  sdiag0_trset0_wmat_prod *= prop_snk_y_0_src_z_0_wmat;
	  Rcomplex sdiag0_trset0_cmplx;
	  sdiag0_trset0_cmplx = sdiag0_trset0_wmat_prod.Trace();
	  result_subdiag0 *= sdiag0_trset0_cmplx;
	}
	g_prod *= result_subdiag0;
      
	Rcomplex result_subdiag1(1.0);
	{
	  WilsonMatrix sdiag1_trset0_wmat_prod(prop_snk_y_s_src_x_s_hconj_wmat);
	  sdiag1_trset0_wmat_prod.gr(gidx0);
	  sdiag1_trset0_wmat_prod *= prop_snk_y_0_src_x_0_wmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_wmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),2,y_pos_vec[3]) += contraction;
    }
    /*Starting contraction 3*/
    /*[-2 ]*[{\rm tr}_{sc,0}\left\{\mathcal{G}^{(s) \dagger}_{y,x} \gamma^\mu \mathcal{G}^{(0)}_{y,z} \mathcal{G}^{(s) \dagger}_{y,z} \gamma^\mu \mathcal{G}^{(0)}_{y,x}\right\}_{0}  ]*/
  
    {
      Rcomplex contraction(-2 , 0);
      contraction *= sink_phasefac(desired_mom_y,y_pos_vec,false);
    
      /*Contraction contains implicit sum over 1 free gamma matrix indices*/
      Rcomplex gsum_result(0.0);
      for(int gidx0=0;gidx0<4;gidx0++){
	Rcomplex g_prod(1.0);
      
	Rcomplex result_subdiag1(1.0);
	{
	  WilsonMatrix sdiag1_trset0_wmat_prod(prop_snk_y_s_src_x_s_hconj_wmat);
	  sdiag1_trset0_wmat_prod.gr(gidx0);
	  sdiag1_trset0_wmat_prod *= prop_snk_y_0_src_z_0_wmat;
	  sdiag1_trset0_wmat_prod *= prop_snk_y_s_src_z_s_hconj_wmat;
	  sdiag1_trset0_wmat_prod.gr(gidx0);
	  sdiag1_trset0_wmat_prod *= prop_snk_y_0_src_x_0_wmat;
	  Rcomplex sdiag1_trset0_cmplx;
	  sdiag1_trset0_cmplx = sdiag1_trset0_wmat_prod.Trace();
	  result_subdiag1 *= sdiag1_trset0_cmplx;
	}
	g_prod *= result_subdiag1;
      
	gsum_result += g_prod;
      }
      contraction *= gsum_result;
    
      corrfunc(omp_get_thread_num(),3,y_pos_vec[3]) += contraction;
    }
  }

  std::ostringstream file; file << args.file << "." << conf_idx;
  corrfunc.write(file.str().c_str());

#if 0
  //compare against AlgThreePt
  figure8(prop_src_x_s_pcon.getProp(AlgLattice()), 
	  prop_src_x_0_pcon.getProp(AlgLattice()), 
	  prop_src_z_s_pcon.getProp(AlgLattice()), 
	  prop_src_z_0_pcon.getProp(AlgLattice()),
	  conf_idx);
#endif
}


void AlgGparityContract::contract_OVVpAA(const ContractionTypeOVVpAA &args, const int &conf_idx){
  if(GJP.Gparity()){
    return contract_OVVpAA_gparity(args,conf_idx);
  }else{
    return contract_OVVpAA_std(args,conf_idx);
  }
}


void AlgGparityContract::spectrum(const GparityMeasurement &measargs,const int &conf_idx){
  if(measargs.type == CONTRACTION_TYPE_LL_MESONS) contract_LL_mesons(measargs.GparityMeasurement_u.contraction_type_ll_mesons, conf_idx);
  else if(measargs.type == CONTRACTION_TYPE_HL_MESONS) contract_HL_mesons(measargs.GparityMeasurement_u.contraction_type_hl_mesons, conf_idx);
  else if(measargs.type == CONTRACTION_TYPE_O_VV_P_AA) contract_OVVpAA(measargs.GparityMeasurement_u.contraction_type_o_vv_p_aa, conf_idx);
  else ERR.General("AlgGparityContract","spectrum(...)","Invalid contraction type");
}








#if 0

  if(!GJP.Gparity()){
    //If Gparity boundary conditions are not used, produce standard correlation functions
    char out[500];

    if(strcmp(measargs.prop_1,measargs.prop_2)==0){
      sprintf(out,"%s pi^+/-",measargs.label_stub);
      CorrelationFunction c_pion_nogparity(out);
      
      bool allowed = pion_nogparity(measargs.prop_1,c_pion_nogparity);
      if(allowed){
	sprintf(out,"%s_pion_nogparity.%d.dat",measargs.file_stub,conf_idx);
	c_pion_nogparity.write(out);
      }
    }else{
      ERR.General(cname, "spectrum(....)", "Could not find any valid contractions for propagators %s and %s\n",measargs.prop_1,measargs.prop_2);
    }

  }else{
    //Decide which contractions are possible with the given quark flavours
    char ** prop_f[3]; for(int i=0;i<3;i++) prop_f[i] = new char*[2];
    int got_f[3] = {0,0,0}; //d, C\bar u^T, s

    PropagatorContainer &p1 = PropManager::getProp(measargs.prop_1);
    PropagatorContainer &p2 = PropManager::getProp(measargs.prop_2);
  
    int f1 = p1.flavor(); int f2 = p2.flavor();
    prop_f[f1][got_f[f1]++] = measargs.prop_1;
    prop_f[f2][got_f[f2]++] = measargs.prop_2;

    char out[500];

    if(got_f[0]==1 && got_f[1]==1){
      sprintf(out,"%s pi^+",measargs.label_stub);
      CorrelationFunction c_piplus(out);
      sprintf(out,"%s pi^-",measargs.label_stub);
      CorrelationFunction c_piminus(out);
      sprintf(out,"%s degenerate kaon",measargs.label_stub);
      CorrelationFunction c_degenkaon(out);

      if(UniqueID()==0) printf("Doing pi^+ contractions\n");
      bool pi_plus_done = pi_plus(prop_f[0][0],prop_f[1][0],c_piplus);
      if(UniqueID()==0) printf("Doing pi^- contractions\n");
      bool pi_minus_done = pi_minus(prop_f[0][0],prop_f[1][0],c_piminus);
      if(UniqueID()==0) printf("Doing degen kaon contractions\n");
      bool degen_kaon_done = degen_kaon(prop_f[0][0],prop_f[1][0],c_degenkaon);

      sprintf(out,"%s_pi_plus.%d.dat",measargs.file_stub,conf_idx);
      if(pi_plus_done) c_piplus.write(out);
      sprintf(out,"%s_pi_minus.%d.dat",measargs.file_stub,conf_idx);
      if(pi_minus_done) c_piminus.write(out);
      sprintf(out,"%s_degen_kaon.%d.dat",measargs.file_stub,conf_idx);
      if(degen_kaon_done) c_degenkaon.write(out);
    }else if(got_f[0]==2){
      //two quarks of same flavour
      sprintf(out,"%s pion with 2 f0 sources",measargs.label_stub);
      CorrelationFunction c_piplus(out);

      if(UniqueID()==0) printf("Doing pion contractions with 2 f0 props\n");
      bool pi_plus_f0src_done = pi_plus_f0src(prop_f[0][0], prop_f[0][1], c_piplus);

      sprintf(out,"%s_pion_2f0src.%d.dat",measargs.file_stub,conf_idx);
      if(pi_plus_f0src_done) c_piplus.write(out);

    }else{
      ERR.General(cname, "spectrum(....)", "Could not find any valid contractions for propagators %s and %s\n",measargs.prop_1,measargs.prop_2);
    }
    
  }

}

bool AlgGparityContract::pi_plus(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr){

#if 0
  PropagatorContainer &q_f1_pc = PropManager::getProp(q_f1_tag);
  PropagatorContainer &q_f0_pc = PropManager::getProp(q_f0_tag);

  //momentum
  int momphase[3] = {0,0,0}; //units of pi/2L
  sum_momphase(momphase,q_f1_pc);
  sum_momphase(momphase,q_f0_pc);
  //check sum is odd multiple of pi/L 
  for(int d=0;d<3;d++){
    if(GJP.Bc(d)==BND_CND_GPARITY && (momphase[d]%2!=0 || (momphase[d]/2)%2 !=1) ) {
      if(!UniqueID()) printf("%s::%s Total momentum is not an odd-multiple of pi/L\n",cname,"pi_plus");
      return false;
    }
  }
  if(!UniqueID()) printf("Sum of quark momentum pi/2L * {%d,%d,%d}\n",momphase[0],momphase[1],momphase[2]);

  QPropW &q_f1_qpw = q_f1_pc.getProp(AlgLattice());
  QPropW &q_f0_qpw = q_f0_pc.getProp(AlgLattice());

  corr.setNcontractions(2);
  Rcomplex tmpc1, tmpc2;
  Rcomplex tmp_cnum_0;
  WilsonMatrix tmp_scmat_0, tmp_scmat_1;
  
  //DEBUG, LOOK AT POSITION DEPENDENCE OF CORRELATION FUNCTION COMPONENTS (REMOVE ALL PHASE FACTORS INCLUDING THOSE FROM SOURCE - ONLY POSSIBLE FOR POINT SOURCES)
  //if(!UniqueID()) printf("Writing position dependence to piplus_pos.<NODE>\n");
  //FILE * ftxt = Fopen(ADD_ID,"piplus_pos","w");

  //Rcomplex src_phase_conj;
  bool point_sources(false);
  
  //PointSourceAttrArg *pt0, *pt1;
  //point_sources = q_f0_pc.getAttr(pt0) && q_f1_pc.getAttr(pt1);


  //   //find cconj of source phase factors
  //   int p[3];
  //   q_f0_pc.momentum(p);
  //   Rcomplex srcphse0 = sink_phasefac(p,pt0->pos); //src uses e^-ipx sink uses e^ipx
  //   q_f1_pc.momentum(p);
  //   Rcomplex srcphse1 = sink_phasefac(p,pt1->pos);

  //   src_phase_conj = srcphse0 * srcphse1;

  //   GenericPropAttrArg *generics;
  //   q_f0_pc.getAttr(generics);
  //   if(!UniqueID()) printf("Propagator %s cconj of source phase factor is (%e,%e)\n",
  // 			   generics->tag,srcphse0.real(),srcphse0.imag());
  //   q_f1_pc.getAttr(generics);
  //   if(!UniqueID()) printf("Propagator %s cconj of source phase factor is (%e,%e)\n",
  // 			   generics->tag,srcphse1.real(),srcphse1.imag());
  // }
  //DEBUG

  for(int i=0;i<GJP.VolNodeSites();i++){
    int t = i/spatial_vol + shift_t;

    Rcomplex snk_phasefac = sink_phasefac(momphase,i);

    //Doing contraction 1 of 2

    //\mathrm{tr}\left{\mathcal{G}^{(2,2)\,T}_{x ; y} \gamma^5 C \mathcal{G}^{(1,1)}_{x ; y} \gamma^5 C \right}
    tmpc1.real(1.0); tmpc1.imag(0.0);

    //Lowest level Spin-clr trace
    tmp_scmat_0 = q_f1_qpw.SiteMatrix(i,1);
    tmp_scmat_0.transpose();
    tmp_scmat_0.gr(-5).ccr(1);
    tmp_scmat_1 = q_f0_qpw.SiteMatrix(i,0);
    tmp_scmat_1.gr(-5).ccr(1);
    tmp_cnum_0 = Trace(tmp_scmat_0,tmp_scmat_1);

    tmpc1 *= tmp_cnum_0;

    //Finalising contraction 1
    corr(0,t) += tmpc1 * snk_phasefac;

    //Doing contraction 2 of 2

    //\mathrm{tr}\left{\mathcal{G}^{(1,2)\,T}_{x ; y} \gamma^5 C \mathcal{G}^{(2,1)}_{x ; y} \gamma^5 C \right}
    tmpc2.real(1.0); tmpc2.imag(0.0);

    //Lowest level Spin-clr trace
    tmp_scmat_0 = q_f1_qpw.SiteMatrix(i,0);
    tmp_scmat_0.transpose();
    tmp_scmat_0.gr(-5).ccr(1);
    tmp_scmat_1 = q_f0_qpw.SiteMatrix(i,1);
    tmp_scmat_1.gr(-5).ccr(1);
    tmp_cnum_0 = Trace(tmp_scmat_0,tmp_scmat_1);

    tmpc2 *= tmp_cnum_0;

    //Finalising contraction 2
    corr(1,t) += tmpc2 * snk_phasefac;

    //DEBUG, LOOK AT POSITION DEPENDENCE OF CORRELATION FUNCTION COMPONENTS (REMOVE ALL PHASE FACTORS INCLUDING THOSE FROM SOURCE - ONLY POSSIBLE FOR POINT SOURCES)
    // if(point_sources){
    //   Rcomplex tc1 = tmpc1;// * src_phase_conj;
    //   Rcomplex tc2 = tmpc2;// * src_phase_conj;

    //   int pos[4];
    //   global_coord(i,pos);      
    //   // printf("Piplus (%d,%d,%d,%d)\n",pos[0],pos[1],pos[2],pos[3]);

    //   //Check antisymmetry of correlation function
    //   Rcomplex sum = tc1 + tc2;

    //   Fprintf(ADD_ID,ftxt,"Piplus (%d,%d,%d,%d): c1 (%e %e) c2 (%e %e) c1+c2 (%e %e)\n",pos[0],pos[1],pos[2],pos[3],tc1.real(),tc1.imag(),tc2.real(),tc2.imag(),sum.real(),sum.imag()); fflush(stdout);
    // }
    //DEBUG
  }
  
  //DEBUG
  //Fclose(ADD_ID,ftxt);
  //DEBUG
#endif

  return true;
}


bool AlgGparityContract::pi_plus_f0src(const char *q0_f0_tag, const char *q1_f0_tag, CorrelationFunction &corr){
#if 0

  PropagatorContainer &q0_f0_pc = PropManager::getProp(q0_f0_tag);
  PropagatorContainer &q1_f0_pc = PropManager::getProp(q1_f0_tag);
  //[1]*[(\mathcal{G}^{(0) \dagger}_{y,x})_{#s1 #s17 #c2 #c5 }(\mathcal{G}^{(0)}_{y,x})_{#s17 #s1 #c5 #c2 }] + [-1]*[(\mathcal{G}^{(1,0) \dagger}_{y,x})_{#s21 #s4 #c2 #c5 }(\mathcal{G}^{(1,0) }_{y,x})_{#s4 #s21 #c5 #c2 }]
  //In order for the total momentum to be p, we need one propagator with momentum -p (to which the dagger is applied) and one with +p.

  //NOTE: I have multiplied everything by -1 to match the pi_plus function above, which did not use the factor of i in the definitions of the pion creation/annihilation operator

  //which guy has momentum p and which -p?
  int p0[3]; int p1[3];
  q0_f0_pc.momentum(p0);
  q1_f0_pc.momentum(p1);
  
  //check components are related by minus sign
  for(int i=0;i<3;i++){
    if(p0[i] != -p1[i]){
      if(!UniqueID()) printf("%s::%s Quark momenta are not equal and opposite\n",cname,"pi_plus_f0src");
      return false;
    }
  }

  //which has more positive momentum components
  int nplus0(0), nplus1(0);
  for(int i=0;i<3;i++){ 
    if(p0[i]>0) nplus0++;
    if(p1[i]>0) nplus1++;
  }
  PropagatorContainer *p = &q0_f0_pc;
  PropagatorContainer *m = &q1_f0_pc;
  if(nplus1 > nplus0){
    p = &q1_f0_pc;
    m = &q0_f0_pc;
  }
  PropagatorContainer &q_f0_plus_p_pc = *p;
  PropagatorContainer &q_f0_minus_p_pc = *m;

  //momentum
  int momphase[3] = {0,0,0}; //units of pi/2L
  sum_momphase(momphase,q_f0_plus_p_pc,false);
  sum_momphase(momphase,q_f0_minus_p_pc,true); //add -(-p)
  //check sum is odd multiple of pi/L 
  for(int d=0;d<3;d++){
    if(GJP.Bc(d)==BND_CND_GPARITY && (momphase[d]%2!=0 || (momphase[d]/2)%2 !=1) ){
      if(!UniqueID()) printf("%s::%s Total momentum is not an odd-multiple of pi/L\n",cname,"pi_minus");
      return false;
    }
  }
  if(!UniqueID()) printf("Sum of quark momentum pi/2L * {%d,%d,%d}\n",momphase[0],momphase[1],momphase[2]);

  QPropW &q_f0_plus_p_qpw = q_f0_plus_p_pc.getProp(AlgLattice());
  QPropW &q_f0_minus_p_qpw = q_f0_minus_p_pc.getProp(AlgLattice());

  corr.setNcontractions(2);
  Rcomplex tmpc;
  Rcomplex tmp_cnum_0;
  WilsonMatrix tmp_scmat_0, tmp_scmat_1;

  for(int i=0;i<GJP.VolNodeSites();i++){
    int t = i/spatial_vol + shift_t;

    Rcomplex snk_phasefac = sink_phasefac(momphase,i);

    //[1]*[(\mathcal{G}^{(0) \dagger}_{y,x})_{#s1 #s17 #c2 #c5 }(\mathcal{G}^{(0)}_{y,x})_{#s17 #s1 #c5 #c2 }]
    tmpc = snk_phasefac * Complex(-1.0,0.0); //see note above

    tmp_scmat_0 = q_f0_minus_p_qpw.SiteMatrix(i,0);
    tmp_scmat_0.hconj();
    tmp_cnum_0 = Trace(tmp_scmat_0, q_f0_plus_p_qpw.SiteMatrix(i,0) );

    tmpc *=tmp_cnum_0;

    corr(0,t) += tmpc;

    //[-1]*[(\mathcal{G}^{(1,0) \dagger}_{y,x})_{#s21 #s4 #c2 #c5 }(\mathcal{G}^{(1,0) }_{y,x})_{#s4 #s21 #c5 #c2 }]
    tmpc = snk_phasefac;
    
    tmp_scmat_0 = q_f0_minus_p_qpw.SiteMatrix(i,1);
    tmp_scmat_0.hconj();
    tmp_cnum_0 = Trace(tmp_scmat_0, q_f0_plus_p_qpw.SiteMatrix(i,1) );

    tmpc *=tmp_cnum_0;

    corr(1,t) += tmpc;
  }
  return true;
}





bool AlgGparityContract::pi_minus(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr){
  PropagatorContainer &q_f1_pc = PropManager::getProp(q_f1_tag);
  PropagatorContainer &q_f0_pc = PropManager::getProp(q_f0_tag);

  //momentum
  int momphase[3] = {0,0,0}; //units of pi/2L
  sum_momphase(momphase,q_f1_pc);
  sum_momphase(momphase,q_f0_pc);
  //check sum is odd multiple of pi/L 
  for(int d=0;d<3;d++){
    if(GJP.Bc(d)==BND_CND_GPARITY && (momphase[d]%2!=0 || (momphase[d]/2)%2 !=1) ){
      if(!UniqueID()) printf("%s::%s Total momentum is not an odd-multiple of pi/L\n",cname,"pi_minus");
      return false;
    }
  }
  if(!UniqueID()) printf("Sum of quark momentum pi/2L * {%d,%d,%d}\n",momphase[0],momphase[1],momphase[2]);

  QPropW &q_f1_qpw = q_f1_pc.getProp(AlgLattice());
  QPropW &q_f0_qpw = q_f0_pc.getProp(AlgLattice());

  corr.setNcontractions(2);
  Rcomplex tmpc;
  Rcomplex tmp_cnum_0;
  WilsonMatrix tmp_scmat_0, tmp_scmat_1;

  for(int i=0;i<GJP.VolNodeSites();i++){
    int t = i/spatial_vol + shift_t;

    Rcomplex snk_phasefac = sink_phasefac(momphase,i);

    //Doing contraction 1 of 2

    //\mathrm{tr}\left{\mathcal{G}^{(2,2)}_{x ; y} \gamma^5 C \mathcal{G}^{(1,1)\,T}_{x ; y} \gamma^5 C \right}
    tmpc = snk_phasefac;

    //Lowest level Spin-clr trace
    tmp_scmat_0 = q_f1_qpw.SiteMatrix(i,1);
    tmp_scmat_0.gr(-5).ccr(1);
    tmp_scmat_1 = q_f0_qpw.SiteMatrix(i,0);
    tmp_scmat_1.transpose();
    tmp_scmat_1.gr(-5).ccr(1);
    tmp_cnum_0 = Trace(tmp_scmat_0,tmp_scmat_1);

    tmpc *=tmp_cnum_0;

    //Finalising contraction 1
    corr(0,t) += tmpc;

    //Doing contraction 2 of 2

    //\mathrm{tr}\left{\mathcal{G}^{(2,1)}_{x ; y} \gamma^5 C \mathcal{G}^{(1,2)\,T}_{x ; y} \gamma^5 C \right}
    tmpc = snk_phasefac;

    //Lowest level Spin-clr trace
    tmp_scmat_0 = q_f0_qpw.SiteMatrix(i,1);
    tmp_scmat_0.gr(-5).ccr(1);
    tmp_scmat_1 = q_f1_qpw.SiteMatrix(i,0);
    tmp_scmat_1.transpose();
    tmp_scmat_1.gr(-5).ccr(1);
    tmp_cnum_0 = Trace(tmp_scmat_0,tmp_scmat_1);

    tmpc *=tmp_cnum_0;

    //Finalising contraction 2
    corr(1,t) += tmpc;
  }
  return true;

#endif
}

bool AlgGparityContract::degen_kaon(const char *q_f0_tag, const char *q_f1_tag, CorrelationFunction &corr){
#if 0

  //This is a G-parity even state that you get when you set the masses of the (s',s) doublet equal to the (u,d) doublet
  //and do the K^0 contractions. The contractions are exactly the same as for the pion, but with a relative minus sign.

  PropagatorContainer &q_f1_pc = PropManager::getProp(q_f1_tag);
  PropagatorContainer &q_f0_pc = PropManager::getProp(q_f0_tag);

  //momentum
  int momphase[3] = {0,0,0}; //units of pi/2L
  sum_momphase(momphase,q_f1_pc);
  sum_momphase(momphase,q_f0_pc);
  //check sum is odd multiple of pi/L 
  for(int d=0;d<3;d++){
    if(GJP.Bc(d)==BND_CND_GPARITY && momphase[d]!=0){
      if(!UniqueID()) printf("%s::%s Total momentum is not an zero\n",cname,"degen_kaon");
      return false;
    }
  }
  if(!UniqueID()) printf("Sum of quark momentum pi/2L * {%d,%d,%d}\n",momphase[0],momphase[1],momphase[2]);

  QPropW &q_f1_qpw = q_f1_pc.getProp(AlgLattice());
  QPropW &q_f0_qpw = q_f0_pc.getProp(AlgLattice());

  corr.setNcontractions(2);
  Rcomplex tmpc1, tmpc2;
  Rcomplex tmp_cnum_0;
  WilsonMatrix tmp_scmat_0, tmp_scmat_1;
  
  for(int i=0;i<GJP.VolNodeSites();i++){
    int t = i/spatial_vol + shift_t;

    Rcomplex snk_phasefac = sink_phasefac(momphase,i);

    //Doing contraction 1 of 2

    //\mathrm{tr}\left{\mathcal{G}^{(2,2)\,T}_{x ; y} \gamma^5 C \mathcal{G}^{(1,1)}_{x ; y} \gamma^5 C \right}
    tmpc1.real(1.0); tmpc1.imag(0.0);

    //Lowest level Spin-clr trace
    tmp_scmat_0 = q_f1_qpw.SiteMatrix(i,1);
    tmp_scmat_0.transpose();
    tmp_scmat_0.gr(-5).ccr(1);
    tmp_scmat_1 = q_f0_qpw.SiteMatrix(i,0);
    tmp_scmat_1.gr(-5).ccr(1);
    tmp_cnum_0 = Trace(tmp_scmat_0,tmp_scmat_1);

    tmpc1 *= tmp_cnum_0;

    //Finalising contraction 1
    corr(0,t) += tmpc1 * snk_phasefac;

    //Doing contraction 2 of 2

    //-\mathrm{tr}\left{\mathcal{G}^{(1,2)\,T}_{x ; y} \gamma^5 C \mathcal{G}^{(2,1)}_{x ; y} \gamma^5 C \right}
    tmpc2.real(-1.0); tmpc2.imag(0.0);

    //Lowest level Spin-clr trace
    tmp_scmat_0 = q_f1_qpw.SiteMatrix(i,0);
    tmp_scmat_0.transpose();
    tmp_scmat_0.gr(-5).ccr(1);
    tmp_scmat_1 = q_f0_qpw.SiteMatrix(i,1);
    tmp_scmat_1.gr(-5).ccr(1);
    tmp_cnum_0 = Trace(tmp_scmat_0,tmp_scmat_1);

    tmpc2 *= tmp_cnum_0;

    //Finalising contraction 2
    corr(1,t) += tmpc2 * snk_phasefac;
  }
  return true;
#endif
}




bool AlgGparityContract::pion_nogparity(const char *q_f0_tag, CorrelationFunction &corr){


  if(UniqueID()==0) printf("Doing standard pi^+/- contraction\n");
#if 0
  int shift_t = GJP.TnodeCoor()*GJP.TnodeSites();
  int spatial_vol = GJP.VolNodeSites()/GJP.TnodeSites();

  corr.setNcontractions(1);

  WilsonMatrix trace_a, trace_b, tmp;
  PropagatorContainer &q_f0_pc = PropManager::getProp(q_f0_tag);

  QPropW &q_f0_qpw = q_f0_pc.getProp(AlgLattice());

  Rcomplex tmpc;

  for(int i=0;i<GJP.VolNodeSites();i++){
    int t = i/spatial_vol + shift_t;

    //Doing contraction 1 of 1
    Rcomplex &wick0 = corr(0,t); 
    tmpc.real(1.0); tmpc.imag(0.0);

    //Doing trace 1 of 1
    trace_a = q_f0_qpw[i];
    trace_a.hconj();

    tmpc*= Trace(trace_a,q_f0_qpw[i]);

    wick0 += tmpc;
  }
#endif
  return true;
}


#endif


CPS_END_NAMESPACE
