#ifndef _COMPUTE_KAON_H
#define _COMPUTE_KAON_H

#include<alg/a2a/required_momenta.h>

//Compute stationary kaon two-point function with and without GPBC
CPS_START_NAMESPACE

//Policy for stationary kaon with and without GPBC
class StationaryKaonMomentaPolicy{
public:
  void setupMomenta(const int &ngp){
    RequiredMomentum<StationaryKaonMomentaPolicy> *tt = static_cast<RequiredMomentum<StationaryKaonMomentaPolicy>*>(this);
    if(ngp == 0){
      tt->addP("(0,0,0) + (0,0,0)");
    }else if(ngp == 1){
      tt->addP("(-1,0,0) + (1,0,0)");
    }else if(ngp == 2){
      tt->addP("(-1,-1,0) + (1,1,0)");
    }else if(ngp == 3){
      tt->addP("(-1,-1,-1) + (1,1,1)");
    }else{
      ERR.General("StationaryKaonMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
};


template<typename mf_Policies>
class ComputeKaon{
 public:
  typedef typename A2Asource<typename mf_Policies::SourcePolicies::ComplexType, typename mf_Policies::SourcePolicies::DimensionPolicy, typename mf_Policies::SourcePolicies::AllocPolicy>::FieldType::InputParamType FieldParamType;
  
  //Compute the two-point function using a hydrogen-wavefunction source of radius 'rad'
  //result is indexed by (tsrc, tsep)  where tsep is the source-sink separation
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into,
		      const A2AvectorW<mf_Policies> &W, const A2AvectorV<mf_Policies> &V, 
		      const A2AvectorW<mf_Policies> &W_s, const A2AvectorV<mf_Policies> &V_s,
		      const Float &rad, Lattice &lattice,
		      const FieldParamType &src_setup_params = NullObject()){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    typedef typename mf_Policies::SourcePolicies SourcePolicies;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    RequiredMomentum<StationaryKaonMomentaPolicy> kaon_momentum;
    ThreeMomentum p_w_src = kaon_momentum.getWmom(0);
    ThreeMomentum p_v_src = kaon_momentum.getVmom(0);
    
    ThreeMomentum p_w_snk = -p_w_src; //sink momentum is opposite source
    ThreeMomentum p_v_snk = -p_v_src;

    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > mf_ls(Lt);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > mf_sl(Lt);

    typedef typename mf_Policies::FermionFieldType::InputParamType VWfieldInputParams;
    VWfieldInputParams fld_params = V.getVh(0).getDimPolParams(); //use same field setup params as V/W input
    
    A2AvectorVfftw<mf_Policies> fftw_V(V.getArgs(),fld_params);
    A2AvectorWfftw<mf_Policies> fftw_W(W.getArgs(),fld_params);

    A2AvectorVfftw<mf_Policies> fftw_V_s(V_s.getArgs(),fld_params);
    A2AvectorWfftw<mf_Policies> fftw_W_s(W_s.getArgs(),fld_params);

    
    if(!GJP.Gparity()){
      A2AexpSource<SourcePolicies> expsrc(rad,src_setup_params);
      SCspinInnerProduct<ComplexType, A2AexpSource<SourcePolicies> > mf_struct(15,expsrc);

      fftw_W.gaugeFixTwistFFT(W,p_w_src.ptr(),lattice);
      fftw_V_s.gaugeFixTwistFFT(V_s,p_v_src.ptr(),lattice);

      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_ls, fftw_W, mf_struct, fftw_V_s);

      fftw_W_s.gaugeFixTwistFFT(W_s,p_w_snk.ptr(),lattice);
      fftw_V.gaugeFixTwistFFT(V,p_v_snk.ptr(),lattice);

      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_sl, fftw_W_s, mf_struct, fftw_V);
    }else{ //For GPBC we need a different smearing function for source and sink because the flavor structure depends on the momentum of the V field, which is opposite between source and sink
      A2AflavorProjectedExpSource<SourcePolicies> fpexp_src(rad, p_v_src.ptr(),src_setup_params);
      SCFspinflavorInnerProduct<ComplexType, A2AflavorProjectedExpSource<SourcePolicies> > mf_struct_src(sigma0,15,fpexp_src);

      fftw_W.gaugeFixTwistFFT(W,p_w_src.ptr(),lattice);
      fftw_V_s.gaugeFixTwistFFT(V_s,p_v_src.ptr(),lattice);

      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_ls, fftw_W, mf_struct_src, fftw_V_s);

      A2AflavorProjectedExpSource<SourcePolicies> fpexp_snk(rad, p_v_snk.ptr(),src_setup_params);
      SCFspinflavorInnerProduct<ComplexType, A2AflavorProjectedExpSource<SourcePolicies> > mf_struct_snk(sigma0,15,fpexp_snk);

      fftw_W_s.gaugeFixTwistFFT(W_s,p_w_snk.ptr(),lattice);
      fftw_V.gaugeFixTwistFFT(V,p_v_snk.ptr(),lattice);

      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_sl, fftw_W_s, mf_struct_snk, fftw_V);
    }

    //Compute the two-point function
    trace(into,mf_sl,mf_ls);
    into *= ScalarComplexType(0.5,0);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
  }


  






};

CPS_END_NAMESPACE

#endif

