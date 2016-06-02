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


template<typename mf_Float>
class ComputeKaon{
 public:
  typedef std::complex<mf_Float> mf_Complex;
  
  //Compute the two-point function using a hydrogen-wavefunction source of radius 'rad'
  //result is indexed by (tsrc, tsep)  where tsep is the source-sink separation
  static void compute(fMatrix<mf_Float> &into,
		      const A2AvectorW<mf_Complex> &W, const A2AvectorV<mf_Complex> &V, 
		      const A2AvectorW<mf_Complex> &W_s, const A2AvectorV<mf_Complex> &V_s,
		      const Float &rad, Lattice &lattice){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    RequiredMomentum<StationaryKaonMomentaPolicy> kaon_momentum;
    ThreeMomentum p_w_src = kaon_momentum.getWmom(0);
    ThreeMomentum p_v_src = kaon_momentum.getVmom(0);
    
    ThreeMomentum p_w_snk = -p_w_src; //sink momentum is opposite source
    ThreeMomentum p_v_snk = -p_v_src;

    //Construct the meson fields
    std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > mf_ls(Lt);
    std::vector<A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw> > mf_sl(Lt);

    A2AvectorVfftw<mf_Complex> fftw_V(V.getArgs());
    A2AvectorWfftw<mf_Complex> fftw_W(W.getArgs());

    A2AvectorVfftw<mf_Complex> fftw_V_s(V_s.getArgs());
    A2AvectorWfftw<mf_Complex> fftw_W_s(W_s.getArgs());

    
    if(!GJP.Gparity()){
      A2AexpSource expsrc(rad);
      SCspinInnerProduct<mf_Complex> mf_struct(15,expsrc);

      fftw_W.gaugeFixTwistFFT(W,p_w_src.ptr(),lattice);
      fftw_V_s.gaugeFixTwistFFT(V_s,p_v_src.ptr(),lattice);

      //for(int t=0;t<Lt;t++)
      //mf_ls[t].compute(fftw_W, mf_struct, fftw_V_s,t);

      A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_ls, fftw_W, mf_struct, fftw_V_s);

      fftw_W_s.gaugeFixTwistFFT(W_s,p_w_snk.ptr(),lattice);
      fftw_V.gaugeFixTwistFFT(V,p_v_snk.ptr(),lattice);

      A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_sl, fftw_W_s, mf_struct, fftw_V);

      //for(int t=0;t<Lt;t++)
      //mf_sl[t].compute(fftw_W_s, mf_struct, fftw_V,t);

    }else{ //For GPBC we need a different smearing function for source and sink because the flavor structure depends on the momentum of the V field, which is opposite between source and sink
      A2AflavorProjectedExpSource fpexp_src(rad, p_v_src.ptr());
      SCFspinflavorInnerProduct<mf_Complex, A2AflavorProjectedExpSource> mf_struct_src(sigma0,15,fpexp_src);

      fftw_W.gaugeFixTwistFFT(W,p_w_src.ptr(),lattice);
      fftw_V_s.gaugeFixTwistFFT(V_s,p_v_src.ptr(),lattice);

      //for(int t=0;t<Lt;t++)
      //mf_ls[t].compute(fftw_W, mf_struct_src, fftw_V_s,t);
      A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_ls, fftw_W, mf_struct_src, fftw_V_s);

      A2AflavorProjectedExpSource fpexp_snk(rad, p_v_snk.ptr());
      SCFspinflavorInnerProduct<mf_Complex, A2AflavorProjectedExpSource> mf_struct_snk(sigma0,15,fpexp_snk);

      fftw_W_s.gaugeFixTwistFFT(W_s,p_w_snk.ptr(),lattice);
      fftw_V.gaugeFixTwistFFT(V,p_v_snk.ptr(),lattice);

      A2AmesonField<mf_Float,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_sl, fftw_W_s, mf_struct_snk, fftw_V);

      //for(int t=0;t<Lt;t++)
      //	mf_sl[t].compute(fftw_W_s, mf_struct_snk, fftw_V,t);
    }

    //Compute the two-point function
    trace(into,mf_sl,mf_ls);
    into *= mf_Float(0.5);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
  }


  






};

CPS_END_NAMESPACE

#endif

