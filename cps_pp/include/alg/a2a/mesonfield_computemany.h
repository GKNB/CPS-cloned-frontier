//Convenience functions for computing multiple meson fields with an array of sources and/or quark momenta

#ifndef _MESONFIELD_COMPUTE_MANY_H
#define _MESONFIELD_COMPUTE_MANY_H

#include<alg/a2a/threemomentum.h>

CPS_START_NAMESPACE


template<typename mf_Policies, typename StorageType>
class ComputeMesonFields{
 public:

  //W and V are indexed by the quark type index
  static void compute(StorageType &into, const std::vector< A2AvectorW<mf_Policies> const*> &W, const std::vector< A2AvectorV<mf_Policies> const*> &V,  Lattice &lattice){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::SourcePolicies SourcePolicies;
    typedef typename mf_Policies::FermionFieldType::InputParamType VWfieldInputParams;
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    assert(W.size() == V.size());
    const int nspecies = W.size();

    std::vector<bool> precompute_base_wffts(nspecies,false);
    std::vector<bool> precompute_base_vffts(nspecies,false);
     
#ifndef DISABLE_FFT_RELN_USAGE
    //Precompute the limited set of *base* FFTs  (remainder are related to base by cyclic permutation) provided this results in a cost saving
    const int nbase = GJP.Gparity() ? 2 : 1; //number of base FFTs
    const int nbase_max = 2;
    
    for(int s=0;s<nspecies;s++){
      if(into.nWffts(s) > nbase){
	precompute_base_wffts[s] = true;
	if(!UniqueID()) printf("ComputeMesonFields::compute precomputing W FFTs for species %d as nFFTs %d > %d\n",s,into.nWffts(s), nbase);
      }else if(!UniqueID()) printf("ComputeMesonFields::compute NOT precomputing W FFTs for species %d as nFFTs %d <= %d\n",s,into.nWffts(s), nbase);
      
      if(into.nVffts(s) > nbase){
	precompute_base_vffts[s] = true;
	if(!UniqueID()) printf("ComputeMesonFields::compute precomputing V FFTs for species %d as nFFTs %d > %d\n",s,into.nVffts(s), nbase);
      }else if(!UniqueID()) printf("ComputeMesonFields::compute NOT precomputing V FFTs for species %d as nFFTs %d <= %d\n",s,into.nVffts(s), nbase);
      
    }

    std::vector< std::vector<A2AvectorWfftw<mf_Policies>* > > Wfftw_base(nspecies);
    std::vector< std::vector<A2AvectorVfftw<mf_Policies>* > > Vfftw_base(nspecies);
    
    int p_0[3] = {0,0,0};
    int p_p1[3], p_m1[3];
    GparityBaseMomentum(p_p1,+1);
    GparityBaseMomentum(p_m1,-1);
    
    for(int s=0;s<nspecies;s++){
      Wfftw_base[s].resize(nbase_max,NULL);
      Vfftw_base[s].resize(nbase_max,NULL);
      
      for(int b=0;b<nbase;b++){
	if(precompute_base_wffts[s])
	  Wfftw_base[s][b] = new A2AvectorWfftw<mf_Policies>(W[s]->getArgs(), W[s]->getWh(0).getDimPolParams() );
	if(precompute_base_vffts[s])
	  Vfftw_base[s][b] = new A2AvectorVfftw<mf_Policies>(V[s]->getArgs(), V[s]->getMode(0).getDimPolParams() );
      }
	
      if(GJP.Gparity()){ //0 = +pi/2L  1 = -pi/2L  for each GP dir
	if(precompute_base_wffts[s]){
	  Wfftw_base[s][0]->gaugeFixTwistFFT(*W[s], p_p1,lattice);
	  Wfftw_base[s][1]->gaugeFixTwistFFT(*W[s], p_m1,lattice);
	}
	if(precompute_base_vffts[s]){
	  Vfftw_base[s][0]->gaugeFixTwistFFT(*V[s], p_p1,lattice);
	  Vfftw_base[s][1]->gaugeFixTwistFFT(*V[s], p_m1,lattice);
	}
      }else{
	if(precompute_base_wffts[s])
	  Wfftw_base[s][0]->gaugeFixTwistFFT(*W[s], p_0,lattice);
	if(precompute_base_vffts[s])
	  Vfftw_base[s][0]->gaugeFixTwistFFT(*V[s], p_0,lattice);
      }      
    } 
#endif

    for(int cidx=0; cidx < into.nCompute(); cidx++){
      typename StorageType::mfComputeInputFormat cdest = into.getMf(cidx);
      const typename StorageType::InnerProductType &M = into.getInnerProduct(cidx);

      int qidx_w, qidx_v;
      ThreeMomentum p_w, p_v;      
      into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,cidx);

      if(!UniqueID()){ printf("ComputeMesonFields::compute Computing mesonfield with W species %d and momentum %s and V species %d and momentum %s\n",qidx_w,p_w.str().c_str(),qidx_v,p_v.str().c_str()); fflush(stdout); }
      assert(qidx_w < nspecies && qidx_v < nspecies);
      
      A2AvectorWfftw<mf_Policies> fftw_W(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams() ); //temp storage for W
      A2AvectorVfftw<mf_Policies> fftw_V(V[qidx_v]->getArgs(), V[qidx_v]->getMode(0).getDimPolParams() );
            
#ifndef DISABLE_FFT_RELN_USAGE
      if(precompute_base_wffts[qidx_w])
	fftw_W.getTwistedFFT(p_w.ptr(), Wfftw_base[qidx_w][0], Wfftw_base[qidx_w][1]);
      else
	fftw_W.gaugeFixTwistFFT(*W[qidx_w], p_w.ptr(),lattice);

      if(precompute_base_vffts[qidx_v])
	fftw_V.getTwistedFFT(p_v.ptr(), Vfftw_base[qidx_v][0], Vfftw_base[qidx_v][1]);
      else
	fftw_V.gaugeFixTwistFFT(*V[qidx_v], p_v.ptr(),lattice);
	
#else
      fftw_W.gaugeFixTwistFFT(*W[qidx_w], p_w.ptr(),lattice);
      fftw_V.gaugeFixTwistFFT(*V[qidx_v], p_v.ptr(),lattice); 
#endif

      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(cdest,fftw_W, M, fftw_V);
    }
    
#ifndef DISABLE_FFT_RELN_USAGE
    for(int s=0;s<nspecies;s++){
      for(int b=0;b<nbase_max;b++){
	if(Wfftw_base[s][b] != NULL) delete Wfftw_base[s][b];
	if(Vfftw_base[s][b] != NULL) delete Vfftw_base[s][b];
      }
    }
#endif
    
  }

};



struct computeParams{
  int qidx_w;
  int qidx_v;
  ThreeMomentum p_w;
  ThreeMomentum p_v;

  bool do_src_shift;
  int src_shift[3]; //barrel shift the source FFT by this vector
  
  computeParams(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v), do_src_shift(false){}
  
  computeParams(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v, const int _src_shift[3]): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v), do_src_shift(true){
    memcpy(src_shift,_src_shift,3*sizeof(int));
  }
};

class MesonFieldStorageBase{
protected:
  std::vector<computeParams> clist;
  void getGPmomParams(int a[3], int k[3], const int p[3]){
    //Any allowed G-parity quark momentum can be written as   4*\vec a + \vec k   where k=(+1,+1,+1) or (-1,-1,-1)  [replace with zeroes when not Gparity directions]
    //Return the vectors a and k. For non GPBC directions set a[d]=p[d] and k[d]=0
    if(      (p[0]-1) % 4 == 0){
      for(int d=0;d<3;d++){
	if(GJP.Bc(d) == BND_CND_GPARITY){
	  assert( (p[d]-1) %4 == 0);
	  a[d] = (p[d]-1)/4;
	  k[d] = 1;
	}else{
	  a[d] = p[d]; k[d] = 0;
	}
      }
    }else if( (p[0]+1) % 4 == 0){
      for(int d=0;d<3;d++){
	if(GJP.Bc(d) == BND_CND_GPARITY){
	  assert( (p[d]+1) %4 == 0);
	  a[d] = (p[d]+1)/4;
	  k[d] = -1;
	}else{
	  a[d] = p[d]; k[d] = 0;
	}
      }
    }else ERR.General("MesonFieldStorageBase","getGPmomParams","Invalid momentum (%d,%d,%d)   p[0]-1 % 4 = %d   p[0]+1 % 4 = %d\n",p[0],p[1],p[2], (p[0]-1) % 4, (p[0]+1) % 4);
  }

  
public:
  void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v, bool use_mf_reln_simpl = false){
    if(!GJP.Gparity() || !use_mf_reln_simpl) clist.push_back( computeParams(qidx_w,qidx_v,p_w,p_v) );
    else{
      //M_ij^{4a+k,4b+l} =  \sum_{n=0}^{L-1} \Omega^{\dagger,4a+k}_i(n) \Gamma \gamma(n) N^{4b+l}_j(n)     (1)
      //                    \sum_{n=0}^{L-1} \Omega^{\dagger,k}_i(n-a-b) \Gamma \gamma(n-b) N^l_j(n)         (2)
      
      //\Omega^{\dagger,k}_i(n) = [ \sum_{x=0}^{L-1} e^{-2\pi i nx/L} e^{- (-k) \pi ix/2L} W_i(x) ]^\dagger
      //N^l_j(n) = \sum_{x=0}^{L-1} e^{-2\pi ix/L} e^{-l \pi ix/2L} V_i(x)

      //Use \Omega^{\dagger,k}_i(n-a-b) = \Omega^{\dagger,4a+4b+k}_i(n)   because the code handles the FFT relations for the V, W vectors separately
      ThreeMomentum a, k, b, l;
      ThreeMomentum p_wdag = -p_w;
      getGPmomParams(a.ptr(),k.ptr(),p_wdag.ptr());
      getGPmomParams(b.ptr(),l.ptr(),p_v.ptr());

      int src_shift[3] = {0,0,0};
      ThreeMomentum new_p_wdag = p_wdag, new_p_v = p_v;
      for(int i=0;i<3;i++)
	if(GJP.Bc(i) == BND_CND_GPARITY){
	  new_p_wdag(i) = 4*a(i) + 4*b(i) + k(i);
	  new_p_v(i) = l(i);
	  src_shift[i] = b(i); //shift in +b direction  \gamma(n') = \gamma(n-b)
	}
      
      if(!UniqueID()) printf("MesonFieldStorageBase: Converted p_wdag  %s = 4%s + %s   and p_v  %s = 4%s + %s to  p_wdag %s and p_v %s accompanied by source shift (%d,%d,%d)\n",
			     p_wdag.str().c_str(), a.str().c_str(), k.str().c_str(),
			     p_v.str().c_str(), b.str().c_str(), l.str().c_str(),
			     new_p_wdag.str().c_str(), new_p_v.str().c_str(),
			     src_shift[0],src_shift[1],src_shift[2]);
      
      clist.push_back( computeParams(qidx_w,qidx_v, -new_p_wdag, new_p_v, src_shift) );
    }
  }
  int nWffts(const int qidx) const{
    int count = 0;
    for(int i=0;i<clist.size();i++) if(clist[i].qidx_w == qidx) ++count;
    return count;
  }
  int nVffts(const int qidx) const{
    int count = 0;
    for(int i=0;i<clist.size();i++) if(clist[i].qidx_v == qidx) ++count;
    return count;
  }
  int nCompute() const{ return clist.size(); }

  void getComputeParameters(int &qidx_w, int &qidx_v, ThreeMomentum &p_w, ThreeMomentum &p_v, const int cidx) const{
    qidx_w = clist[cidx].qidx_w;    qidx_v = clist[cidx].qidx_v;
    p_w = clist[cidx].p_w;     p_v = clist[cidx].p_v; 
  }
  bool getSourceShift(int shift[3], const int cidx) const{
    if(clist[cidx].do_src_shift){
      memcpy(shift,clist[cidx].src_shift,3*sizeof(int));
      return true;
    }else return false;
  }
};

//Storage with source that remains constant for all computations
template<typename mf_Policies, typename InnerProduct, typename my_enable_if<!has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value, int>::type = 0>
class BasicSourceStorage: public MesonFieldStorageBase{
public:
  typedef InnerProduct InnerProductType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef storageType& mfComputeInputFormat;

private:
  const InnerProductType& inner;
  std::vector<storageType> mf;
public:  
  BasicSourceStorage(const InnerProductType& _inner): inner(_inner){}

  const storageType & operator[](const int cidx) const{ return mf[cidx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != clist.size()) mf.resize(clist.size());
    return mf[cidx]; //returns *reference*
  }  
  const InnerProductType & getInnerProduct(const int cidx){
    return inner;
  }
};

//Flavor projected operator needs to be fed the momentum of the second quark field in the bilinear
template<typename mf_Policies, typename InnerProduct, typename my_enable_if<!has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value, int>::type = 0>
class GparityFlavorProjectedBasicSourceStorage: public MesonFieldStorageBase{
public:
  typedef InnerProduct InnerProductType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef storageType& mfComputeInputFormat;

private:
  InnerProductType& inner;
  std::vector<storageType> mf;
public:  
  GparityFlavorProjectedBasicSourceStorage(InnerProductType& _inner): inner(_inner){} //note 'inner' will have its momentum sign changed dynamically

  const storageType & operator[](const int cidx) const{ return mf[cidx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != clist.size()) mf.resize(clist.size());
    return mf[cidx]; //returns *reference*
  }  
  const InnerProductType & getInnerProduct(const int cidx){
    inner.getSrc().setMomentum(clist[cidx].p_v.ptr());
    return inner;
  }
};




//Storage for a multi-source type requires different output meson fields for each source in the compound type
template<typename mf_Policies, typename InnerProduct, typename my_enable_if<has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value, int>::type = 0>
class MultiSourceStorage: public MesonFieldStorageBase{
public:
  typedef InnerProduct InnerProductType;
  typedef typename InnerProductType::InnerProductSourceType MultiSourceType;
  enum {nSources = MultiSourceType::nSources };
  
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef std::vector<storageType* > mfComputeInputFormat;

private:
  const InnerProductType& inner;
  std::vector<storageType> mf;
public:  
  MultiSourceStorage(const InnerProductType& _inner): inner(_inner){}

  const storageType & operator()(const int src_idx, const int cidx) const{ return mf[src_idx + nSources*cidx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != nSources*clist.size()) mf.resize(nSources*clist.size());
    mfComputeInputFormat ret(nSources);
    for(int i=0;i<nSources;i++) ret[i] = &mf[i + nSources*cidx];
    return ret;
  }  
  const InnerProductType & getInnerProduct(const int cidx) const{
    return inner;
  }
};


//Flavor projected version again requires momentum of right quark field in bilinear
template<typename MultiSrc, int Size, int I=0>
struct _multiSrcRecurse{
  static inline void setMomentum(MultiSrc &src, const int p[3]){
    src.template getSource<I>().setMomentum(p);
    _multiSrcRecurse<MultiSrc,Size-1,I+1>::setMomentum(src,p);
  }
};
template<typename MultiSrc, int I>
struct _multiSrcRecurse<MultiSrc,0,I>{
  static inline void setMomentum(MultiSrc &src, const int p[3]){}
};

template<typename mf_Policies, typename InnerProduct, typename my_enable_if<has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value, int>::type = 0>
class GparityFlavorProjectedMultiSourceStorage: public MesonFieldStorageBase{
public:
  typedef InnerProduct InnerProductType;
  typedef typename InnerProductType::InnerProductSourceType MultiSourceType;
  enum {nSources = MultiSourceType::nSources };
  
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef std::vector<storageType* > mfComputeInputFormat;

private:
  InnerProductType& inner;
  std::vector<storageType> mf;
public:  
  GparityFlavorProjectedMultiSourceStorage(InnerProductType& _inner): inner(_inner){} //note 'inner' will have its momentum sign changed dynamically

  const storageType & operator()(const int src_idx, const int cidx) const{ return mf[src_idx + nSources*cidx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != nSources*clist.size()) mf.resize(nSources*clist.size());
    mfComputeInputFormat ret(nSources);
    for(int i=0;i<nSources;i++) ret[i] = &mf[i + nSources*cidx];
    return ret;
  }  
  const InnerProductType & getInnerProduct(const int cidx){
    _multiSrcRecurse<MultiSourceType,nSources>::setMomentum(inner.getSrc(),clist[cidx].p_v.ptr());
    return inner;
  }
};









// struct computeParams{
//   int qidx_w;
//   int qidx_v;
//   ThreeMomentum p_w;
//   ThreeMomentum p_v;

//   bool do_src_shift;
//   int src_shift[3]; //barrel shift the source FFT by this vector
  
//   computeParams(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v), do_src_shift(false){}
  
//   computeParams(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v, const int _src_shift[3]): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v), do_src_shift(true){
//     memcpy(src_shift,_src_shift,3*sizeof(int));
//   }
// };

// class MesonFieldStorageBase{
// protected:
//   std::vector<computeParams> clist;
//   void getGPmomParams(int a[3], int k[3], const int p[3]){
//     //Any allowed G-parity quark momentum can be written as   4*\vec a + \vec k   where k=(+1,+1,+1) or (-1,-1,-1)  [replace with zeroes when not Gparity directions]
//     //Return the vectors a and k. For non GPBC directions set a[d]=p[d] and k[d]=0
//     if(      (p[0]-1) % 4 == 0){
//       for(int d=0;d<3;d++){
// 	if(GJP.Bc(d) == BND_CND_GPARITY){
// 	  assert( (p[d]-1) %4 == 0);
// 	  a[d] = (p[d]-1)/4;
// 	  k[d] = 1;
// 	}else{
// 	  a[d] = p[d]; k[d] = 0;
// 	}
//       }
//     }else if( (p[0]+1) % 4 == 0){
//       for(int d=0;d<3;d++){
// 	if(GJP.Bc(d) == BND_CND_GPARITY){
// 	  assert( (p[d]+1) %4 == 0);
// 	  a[d] = (p[d]+1)/4;
// 	  k[d] = -1;
// 	}else{
// 	  a[d] = p[d]; k[d] = 0;
// 	}
//       }
//     }else ERR.General("MesonFieldStorageBase","getGPmomParams","Invalid momentum (%d,%d,%d)   p[0]-1 % 4 = %d   p[0]+1 % 4 = %d\n",p[0],p[1],p[2], (p[0]-1) % 4, (p[0]+1) % 4);
//   }

  
// public:
//   void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v, bool use_mf_reln_simpl = false){
//     if(!GJP.Gparity() || !use_mf_reln_simpl) clist.push_back( computeParams(qidx_w,qidx_v,p_w,p_v) );
//     else{
//       //M_ij^{4a+k,4b+l} =  \sum_{n=0}^{L-1} \Omega^{\dagger,4a+k}_i(n) \Gamma \gamma(n) N^{4b+l}_j(n)     (1)
//       //                    \sum_{n=0}^{L-1} \Omega^{\dagger,k}_i(n-a-b) \Gamma \gamma(n-b) N^l_j(n)         (2)
      
//       //\Omega^{\dagger,k}_i(n) = [ \sum_{x=0}^{L-1} e^{-2\pi i nx/L} e^{- (-k) \pi ix/2L} W_i(x) ]^\dagger
//       //N^l_j(n) = \sum_{x=0}^{L-1} e^{-2\pi ix/L} e^{-l \pi ix/2L} V_i(x)

//       //Use \Omega^{\dagger,k}_i(n-a-b) = \Omega^{\dagger,4a+4b+k}_i(n)   because the code handles the FFT relations for the V, W vectors separately
//       ThreeMomentum a, k, b, l;
//       ThreeMomentum p_wdag = -p_w;
//       getGPmomParams(a.ptr(),k.ptr(),p_wdag.ptr());
//       getGPmomParams(b.ptr(),l.ptr(),p_v.ptr());

//       int src_shift[3] = {0,0,0};
//       ThreeMomentum new_p_wdag = p_wdag, new_p_v = p_v;
//       for(int i=0;i<3;i++)
// 	if(GJP.Bc(i) == BND_CND_GPARITY){
// 	  new_p_wdag(i) = 4*a(i) + 4*b(i) + k(i);
// 	  new_p_v(i) = l(i);
// 	  src_shift[i] = b(i); //shift in +b direction  \gamma(n') = \gamma(n-b)
// 	}
      
//       if(!UniqueID()) printf("MesonFieldStorageBase: Converted p_wdag  %s = 4%s + %s   and p_v  %s = 4%s + %s to  p_wdag %s and p_v %s accompanied by source shift (%d,%d,%d)\n",
// 			     p_wdag.str().c_str(), a.str().c_str(), k.str().c_str(),
// 			     p_v.str().c_str(), b.str().c_str(), l.str().c_str(),
// 			     new_p_wdag.str().c_str(), new_p_v.str().c_str(),
// 			     src_shift[0],src_shift[1],src_shift[2]);
      
//       clist.push_back( computeParams(qidx_w,qidx_v, -new_p_wdag, new_p_v, src_shift) );
//     }
//   }
//   int nWffts(const int qidx) const{
//     int count = 0;
//     for(int i=0;i<clist.size();i++) if(clist[i].qidx_w == qidx) ++count;
//     return count;
//   }
//   int nVffts(const int qidx) const{
//     int count = 0;
//     for(int i=0;i<clist.size();i++) if(clist[i].qidx_v == qidx) ++count;
//     return count;
//   }
//   int nCompute() const{ return clist.size(); }

//   void getComputeParameters(int &qidx_w, int &qidx_v, ThreeMomentum &p_w, ThreeMomentum &p_v, const int cidx) const{
//     qidx_w = clist[cidx].qidx_w;    qidx_v = clist[cidx].qidx_v;
//     p_w = clist[cidx].p_w;     p_v = clist[cidx].p_v; 
//   }
//   bool getSourceShift(int shift[3], const int cidx) const{
//     if(clist[cidx].do_src_shift){
//       memcpy(shift,clist[cidx].src_shift,3*sizeof(int));
//       return true;
//     }else return false;
//   }
// };






CPS_END_NAMESPACE

#endif
