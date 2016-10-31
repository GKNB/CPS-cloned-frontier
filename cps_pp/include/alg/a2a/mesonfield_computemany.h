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

    VWfieldInputParams fld_params = V[0]->getVh(0).getDimPolParams(); //use same field setup params as V/W input
    
    A2AvectorWfftw<mf_Policies> fftw_W(W[0]->getArgs(), fld_params); //temp storage for W
    A2AvectorVfftw<mf_Policies> fftw_V(V[0]->getArgs(), fld_params);

    assert(W.size() == V.size());
    const int nspecies = W.size();

    std::vector<bool> precompute_base_wffts(nspecies,false);
    std::vector<bool> precompute_base_vffts(nspecies,false);
     
#ifndef DISABLE_FFT_RELN_USAGE
    //Precompute the limited set of *base* FFTs  (remainder are related to base by cyclic permutation) provided this results in a cost saving
    int nbase = GJP.Gparity() ? 2 : 1; //number of base FFTs
    int nbase_max = 2;
    for(int s=0;s<nspecies;s++){
      if(into.nWmomenta(s) > nbase){
	precompute_base_wffts[s] = true;
	if(!UniqueID()) printf("ComputeMesonFields::compute precomputing W FFTs for species %d as nmomenta %d > %d\n",s,into.nWmomenta(s), nbase);
      }else if(!UniqueID()) printf("ComputeMesonFields::compute NOT precomputing W FFTs for species %d as nmomenta %d <= %d\n",s,into.nWmomenta(s), nbase);
      
      if(into.nVmomenta(s) > nbase){
	precompute_base_vffts[s] = true;
	if(!UniqueID()) printf("ComputeMesonFields::compute precomputing V FFTs for species %d as nmomenta %d > %d\n",s,into.nVmomenta(s), nbase);
      }else if(!UniqueID()) printf("ComputeMesonFields::compute NOT precomputing V FFTs for species %d as nmomenta %d <= %d\n",s,into.nVmomenta(s), nbase);
      
    }

    std::vector< std::vector<A2AvectorWfftw<mf_Policies>* > > Wfftw_base(nspecies, std::vector<A2AvectorWfftw<mf_Policies>* >(nbase_max,NULL) );
    std::vector< std::vector<A2AvectorVfftw<mf_Policies>* > > Vfftw_base(nspecies, std::vector<A2AvectorVfftw<mf_Policies>* >(nbase_max,NULL) );

    int p_0[3] = {0,0,0};
    int p_p1[3], p_m1[3];
    GparityBaseMomentum(p_p1,+1);
    GparityBaseMomentum(p_m1,-1);
    
    for(int s=0;s<nspecies;s++){
      for(int b=0;b<nbase;b++){
	if(precompute_base_wffts[s])
	  Wfftw_base[s][b] = new A2AvectorWfftw<mf_Policies>(fftw_W.getArgs(), fld_params);
	if(precompute_base_vffts[s])
	  Vfftw_base[s][b] = new A2AvectorVfftw<mf_Policies>(fftw_V.getArgs(), fld_params);
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
 
  }

};





struct computeParams{
  int qidx_w;
  int qidx_v;
  ThreeMomentum p_w;
  ThreeMomentum p_v;

  computeParams(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v){}
};

class MesonFieldStorageBase{
protected:
  std::vector<computeParams> clist;
  std::map<int, std::set<ThreeMomentum> > w_species_mom_map;
  std::map<int, std::set<ThreeMomentum> > v_species_mom_map;
public:
  void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v){
    clist.push_back( computeParams(qidx_w,qidx_v,p_w,p_v) );
    w_species_mom_map[qidx_w].insert(p_w);
    v_species_mom_map[qidx_v].insert(p_v);		     
  }
  int nWmomenta(const int qidx) const{
    std::map<int, std::set<ThreeMomentum> >::const_iterator it = w_species_mom_map.find(qidx);
    if(it == w_species_mom_map.end()) return 0;
    else return it->second.size();
  }
  int nVmomenta(const int qidx) const{
    std::map<int, std::set<ThreeMomentum> >::const_iterator it = v_species_mom_map.find(qidx);
    if(it == v_species_mom_map.end()) return 0;
    else return it->second.size();
  }
  int nCompute() const{ return clist.size(); }

  void getComputeParameters(int &qidx_w, int &qidx_v, ThreeMomentum &p_w, ThreeMomentum &p_v, const int cidx) const{
    qidx_w = clist[cidx].qidx_w;    qidx_v = clist[cidx].qidx_v;
    p_w = clist[cidx].p_w;     p_v = clist[cidx].p_v; 
  }
};

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



CPS_END_NAMESPACE

#endif
