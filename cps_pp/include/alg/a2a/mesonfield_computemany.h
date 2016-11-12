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
    
    if(!UniqueID()) printf("ComputeMesonFields::compute Memory prior to V,W FFT precompute:\n");
    printMem();

    for(int s=0;s<nspecies;s++){
      Wfftw_base[s].resize(nbase_max,NULL);
      Vfftw_base[s].resize(nbase_max,NULL);
      
      for(int b=0;b<nbase;b++){
	if(precompute_base_wffts[s]){
	  if(!UniqueID()){ 
	    printf("ComputeMesonFields::compute Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[s]->getArgs(), W[s]->getWh(0).getDimPolParams())); fflush(stdout);
	  }
	  Wfftw_base[s][b] = new A2AvectorWfftw<mf_Policies>(W[s]->getArgs(), W[s]->getWh(0).getDimPolParams() );
	}
	if(precompute_base_vffts[s]){
	  if(!UniqueID()){ 
	    printf("ComputeMesonFields::compute Allocating a V FFT of size %f MB\n", A2AvectorVfftw<mf_Policies>::Mbyte_size(V[s]->getArgs(), V[s]->getMode(0).getDimPolParams())); fflush(stdout);
	  }
	  Vfftw_base[s][b] = new A2AvectorVfftw<mf_Policies>(V[s]->getArgs(), V[s]->getMode(0).getDimPolParams() );
	}
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

    if(!UniqueID()) printf("ComputeMesonFields::compute Memory prior to compute loop:\n");
    printMem();

    for(int cidx=0; cidx < into.nCompute(); cidx++){
      typename StorageType::mfComputeInputFormat cdest = into.getMf(cidx);
      const typename StorageType::InnerProductType &M = into.getInnerProduct(cidx);

      int qidx_w, qidx_v;
      ThreeMomentum p_w, p_v;      
      into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,cidx);

      if(!UniqueID()){ printf("ComputeMesonFields::compute Computing mesonfield with W species %d and momentum %s and V species %d and momentum %s\n",qidx_w,p_w.str().c_str(),qidx_v,p_v.str().c_str()); fflush(stdout); }
      assert(qidx_w < nspecies && qidx_v < nspecies);
      
      if(!UniqueID()){ 
	printf("ComputeMesonFields::compute Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams())); fflush(stdout);
      }
      A2AvectorWfftw<mf_Policies> fftw_W(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams() ); //temp storage for W

      if(!UniqueID()){ 
	printf("ComputeMesonFields::compute Allocating a V FFT of size %f MB\n", A2AvectorVfftw<mf_Policies>::Mbyte_size(V[qidx_v]->getArgs(), V[qidx_v]->getMode(0).getDimPolParams())); fflush(stdout);
      }
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

    if(!UniqueID()) printf("ComputeMesonFields::compute Memory after compute loop:\n");
    printMem();
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
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef InnerProduct InnerProductType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef storageType& mfComputeInputFormat;

private:
  const InnerProductType& inner;
  SourceType &src; //altered by changing momentum projection
  std::vector<storageType> mf;
public:  
  GparityFlavorProjectedBasicSourceStorage(const InnerProductType& _inner, SourceType &_src): inner(_inner), src(_src){} //note 'inner' will have its momentum sign changed dynamically

  const storageType & operator[](const int cidx) const{ return mf[cidx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != clist.size()) mf.resize(clist.size());
    return mf[cidx]; //returns *reference*
  }  
  const InnerProductType & getInnerProduct(const int cidx){
    src.setMomentum(clist[cidx].p_v.ptr());
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
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef InnerProduct InnerProductType;
  typedef typename InnerProductType::InnerProductSourceType MultiSourceType;
  enum {nSources = MultiSourceType::nSources };
  
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef std::vector<storageType* > mfComputeInputFormat;

private:
  const InnerProductType& inner;
  SourceType &src;
  std::vector<storageType> mf;
public:  
  GparityFlavorProjectedMultiSourceStorage(const InnerProductType& _inner, SourceType &_src): inner(_inner), src(_src){} //note 'inner' will have its momentum sign changed dynamically

  const storageType & operator()(const int src_idx, const int cidx) const{ return mf[src_idx + nSources*cidx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != nSources*clist.size()) mf.resize(nSources*clist.size());
    mfComputeInputFormat ret(nSources);
    for(int i=0;i<nSources;i++) ret[i] = &mf[i + nSources*cidx];
    return ret;
  }  
  const InnerProductType & getInnerProduct(const int cidx){
    _multiSrcRecurse<MultiSourceType,nSources>::setMomentum(src,clist[cidx].p_v.ptr());
    return inner;
  }
};




struct computeParamsMultiShift{
  int qidx_w;
  int qidx_v;
  ThreeMomentum p_w;
  ThreeMomentum p_v;

  std::vector<std::vector<int> > shifts;

  computeParamsMultiShift(){}
  computeParamsMultiShift(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v, const int _src_shift[3]): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v){
    addShift(_src_shift);
  }
  void addShift(const int _src_shift[3]){
    std::vector<int> s(3); for(int i=0;i<3;i++) s[i] = _src_shift[i];    
    shifts.push_back(s);
  }
};


struct KeyOnSpeciesAndMomentum{
  inline bool operator()(const computeParams &l, const computeParams &r){ 
    if(l.qidx_w < r.qidx_w) return true;
    else if(l.qidx_w > r.qidx_w) return false;
    
    if(l.qidx_v < r.qidx_v) return true;
    else if(l.qidx_v > r.qidx_v) return false;
    
    if(l.p_w < r.p_w) return true;
    else if(l.p_w > r.p_w) return false;
    
    if(l.p_v < r.p_v) return true;
    else if(l.p_v > r.p_v) return false;

    return false; //is equal
  }
};

class MesonFieldShiftSourceStorageBase : MesonFieldStorageBase{
protected:
  typedef std::map<computeParams,int,KeyOnSpeciesAndMomentum> MapType;
  std::vector<computeParamsMultiShift> optimized_clist; //shifts with same quark species and momenta combined
  std::vector< std::pair<int,int> > clist_opt_map; //mapping from original clist index to those in optimized storage
  bool optimized;
  
  void optimizeContractionList(){
    if(optimized) return;
    
    optimized_clist.resize(0);
    clist_opt_map.resize(clist.size());
    MapType keymap;
    
    for(int i=0;i<clist.size();i++){
      int shift[3] = {0,0,0};
      if(clist[i].do_src_shift) memcpy(shift, clist[i].src_shift, 3*sizeof(int));

      MapType::iterator loc = keymap.find(clist[i]);
      if(loc == keymap.end()){	//is a new one
	keymap[clist[i]] = optimized_clist.size();
	clist_opt_map[i] = std::pair<int,int>(optimized_clist.size(), 0);	
	optimized_clist.push_back( computeParamsMultiShift(clist[i].qidx_w, clist[i].qidx_v, clist[i].p_w, clist[i].p_v, shift) );	
      }else{
	clist_opt_map[i] = std::pair<int,int>(loc->second, optimized_clist[loc->second].shifts.size());
	optimized_clist[loc->second].addShift(shift);
      }     
    }
    if(!UniqueID()){
      printf("GparityFlavorProjectedShiftSourceStorage combined source shifts to:\n");
      for(int i=0;i<optimized_clist.size();i++){
	const computeParamsMultiShift &c = optimized_clist[i];
	printf("%d %d p_wdag %s p_v %s shifts : ",c.qidx_w, c.qidx_v, (-c.p_w).str().c_str(), c.p_v.str().c_str());
	for(int s=0;s<c.shifts.size();s++)
	  printf("(%d,%d,%d) ",c.shifts[s][0],c.shifts[s][1],c.shifts[s][2]);
	printf("\n");
      }
      printf("Internal mapping of cidx to <optimized cidx, shift index>:\n");
      for(int i=0;i<clist.size();i++)
	printf("%d -> (%d,%d)  ",i,clist_opt_map[i].first,clist_opt_map[i].second);
      printf("\n");
    }
    optimized = true;
  }
  
private:
  bool getSourceShift(int shift[3], const int cidx) const{
    assert(0);
  }    
public:
  
  MesonFieldShiftSourceStorageBase() : optimized_clist(0), optimized(false){}

  void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v){
    if(optimized){ optimized_clist.clear(); optimized = false; } //adding new computes means we have to redo the optimization
    this->MesonFieldStorageBase::addCompute(qidx_w,qidx_v,p_w,p_v,true);
  }
  
  //override base functions to use optimized list
  int nWffts(const int qidx){
    optimizeContractionList();
    int count = 0;
    for(int i=0;i<optimized_clist.size();i++) if(optimized_clist[i].qidx_w == qidx) ++count;
    return count;
  }
  int nVffts(const int qidx){
    optimizeContractionList();
    int count = 0;
    for(int i=0;i<optimized_clist.size();i++) if(optimized_clist[i].qidx_v == qidx) ++count;
    return count;
  }
  int nCompute(){
    optimizeContractionList();
    return optimized_clist.size();
  }

  void getComputeParameters(int &qidx_w, int &qidx_v, ThreeMomentum &p_w, ThreeMomentum &p_v, const int cidx){
    optimizeContractionList();
    qidx_w = optimized_clist[cidx].qidx_w;    qidx_v = optimized_clist[cidx].qidx_v;
    p_w = optimized_clist[cidx].p_w;     p_v = optimized_clist[cidx].p_v; 
  }
};
  


  
template<typename mf_Policies, typename InnerProduct>
class GparityFlavorProjectedShiftSourceStorage: public MesonFieldShiftSourceStorageBase{
public:
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef InnerProduct InnerProductType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef std::vector< storageType* > mfComputeInputFormat;

private:

  InnerProductType& inner;
  SourceType &src;
  std::vector<std::vector<storageType> > mf; //indexed by key(qidx_w,qidx_v,p_w,p_v) and shift
  bool locked;

  template<typename S=SourceType>
  inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type
  getNmf(const int opt_cidx){ return this->optimized_clist[opt_cidx].shifts.size(); }

  template<typename S=SourceType>
  inline typename my_enable_if<has_enum_nSources<S>::value, int>::type
  getNmf(const int opt_cidx){ return SourceType::nSources * this->optimized_clist[opt_cidx].shifts.size(); } //indexed by source_idx + nSources*shift_idx

  template<typename S=SourceType>
  inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
  setSourceMomentum(const int opt_cidx){ src.setMomentum(this->optimized_clist[opt_cidx].p_v.ptr()); }

  template<typename S=SourceType>
  inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
  setSourceMomentum(const int opt_cidx){ _multiSrcRecurse<S,S::nSources>::setMomentum(src,this->optimized_clist[opt_cidx].p_v.ptr() ); }
  
public:  
  GparityFlavorProjectedShiftSourceStorage(InnerProductType& _inner, SourceType &_src): inner(_inner),src(_src),locked(false){} //note 'inner' will have its momentum sign changed dynamically

  //Single source
  template<typename S=SourceType>
  typename my_enable_if<!has_enum_nSources<S>::value, const storageType &>::type
  operator[](const int orig_cidx) const{
    const std::pair<int,int> opt_loc = this->clist_opt_map[orig_cidx];
    return mf[opt_loc.first][opt_loc.second];
  }
  template<typename S=SourceType>
  typename my_enable_if<!has_enum_nSources<S>::value, storageType &>::type
  operator[](const int orig_cidx){
    const std::pair<int,int> opt_loc = this->clist_opt_map[orig_cidx];
    return mf[opt_loc.first][opt_loc.second];
  }


  //Multi source
  template<typename S=SourceType>
  typename my_enable_if<has_enum_nSources<S>::value, const storageType &>::type
  operator()(const int src_idx, const int orig_cidx) const{
    const std::pair<int,int> opt_loc = this->clist_opt_map[orig_cidx];
    return mf[opt_loc.first][src_idx + S::nSources*opt_loc.second];
  }
  template<typename S=SourceType>
  typename my_enable_if<has_enum_nSources<S>::value, storageType &>::type
  operator()(const int src_idx, const int orig_cidx){
    const std::pair<int,int> opt_loc = this->clist_opt_map[orig_cidx];
    return mf[opt_loc.first][src_idx + S::nSources*opt_loc.second];
  }



  
  mfComputeInputFormat getMf(const int opt_cidx){
    if(!locked){
      this->optimizeContractionList();
      mf.resize(this->optimized_clist.size());
      for(int c=0;c<this->optimized_clist.size();c++)
	mf[c].resize(getNmf(opt_cidx));
      locked = true;
    }
    std::vector< storageType* > out(getNmf(opt_cidx));
    for(int s=0;s<out.size();s++) out[s] = &mf[opt_cidx][s];
	
    return out;
  }  
  const InnerProductType & getInnerProduct(const int opt_cidx){
    setSourceMomentum(opt_cidx);
    inner.setShifts(this->optimized_clist[opt_cidx].shifts);    
    return inner;
  }

  void nodeDistributeResult(const int opt_cidx){
    for(int s=0;s<mf[opt_cidx].size();s++) nodeDistributeMany(1, &mf[opt_cidx][s]);    
  }
  void nodeGatherAll(){
    for(int c=0;c<mf.size();c++)
      for(int s=0;s<mf[c].size();s++)
	nodeGetMany(1, &mf[c][s]);  
  }
};


//With shifted source storage we guarantee that only 2 FFTs are needed for V; they're all shifted into shifted FFTs of the source and W. Thus we don't need to store the base V FFTs (which consume lots of memory)
template<typename mf_Policies, typename StorageType>
class ComputeMesonFieldsShiftSource{
public:
  //W and V are indexed by the quark type index
  static void compute(StorageType &into, const std::vector< A2AvectorW<mf_Policies> const*> &W, const std::vector< A2AvectorV<mf_Policies> const*> &V,  Lattice &lattice, const bool node_distribute = false){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::SourcePolicies SourcePolicies;
    typedef typename mf_Policies::FermionFieldType::InputParamType VWfieldInputParams;
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    assert(W.size() == V.size());
    const int nspecies = W.size();

    const int nbase = GJP.Gparity() ? 2 : 1; //number of base FFTs
    const int nbase_max = 2;
    int p_0[3] = {0,0,0};
    int p_p1[3], p_m1[3];
    GparityBaseMomentum(p_p1,+1);
    GparityBaseMomentum(p_m1,-1);

    ThreeMomentum pbase[nbase];
    if(GJP.Gparity()){ pbase[0] = ThreeMomentum(p_p1); pbase[1] = ThreeMomentum(p_m1); }
    else pbase[0] = ThreeMomentum(p_0);
    
    std::vector<bool> precompute_base_wffts(nspecies,false);

#ifndef DISABLE_FFT_RELN_USAGE
    //Precompute the limited set of *base* FFTs  (remainder are related to base by cyclic permutation) provided this results in a cost saving
    for(int s=0;s<nspecies;s++){
      if(into.nWffts(s) > nbase){
	precompute_base_wffts[s] = true;
	if(!UniqueID()) printf("ComputeMesonFields::compute <shift source> precomputing W FFTs for species %d as nFFTs %d > %d\n",s,into.nWffts(s), nbase);
      }else if(!UniqueID()) printf("ComputeMesonFields::compute <shift source> NOT precomputing W FFTs for species %d as nFFTs %d <= %d\n",s,into.nWffts(s), nbase);
    }

    std::vector< std::vector<A2AvectorWfftw<mf_Policies>* > > Wfftw_base(nspecies);
    
    if(!UniqueID()) printf("ComputeMesonFields::compute <shift source> Memory prior to V,W FFT precompute:\n");
    printMem();

    for(int s=0;s<nspecies;s++){
      Wfftw_base[s].resize(nbase_max,NULL);
      
      for(int b=0;b<nbase;b++){
	if(precompute_base_wffts[s]){
	  if(!UniqueID()){ 
	    printf("ComputeMesonFields::compute <shift source> Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[s]->getArgs(), W[s]->getWh(0).getDimPolParams())); fflush(stdout);
	  }
	  Wfftw_base[s][b] = new A2AvectorWfftw<mf_Policies>(W[s]->getArgs(), W[s]->getWh(0).getDimPolParams() );
	}
      }
	
      if(GJP.Gparity() && precompute_base_wffts[s]){ //0 = +pi/2L  1 = -pi/2L  for each GP dir
	Wfftw_base[s][0]->gaugeFixTwistFFT(*W[s], p_p1,lattice);
	Wfftw_base[s][1]->gaugeFixTwistFFT(*W[s], p_m1,lattice);
      }else if(precompute_base_wffts[s])
	Wfftw_base[s][0]->gaugeFixTwistFFT(*W[s], p_0,lattice);          
    } 
#endif

    if(!UniqueID()) printf("ComputeMesonFields::compute <shift source> Memory prior to compute loop:\n");
    printMem();

    //We need to group the V shifts by base and species
    std::vector< std::vector< std::vector<int> > > base_cidx_map(nbase, std::vector<std::vector<int> >(nspecies, std::vector<int>(0) ) );
    for(int c=0;c<into.nCompute();c++){
      int qidx_w, qidx_v;
      ThreeMomentum p_w, p_v;      
      into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,c);
      bool match = false;
      for(int b=0;b<nbase;b++)
	if(p_v == pbase[b]){
	  base_cidx_map[b][qidx_v].push_back(c); match = true; break;
	}
      if(!match) ERR.General("ComputeMesonFields","compute <shift source>","V momentum %s for species %d of computation %d is not in the set of base momenta\n",p_v.str().c_str(),qidx_v,c);
    }

    //Do the computations with an outer loop over the base momentum index and species of the Vfftw
    for(int b=0;b<nbase;b++){
      const ThreeMomentum &pvb = pbase[b];      
      for(int s=0;s<nspecies;s++){
	if(base_cidx_map[b][s].size() == 0) continue;

	if(!UniqueID()){ 
	  printf("ComputeMesonFields::compute <shift source> Allocating a V FFT of size %f MB\n", A2AvectorVfftw<mf_Policies>::Mbyte_size(V[s]->getArgs(), V[s]->getMode(0).getDimPolParams())); fflush(stdout);
	}
	A2AvectorVfftw<mf_Policies> fftw_V(V[s]->getArgs(), V[s]->getMode(0).getDimPolParams() );
	fftw_V.gaugeFixTwistFFT(*V[s], pvb.ptr(),lattice);
	
	//Now loop over computations with this V
	for(int cc=0; cc < base_cidx_map[b][s].size(); cc++){
	  const int cidx = base_cidx_map[b][s][cc];
	  
	  typename StorageType::mfComputeInputFormat cdest = into.getMf(cidx);
	  const typename StorageType::InnerProductType &M = into.getInnerProduct(cidx);

	  int qidx_w, qidx_v;
	  ThreeMomentum p_w, p_v;      
	  into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,cidx);

	  assert(p_v == pvb && qidx_v == s);
	  
	  if(!UniqueID()){ printf("ComputeMesonFields::compute <shift source> Computing mesonfield with W species %d and momentum %s and V species %d and momentum %s\n",qidx_w,p_w.str().c_str(),qidx_v,p_v.str().c_str()); fflush(stdout); }
	  assert(qidx_w < nspecies && qidx_v < nspecies);
      
#ifdef DISABLE_FFT_RELN_USAGE
	  if(!UniqueID()){ printf("ComputeMesonFields::compute <shift source> Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams())); fflush(stdout); }
	  A2AvectorWfftw<mf_Policies> fftw_W(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams() );
	  fftw_W.gaugeFixTwistFFT(*W[qidx_w], p_w.ptr(),lattice);
#elif defined(COMPUTEMANY_INPLACE_SHIFT)
	  A2AvectorWfftw<mf_Policies> *fftw_W_ptr = NULL;
	  bool delete_fftw_W_ptr = false, restore_fftw_W_ptr = false;
	  std::vector<int> restore_shift;
	  
	  if(precompute_base_wffts[qidx_w]){
	    if(!UniqueID()){ printf("ComputeMesonFields::compute <shift source> Shifting base Wfftw in place\n"); fflush(stdout); }
	    std::pair< A2AvectorWfftw<mf_Policies>*, std::vector<int> > inplace = A2AvectorWfftw<mf_Policies>::inPlaceTwistedFFT(p_w.ptr(), Wfftw_base[qidx_w][0], Wfftw_base[qidx_w][1]);
	    fftw_W_ptr = inplace.first;
	    restore_fftw_W_ptr = true; restore_shift = inplace.second;
	  }else{
	    if(!UniqueID()){ printf("ComputeMesonFields::compute <shift source> Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams())); fflush(stdout); }
	    fftw_W_ptr = new A2AvectorWfftw<mf_Policies>(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams() );
	    fftw_W_ptr->gaugeFixTwistFFT(*W[qidx_w], p_w.ptr(),lattice);
	    delete_fftw_W_ptr = true;
	  }
	  assert( fftw_W_ptr != NULL );
	  const A2AvectorWfftw<mf_Policies> &fftw_W = *fftw_W_ptr;
#else
	  if(!UniqueID()){ printf("ComputeMesonFields::compute <shift source> Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams())); fflush(stdout); }
	  A2AvectorWfftw<mf_Policies> fftw_W(W[qidx_w]->getArgs(), W[qidx_w]->getWh(0).getDimPolParams() ); 
	  if(precompute_base_wffts[qidx_w])
	    fftw_W.getTwistedFFT(p_w.ptr(), Wfftw_base[qidx_w][0], Wfftw_base[qidx_w][1]);
	  else
	    fftw_W.gaugeFixTwistFFT(*W[qidx_w], p_w.ptr(),lattice);
#endif
	  
	  A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(cdest,fftw_W, M, fftw_V);

#ifdef COMPUTEMANY_INPLACE_SHIFT
	  if(delete_fftw_W_ptr) delete fftw_W_ptr;
	  else if(restore_fftw_W_ptr) fftw_W_ptr->shiftFieldsInPlace(restore_shift);
#endif

	  if(node_distribute){
	    if(!UniqueID()) printf("ComputeMesonFields::compute <shift source> Memory before distribute:\n");
	    printMem();
	    into.nodeDistributeResult(cidx);
	    if(!UniqueID()) printf("ComputeMesonFields::compute <shift source> Memory after distribute:\n");
	    printMem();
	  }
	}//cc
      }//s
    }//b
    
#ifndef DISABLE_FFT_RELN_USAGE
    for(int s=0;s<nspecies;s++)
      for(int b=0;b<nbase_max;b++)
	if(Wfftw_base[s][b] != NULL) delete Wfftw_base[s][b];    
#endif

    if(!UniqueID()) printf("ComputeMesonFields::compute <shift source> Memory after compute loop:\n");
    printMem();
  }

};



template<typename mf_Policies,typename InnerProduct>
class ComputeMesonFields<mf_Policies,GparityFlavorProjectedShiftSourceStorage<mf_Policies,InnerProduct> >: public ComputeMesonFieldsShiftSource<mf_Policies,GparityFlavorProjectedShiftSourceStorage<mf_Policies,InnerProduct> >{ };











CPS_END_NAMESPACE

#endif
