#ifndef _MESONFIELD_COMPUTEMANY_STORAGETYPES_H
#define _MESONFIELD_COMPUTEMANY_STORAGETYPES_H

#include<alg/a2a/threemomentum.h>
#include<alg/a2a/a2a_sources.h>
#include<alg/a2a/mesonfield.h>

CPS_START_NAMESPACE

struct computeParams{
  int qidx_w;
  int qidx_v;
  ThreeMomentum p_w;
  ThreeMomentum p_v;

  bool do_src_shift;
  int src_shift[3]; //barrel shift the source FFT by this vector
  
  computeParams(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v), do_src_shift(false){}
  
  computeParams(const int _qidx_w, const int _qidx_v, const ThreeMomentum &_p_w, const ThreeMomentum &_p_v, const int _src_shift[3]): qidx_w(_qidx_w),  qidx_v(_qidx_v), p_w(_p_w), p_v(_p_v), 
    do_src_shift(true){
    memcpy(src_shift,_src_shift,3*sizeof(int));
  }
};

class MesonFieldStorageBase{
protected:
  std::vector<computeParams> clist;
public:
  static void getGPmomParams(int a[3], int k[3], const int p[3]){
    if(!GJP.Gparity()){
      for(int i=0;i<3;i++){ a[i]=p[i]; k[i]=0; }
      return;
    } 
    
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

  //Add the parameters of a meson field to the list of pending calculations
  inline void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v){
    clist.push_back( computeParams(qidx_w,qidx_v,p_w,p_v) );
  }
  inline int nWffts(const int qidx) const{
    int count = 0;
    for(int i=0;i<clist.size();i++) if(clist[i].qidx_w == qidx) ++count;
    return count;
  }
  inline int nVffts(const int qidx) const{
    int count = 0;
    for(int i=0;i<clist.size();i++) if(clist[i].qidx_v == qidx) ++count;
    return count;
  }
  inline int nCompute() const{ return clist.size(); }

  inline void getComputeParameters(int &qidx_w, int &qidx_v, ThreeMomentum &p_w, ThreeMomentum &p_v, const int cidx) const{
    qidx_w = clist[cidx].qidx_w;    qidx_v = clist[cidx].qidx_v;
    p_w = clist[cidx].p_w;     p_v = clist[cidx].p_v; 
  }
  inline bool needSourceShift(const int cidx) const{ return clist[cidx].do_src_shift; }

  //If some additional action needs to be performed immediately after a meson field is computed
  inline void postContractAction(const int cidx){}
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
  storageType & operator[](const int cidx){ return mf[cidx]; }

  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != clist.size()) mf.resize(clist.size());
    return mf[cidx]; //returns *reference*
  }  
  const InnerProductType & getInnerProduct(const int cidx){
    if(needSourceShift(cidx)) ERR.General("BasicSourceStorage","getInnerProduct","Policy does not support source shifting\n");
    return inner;
  }
  void nodeDistributeResult(const int cidx){
    nodeDistributeMany(1, &mf[cidx]);    
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
  storageType & operator[](const int cidx){ return mf[cidx]; }

  const storageType & operator()(const int cidx) const{ return mf[cidx]; }
  storageType & operator()(const int cidx){ return mf[cidx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != clist.size()) mf.resize(clist.size());
    return mf[cidx]; //returns *reference*
  }  
  const InnerProductType & getInnerProduct(const int cidx){
    if(needSourceShift(cidx)) ERR.General("GparityFlavorProjectedBasicSourceStorage","getInnerProduct","Policy does not support source shifting\n");
    src.setMomentum(clist[cidx].p_v.ptr());
    return inner;
  }
  void nodeDistributeResult(const int cidx){
    nodeDistributeMany(1, &mf[cidx]);    
  }
    
};


//This version sums alternate momenta on the fly
template<typename mf_Policies, typename InnerProduct, typename my_enable_if<!has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value, int>::type = 0>
class GparityFlavorProjectedSumSourceStorage: public MesonFieldStorageBase{
public:
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef InnerProduct InnerProductType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef storageType& mfComputeInputFormat;

private:
  const InnerProductType& inner;
  SourceType &src; //altered by changing momentum projection
  storageType tmp_mf;
  std::vector<storageType> mf;

  std::map<int,int> set_first_cidx; //keep track of the first cidx of a set that is computed
  std::vector< std::vector<int> > set_cidx_map; //[set_idx] -> set of cidx
  std::vector<int> cidx_set_map; //[cidx] -> set_idx

  inline void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v){ assert(0); };
public:  
  GparityFlavorProjectedSumSourceStorage(const InnerProductType& _inner, SourceType &_src): inner(_inner), src(_src){} //note 'inner' will have its momentum sign changed dynamically
  
  void addComputeSet(const int qidx_w, const int qidx_v, std::vector< std::pair<ThreeMomentum, ThreeMomentum> > &mom_wv_pairs){    
    int set_idx = set_cidx_map.size();
    std::vector<int> set_cidx;
    for(int i=0;i<mom_wv_pairs.size();i++){
      int cidx = clist.size();
      clist.push_back( computeParams(qidx_w,qidx_v,mom_wv_pairs[i].first, mom_wv_pairs[i].second) );
      cidx_set_map.push_back(set_idx);

      set_cidx.push_back(cidx);
    }
    set_cidx_map.push_back(set_cidx);
  }

  const storageType & operator[](const int set_idx) const{ return mf[set_idx]; }
  storageType & operator[](const int set_idx){ return mf[set_idx]; }

  const storageType & operator()(const int set_idx) const{ return mf[set_idx]; }
  storageType & operator()(const int set_idx){ return mf[set_idx]; }
  
  mfComputeInputFormat getMf(const int cidx){
    if(mf.size() != set_cidx_map.size()) mf.resize(set_cidx_map.size());
    int set = cidx_set_map[cidx];

    std::map<int,int>::const_iterator it = set_first_cidx.find(set);
    if(it == set_first_cidx.end()){
      set_first_cidx[set] = cidx;
      return mf[set];
    }else{
      return tmp_mf;
    }
  }  

  inline void postContractAction(const int cidx){
    int set = cidx_set_map[cidx];
    int set_first = set_first_cidx[set];
    if(cidx != set_first)
      for(int t=0;t<mf[set].size();t++){
	bool did_gather = false;
	if(!mf[set][t].isOnNode()){ mf[set][t].nodeGet(); did_gather = true; }
	mf[set][t].plus_equals(tmp_mf[t]);
	if(did_gather) mf[set][t].nodeDistribute();
      }
  }

  const InnerProductType & getInnerProduct(const int cidx){
    if(needSourceShift(cidx)) ERR.General("GparityFlavorProjectedSumSourceStorage","getInnerProduct","Policy does not support source shifting\n");
    src.setMomentum(clist[cidx].p_v.ptr());
    return inner;
  }
  void nodeDistributeResult(const int cidx){
    int set = cidx_set_map[cidx];
    int set_first = set_first_cidx[set];
    if(cidx == set_first){
      nodeDistributeMany(1, &mf[set]);    
    }
  }
  
  //The internal operation sums the meson fields; use this method to normalize the sum to an average for all meson fields computed
  void sumToAverage(){
    for(int s=0;s<set_cidx_map.size();s++){
      int nc = set_cidx_map[s].size();
      for(int t=0;t<mf[s].size();t++){
	bool did_gather = false;
	if(!mf[s][t].isOnNode()){ mf[s][t].nodeGet(); did_gather = true; }
	mf[s][t].times_equals(1./nc);
	if(did_gather) mf[s][t].nodeDistribute();
      }
    }
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
    if(needSourceShift(cidx)) ERR.General("MultiSourceStorage","getInnerProduct","Policy does not support source shifting\n");
    return inner;
  }
  void nodeDistributeResult(const int cidx){
    for(int s=0;s<nSources;s++) nodeDistributeMany(1, &mf[s + nSources*cidx]);    
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
    if(needSourceShift(cidx)) ERR.General("GparityFlavorProjectedMultiSourceStorage","getInnerProduct","Policy does not support source shifting\n");
    _multiSrcRecurse<MultiSourceType,nSources>::setMomentum(src,clist[cidx].p_v.ptr());
    return inner;
  }

  void nodeDistributeResult(const int cidx){
    for(int s=0;s<nSources;s++) nodeDistributeMany(1, &mf[s + nSources*cidx]);    
  }
};

//If using G-parity BCs, we can move the momentum shift from base for the right A2A vector into a C-shift in the source accompanied by a different
//momentum for the left A2A vector. This can speed up the calculation.

class MesonFieldShiftSourceStorageBase : public MesonFieldStorageBase{
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
    inline bool operator()(const computeParams &l, const computeParams &r) const{ 
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
protected:
  typedef std::map<computeParams,int,KeyOnSpeciesAndMomentum> MapType;
  std::vector<computeParamsMultiShift> optimized_clist; //shifts with same quark species and momenta combined
  std::vector< std::pair<int,int> > clist_opt_map; //mapping from original clist index to those in optimized storage  orig_cidx -> (opt_cidx, shift_idx)
  bool optimized;
  int nshift_max;

  void optimizeContractionList(){
    if(optimized) return;
    
    optimized_clist.resize(0);
    clist_opt_map.resize(clist.size());
    MapType keymap;
    
    for(int i=0;i<clist.size();i++){
      int shift[3] = {0,0,0};
      if(clist[i].do_src_shift) memcpy(shift, clist[i].src_shift, 3*sizeof(int));

      MapType::iterator loc = keymap.find(clist[i]);
      if(loc == keymap.end() || optimized_clist[loc->second].shifts.size() == nshift_max){	//is a new one
	keymap[clist[i]] = optimized_clist.size();
	clist_opt_map[i] = std::pair<int,int>(optimized_clist.size(), 0);	
	optimized_clist.push_back( computeParamsMultiShift(clist[i].qidx_w, clist[i].qidx_v, clist[i].p_w, clist[i].p_v, shift) );	
      }else{
	clist_opt_map[i] = std::pair<int,int>(loc->second, optimized_clist[loc->second].shifts.size());
	optimized_clist[loc->second].addShift(shift);
      }     
    }
    if(!UniqueID()){
      printf("MesonFieldShiftSourceStorageBase combined source shifts to:\n");
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
  bool needSourceShift(const int cidx) const{
    assert(0);
  }
public:
  
  //nshift_max sets the maximum number of momentum pairs that are combined into shift sources - useful to control memory usage. Set to -1 for unlimited
  MesonFieldShiftSourceStorageBase(const int nshift_max = -1) : optimized_clist(0), optimized(false), nshift_max(nshift_max){}

  void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v){
    if(optimized){ optimized_clist.clear(); optimized = false; } //adding new computes means we have to redo the optimization

    assert(GJP.Gparity());
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
    
    if(!UniqueID()) printf("MesonFieldShiftSourceStorageBase: Converted p_wdag  %s = 4%s + %s   and p_v  %s = 4%s + %s to  p_wdag %s and p_v %s accompanied by source shift (%d,%d,%d)\n",
			   p_wdag.str().c_str(), a.str().c_str(), k.str().c_str(),
			   p_v.str().c_str(), b.str().c_str(), l.str().c_str(),
			   new_p_wdag.str().c_str(), new_p_v.str().c_str(),
			   src_shift[0],src_shift[1],src_shift[2]);
    
    this->clist.push_back( computeParams(qidx_w,qidx_v, -new_p_wdag, new_p_v, src_shift) );
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
class GparityFlavorProjectedShiftSourceStorage;

template<typename mf_Policies, typename InnerProduct, bool isMultiSrc>
class _GparityFlavorProjectedShiftSourceStorageAccessors{};

//Single source
template<typename mf_Policies, typename InnerProduct>
class _GparityFlavorProjectedShiftSourceStorageAccessors<mf_Policies,InnerProduct,false>{
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies,InnerProduct> Derived;
public:
  storageType & operator[](const int orig_cidx){ 
    const std::pair<int,int> opt_loc = static_cast<Derived*>(this)->clist_opt_map[orig_cidx];
    return static_cast<Derived*>(this)->mf[opt_loc.first][opt_loc.second];
  }
  const storageType & operator[](const int orig_cidx) const{ 
    const std::pair<int,int> opt_loc = static_cast<Derived const*>(this)->clist_opt_map[orig_cidx];
    return static_cast<Derived const*>(this)->mf[opt_loc.first][opt_loc.second];
  }
  inline storageType & operator()(const int orig_cidx){ return this->operator[](orig_cidx); }
  inline const storageType & operator()(const int orig_cidx) const{ return this->operator[](orig_cidx); }

  typedef int accessorIdxType;
};
//Multi source
template<typename mf_Policies, typename InnerProduct>
class _GparityFlavorProjectedShiftSourceStorageAccessors<mf_Policies,InnerProduct,true>{
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef GparityFlavorProjectedShiftSourceStorage<mf_Policies,InnerProduct> Derived;
public:
  storageType & operator()(const int src_idx, const int orig_cidx){
    const std::pair<int,int> opt_loc = static_cast<Derived*>(this)->clist_opt_map[orig_cidx];
    return static_cast<Derived*>(this)->mf[opt_loc.first][src_idx + SourceType::nSources*opt_loc.second];
  }
  const storageType & operator()(const int src_idx, const int orig_cidx) const{
    const std::pair<int,int> opt_loc = static_cast<Derived const*>(this)->clist_opt_map[orig_cidx];
    return static_cast<Derived const*>(this)->mf[opt_loc.first][src_idx + SourceType::nSources*opt_loc.second];
  }

  typedef std::pair<int,int> accessorIdxType; //(src, cidx)
  
  storageType & operator()(const accessorIdxType &idx){ return (*this)(idx.first, idx.second); }
  const storageType & operator()(const accessorIdxType &idx) const{ return (*this)(idx.first, idx.second); }
};
  

  
template<typename mf_Policies, typename InnerProduct>
class GparityFlavorProjectedShiftSourceStorage: public MesonFieldShiftSourceStorageBase,
						public _GparityFlavorProjectedShiftSourceStorageAccessors<mf_Policies,InnerProduct,has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value>{
public:
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef InnerProduct InnerProductType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef std::vector< storageType* > mfComputeInputFormat;
  typedef typename _GparityFlavorProjectedShiftSourceStorageAccessors<mf_Policies,InnerProduct,has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value>::accessorIdxType accessorIdxType;
  friend class _GparityFlavorProjectedShiftSourceStorageAccessors<mf_Policies,InnerProduct,has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value>;
private:

  InnerProductType& inner;
  SourceType &src;
  std::vector<std::vector<storageType> > mf; //indexed by key(qidx_w,qidx_v,p_w,p_v) and shift
  bool locked;

  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type
  getNmf(const int opt_cidx){ return this->optimized_clist[opt_cidx].shifts.size(); }

  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, int>::type
  getNmf(const int opt_cidx){ return S::nSources * this->optimized_clist[opt_cidx].shifts.size(); } //indexed by source_idx + nSources*shift_idx

  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
  setSourceMomentum(const int opt_cidx){ src.setMomentum(this->optimized_clist[opt_cidx].p_v.ptr()); }

  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
  setSourceMomentum(const int opt_cidx){ _multiSrcRecurse<S,S::nSources>::setMomentum(src,this->optimized_clist[opt_cidx].p_v.ptr() ); }

public:  
  GparityFlavorProjectedShiftSourceStorage(InnerProductType& _inner, SourceType &_src, const int nshift_max = -1): inner(_inner),src(_src),locked(false), MesonFieldShiftSourceStorageBase(nshift_max){} //note 'inner' will have its momentum sign changed dynamically

  mfComputeInputFormat getMf(const int opt_cidx){
    if(!locked){
      this->optimizeContractionList();
      mf.resize(this->optimized_clist.size());
      for(int c=0;c<this->optimized_clist.size();c++)
	mf[c].resize(getNmf<SourceType>(opt_cidx));
      locked = true;
    }
    std::vector< storageType* > out(getNmf<SourceType>(opt_cidx));
    for(int s=0;s<out.size();s++) out[s] = &mf[opt_cidx][s];
	
    return out;
  }  
  const InnerProductType & getInnerProduct(const int opt_cidx){
    setSourceMomentum<SourceType>(opt_cidx);
    inner.setShifts(this->optimized_clist[opt_cidx].shifts);    
    return inner;
  }

  void nodeDistributeResult(const int opt_cidx){
    for(int s=0;s<mf[opt_cidx].size();s++) nodeDistributeMany(1, &mf[opt_cidx][s]);    
  }  
};


//Version of the above that sums alternate momenta on the fly
template<typename mf_Policies, typename InnerProduct>
class GparityFlavorProjectedShiftSourceSumStorage;

template<typename mf_Policies, typename InnerProduct, bool isMultiSrc>
class _GparityFlavorProjectedShiftSourceSumStorageAccessors{};

//Single source
template<typename mf_Policies, typename InnerProduct>
class _GparityFlavorProjectedShiftSourceSumStorageAccessors<mf_Policies,InnerProduct,false>{
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef GparityFlavorProjectedShiftSourceSumStorage<mf_Policies,InnerProduct> Derived;
public:
  storageType & operator[](const int set_idx){ 
    return static_cast<Derived*>(this)->mf_sum[set_idx][0];
  }
  const storageType & operator[](const int set_idx) const{ 
    return static_cast<Derived const*>(this)->mf_sum[set_idx][0];
  }
  inline storageType & operator()(const int set_idx){ return this->operator[](set_idx); }
  inline const storageType & operator()(const int set_idx) const{ return this->operator[](set_idx); }

  typedef int accessorIdxType;
};
//Multi source
template<typename mf_Policies, typename InnerProduct>
class _GparityFlavorProjectedShiftSourceSumStorageAccessors<mf_Policies,InnerProduct,true>{
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef GparityFlavorProjectedShiftSourceSumStorage<mf_Policies,InnerProduct> Derived;
public:
  storageType & operator()(const int src_idx, const int set_idx){
    return static_cast<Derived*>(this)->mf_sum[set_idx][src_idx];
  }
  const storageType & operator()(const int src_idx, const int set_idx) const{
    return static_cast<Derived const*>(this)->mf_sum[set_idx][src_idx];
  }

  typedef std::pair<int,int> accessorIdxType; //(src, set_idx)
  
  storageType & operator()(const accessorIdxType &idx){ return (*this)(idx.first, idx.second); }
  const storageType & operator()(const accessorIdxType &idx) const{ return (*this)(idx.first, idx.second); }
};
  

template<typename mf_Policies, typename InnerProduct>
class GparityFlavorProjectedShiftSourceSumStorage: public MesonFieldShiftSourceStorageBase,
						   public _GparityFlavorProjectedShiftSourceSumStorageAccessors<mf_Policies,InnerProduct,has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value>{
public:
  typedef typename InnerProduct::InnerProductSourceType SourceType;
  typedef InnerProduct InnerProductType;
  typedef std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > storageType;
  typedef std::vector< storageType* > mfComputeInputFormat;
  typedef typename _GparityFlavorProjectedShiftSourceSumStorageAccessors<mf_Policies,InnerProduct,has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value>::accessorIdxType accessorIdxType;
  friend class _GparityFlavorProjectedShiftSourceSumStorageAccessors<mf_Policies,InnerProduct,has_enum_nSources<typename InnerProduct::InnerProductSourceType>::value>;
private:

  InnerProductType& inner;
  SourceType &src;
  bool locked;

  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type
  getNsrc(){ return 1; }

  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, int>::type
  getNsrc(){ return S::nSources; }

  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, int>::type
  getNmf(const int opt_cidx){ return this->optimized_clist[opt_cidx].shifts.size(); }

  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, int>::type
  getNmf(const int opt_cidx){ return S::nSources * this->optimized_clist[opt_cidx].shifts.size(); } //indexed by source_idx + nSources*shift_idx

  template<typename S>
  inline typename my_enable_if<!has_enum_nSources<S>::value, void>::type
  setSourceMomentum(const int opt_cidx){ src.setMomentum(this->optimized_clist[opt_cidx].p_v.ptr()); }

  template<typename S>
  inline typename my_enable_if<has_enum_nSources<S>::value, void>::type
  setSourceMomentum(const int opt_cidx){ _multiSrcRecurse<S,S::nSources>::setMomentum(src,this->optimized_clist[opt_cidx].p_v.ptr() ); }

  void addCompute(const int qidx_w, const int qidx_v, const ThreeMomentum &p_w, const ThreeMomentum &p_v){ assert(0); }

  std::vector< std::vector<int> > set_cidx_map; //[set_idx] -> set of cidx
  std::vector<int> cidx_set_map; //[cidx] -> set_idx

  std::vector<bool> set_first;
  std::vector< std::vector<storageType> > mf_sum;
  std::vector<storageType*> mf_tmp;
  std::vector<double> set_nrm;

  struct sumInfo{
    storageType* to;
    storageType* from;
    sumInfo(storageType* t, storageType* f): to(t), from(f){}
  };
  struct normInfo{
    storageType* to;
    double nrm;
    normInfo(storageType* to, double nrm): to(to), nrm(nrm){}
  };

  std::map<int, std::vector<normInfo> > norm_queue; //opt_cidx -> (nrm_to, nrm_val)
  std::map<int, std::vector<sumInfo>  > sum_queue; //opt_cidx -> (sum_to, sum_from)
  std::map<int, std::vector<storageType*> > distribute_queue;
public:  
  GparityFlavorProjectedShiftSourceSumStorage(InnerProductType& _inner, SourceType &_src, const int nshift_max = -1): inner(_inner),src(_src),locked(false), MesonFieldShiftSourceStorageBase(nshift_max){} //note 'inner' will have its momentum sign changed dynamically

  ~GparityFlavorProjectedShiftSourceSumStorage(){ for(int i=0;i<mf_tmp.size();i++) delete(mf_tmp[i]); }

  //if !do_average, the sets are just summed
  void addComputeSet(const int qidx_w, const int qidx_v, std::vector< std::pair<ThreeMomentum, ThreeMomentum> > &mom_wv_pairs, bool do_average = false){    
    int set_idx = set_cidx_map.size();
    std::vector<int> set_cidx;
    for(int i=0;i<mom_wv_pairs.size();i++){
      int cidx = clist.size();
      this->MesonFieldShiftSourceStorageBase::addCompute(qidx_w,qidx_v,mom_wv_pairs[i].first, mom_wv_pairs[i].second);
      cidx_set_map.push_back(set_idx);

      set_cidx.push_back(cidx);
    }
    set_cidx_map.push_back(set_cidx);
    set_nrm.push_back(do_average ? 1./mom_wv_pairs.size() : 1.);
  }

  mfComputeInputFormat getMf(const int opt_cidx){
    if(!locked){
      this->optimizeContractionList();
      mf_sum.resize(set_cidx_map.size(), std::vector<storageType>(getNsrc<SourceType>()) ); //one mf for each set up to src multiplicity
      set_first.resize(set_cidx_map.size(), true);
      locked = true;
    }

    int tmpidx = 0;

    int nsrc = getNsrc<SourceType>();

    std::vector< storageType* > out(getNmf<SourceType>(opt_cidx), NULL);   //[ src_idx + nsrc * shift ]

    int nshift = this->optimized_clist[opt_cidx].shifts.size();

    for(int s=0;s<nshift;s++){
      int orig_cidx = -1;
      for(int i=0;i<clist_opt_map.size();i++) 
	if(clist_opt_map[i].first == opt_cidx && clist_opt_map[i].second == s){
	  orig_cidx = i;
	  break;
	}
      assert(orig_cidx != -1);

      int set = cidx_set_map[orig_cidx];

      if(set_first[set]){ //first time set acted upon, point to output
	for(int src=0;src<nsrc;src++){
	  out[src + nsrc*s] = &mf_sum[set][src];
	  distribute_queue[opt_cidx].push_back(&mf_sum[set][src]);
	  if(set_nrm[set] != 1.) norm_queue[opt_cidx].push_back(normInfo(&mf_sum[set][src], set_nrm[set]));
	}
	set_first[set] = false;
      }else{
	//Not first time this set has been acted on. Pointing into tmp
	if(tmpidx == mf_tmp.size()){
	  mf_tmp.resize(mf_tmp.size() + nsrc);
	  for(int i=0;i<nsrc;i++) mf_tmp[mf_tmp.size()-nsrc + i] = new storageType;
	}

	for(int src=0;src<nsrc;src++){
	  out[src + nsrc*s] = mf_tmp[tmpidx];
	  sum_queue[opt_cidx].push_back(sumInfo(&mf_sum[set][src], mf_tmp[tmpidx]));
	  if(set_nrm[set] != 1.) norm_queue[opt_cidx].push_back(normInfo(mf_tmp[tmpidx], set_nrm[set]));
	  ++tmpidx;
	}
      }
    }
    return out;
  }

  inline void postContractAction(const int opt_cidx){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    //Normalize either main or tmp mf. As main mf is only normalized once, just after generation, it will not yet have been distributed, 
    //hence we can skip the gather/dist
    std::vector<normInfo> &tonrm = norm_queue[opt_cidx];
    for(int i=0;i<tonrm.size();i++){
      storageType &to = *tonrm[i].to;
      double nrm = tonrm[i].nrm;
#ifndef MEMTEST_MODE
      for(int t=0;t<Lt;t++){
	to[t].times_equals(nrm);
      }
#endif
    }

    tonrm.clear();
    
    //For temp mf sum into the main mf
    std::vector<sumInfo> &tosum = sum_queue[opt_cidx];
    for(int i=0;i<tosum.size();i++){
      storageType &to = *tosum[i].to;
      storageType &from = *tosum[i].from;

      int gather_distribute = to[0].isOnNode() ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &gather_distribute, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      if(gather_distribute) nodeGetMany(1, &to);

#ifndef MEMTEST_MODE
      for(int t=0;t<Lt;t++){
	to[t].plus_equals(from[t]);
      }
#endif
      
      if(gather_distribute) nodeDistributeMany(1, &to);

    }
    tosum.clear();
  }

  const InnerProductType & getInnerProduct(const int opt_cidx){
    setSourceMomentum<SourceType>(opt_cidx);
    inner.setShifts(this->optimized_clist[opt_cidx].shifts);    
    return inner;
  }

  void nodeDistributeResult(const int opt_cidx){
    std::vector<storageType*> &to_dist = distribute_queue[opt_cidx];
    for(int s=0;s<to_dist.size();s++) nodeDistributeMany(1, to_dist[s]);
    to_dist.clear();
  }    
};



//Replace   mf[offset(0)] with   avg(mf[offset(0)], mf[offset(1)], mf[offset(2)]...., mf[offset(navg-1)])
//All averaged meson fields apart from mf[offset(0)] will have their memory freed
//Preserves distributed status of meson fields unless disable_redistribute = true, in which case it won't redistribute
template<typename StoreType, typename Indexer>
void stridedAverageFree(StoreType &mf_store, const Indexer &offset, const int navg, bool disable_redistribute = false){
  typedef typename StoreType::storageType storageType;
  storageType &mf_base = mf_store(offset(0));
    
#ifdef NODE_DISTRIBUTE_MESONFIELDS
  bool redistribute = false;
  if(!mesonFieldsOnNode(mf_base)){
    nodeGetMany(1,&mf_base); redistribute = true;
  }
#endif
  int Lt = mf_base.size();
    
  for(int i=1; i<navg; i++){
    storageType &mf_alt = mf_store(offset(i));
      
    for(int t=0;t<Lt;t++){
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      mf_alt[t].nodeGet(); //gather iff distributed
#endif	
#ifndef MEMTEST_MODE
      mf_base[t].plus_equals(mf_alt[t]);
#endif
      mf_alt[t].free_mem();
    }
  }
#ifndef MEMTEST_MODE
  for(int t=0;t<Lt;t++)
    mf_base[t].times_equals(1./navg);
#endif    

#ifdef NODE_DISTRIBUTE_MESONFIELDS
  if(!disable_redistribute) nodeDistributeMany(1, &mf_base);  
#endif
}  

CPS_END_NAMESPACE

#endif
