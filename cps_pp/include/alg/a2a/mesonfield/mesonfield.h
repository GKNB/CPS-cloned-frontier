#ifndef _CK_MESON_FIELD
#define _CK_MESON_FIELD

#include<set>
#include<util/time_cps.h>
#include<alg/a2a/utils.h>
#include<alg/a2a/base.h>
#include<alg/a2a/lattice.h>

#include "mesonfield_controls.h"
#include "mesonfield_distributed_storage.h"

#ifdef USE_GRID
#include<Grid/Grid.h>
#endif

CPS_START_NAMESPACE


//We have 2 different mode mappings in play in the high-mode part of the inputs: the fully unpacked dilution mapping used by Vfftw, and the time/flavor packed index  spin_color + 12*hit  used by Wfftw
//For the meson field we want to unpack W in flavor also, introducing a new mapping    sc + 12*( flav + nflav*hit)

//We allow for the timeslices associated with the left and right vectors of the outer product to differ
//In the majority of cases the timeslice of the right-hand vector is equal to that of the left-hand, but in some cases we might desire it to be different,
//for example when taking the product of [[W(t1)*V(t1)]] [[W(t2)*W(t2)]] -> [[W(t1)*W(t2)]]


template<typename mf_Policies, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
class A2AmesonField: public mf_Policies::MesonFieldDistributedStorageType{
public:
  //Deduce the dilution types for the meson field. We unpack the flavor index in W fields
  typedef typename A2AfieldL<mf_Policies>::DilutionType LeftInputDilutionType;
  typedef typename A2AfieldR<mf_Policies>::DilutionType RightInputDilutionType;

  typedef typename FlavorUnpacked<LeftInputDilutionType>::UnpackedType LeftDilutionType;
  typedef typename FlavorUnpacked<RightInputDilutionType>::UnpackedType RightDilutionType;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename mf_Policies::MesonFieldDistributedStorageType MesonFieldDistributedStorageType;
 private:
  int nmodes_l, nmodes_r;
  int fsize; //in units of ScalarComplexType

  LeftDilutionType lindexdilution;
  RightDilutionType rindexdilution;

  int tl, tr; //time coordinates associated with left and right fields of the outer-product

  template<typename, template <typename> class ,  template <typename> class >
  friend class A2AmesonField; //friend this class but with other field types

public:
  A2AmesonField(): fsize(0), nmodes_l(0), nmodes_r(0), MesonFieldDistributedStorageType(){ }

  //Just setup memory (setup is automatically called when 'compute' is called, so this is not necessary. However if you disable the setup at compute time you should setup the memory beforehand)
  A2AmesonField(const A2AfieldL<mf_Policies> &l, const A2AfieldR<mf_Policies> &r): fsize(0), nmodes_l(0), nmodes_r(0), MesonFieldDistributedStorageType(){
    setup(l,r,-1,-1);
  }

  A2AmesonField(const A2AmesonField &r): nmodes_l(r.nmodes_l), nmodes_r(r.nmodes_r),
					 fsize(r.fsize), lindexdilution(r.lindexdilution), rindexdilution(r.rindexdilution),
					 tl(r.tl), tr(r.tr), MesonFieldDistributedStorageType(r){ }


  class View{
    int nmodes_l, nmodes_r;
    int tl, tr;

    A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> const* parent; //for use in host-side calls only!
   
    int fsize; //in units of ScalarComplexType
    ScalarComplexType *data;
    typename mf_Policies::MesonFieldDistributedStorageType::AllocView alloc_view;
    
  public:
    typedef mf_Policies Policies;
    typedef typename A2AfieldL<mf_Policies>::DilutionType LeftInputDilutionType;
    typedef typename A2AfieldR<mf_Policies>::DilutionType RightInputDilutionType;
    
    typedef typename FlavorUnpacked<LeftInputDilutionType>::UnpackedType LeftDilutionType;
    typedef typename FlavorUnpacked<RightInputDilutionType>::UnpackedType RightDilutionType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    //Size in complex
    accelerator_inline int size() const{ return fsize; }

    //Access elements with compressed mode index
    accelerator_inline const ScalarComplexType & operator()(const int i, const int j) const{
      return data[j + nmodes_r*i];
    }

    accelerator_inline int getRowTimeslice() const{ return tl; }
    accelerator_inline int getColTimeslice() const{ return tr; }
    
    //These functions return the number of *packed* modes not the full number of modes
    accelerator_inline int getNrows() const{ return nmodes_l; }
    accelerator_inline int getNcols() const{ return nmodes_r; }

    View(ViewMode mode, const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &mf);
    
    accelerator_inline ScalarComplexType* ptr(){ return (ScalarComplexType*)data; }
    accelerator_inline ScalarComplexType const* ptr() const{ return (ScalarComplexType const*)data; }

    //Access elements with compressed mode index
    accelerator_inline ScalarComplexType & operator()(const int i, const int j){ //Use at your own risk
      return data[j + nmodes_r*i]; //right mode index changes most quickly
    }
  
    //A slow implementation to access elements from full unpacked indices
    inline const ScalarComplexType & elem(const int full_i, const int full_j) const{
      StaticAssert< _equal<LeftDilutionType,StandardIndexDilution>::value || _equal<LeftDilutionType,TimePackedIndexDilution>::value >();
      StaticAssert< _equal<RightDilutionType,StandardIndexDilution>::value || _equal<RightDilutionType,TimePackedIndexDilution>::value >();
    
      static ScalarComplexType zero(0.0,0.0);

      int nll = parent->lindexdilution.getNl();
      int nlr = parent->rindexdilution.getNl();
      int tblockl = parent->lindexdilution.tblock(tl);
      int tblockr = parent->rindexdilution.tblock(tr);

      int packed_i;
      if(_equal<LeftDilutionType,StandardIndexDilution>::value || full_i < nll) packed_i = full_i; //  lindexdilution.getModeType() == StandardIndex
      else{ // W *
	StandardIndexDilution lfulldil(parent->lindexdilution);
	modeIndexSet i_idx; lfulldil.indexUnmap(full_i-nll, i_idx);
	if(i_idx.time != tblockl) return zero; //delta function in time
	else packed_i = nll + parent->lindexdilution.indexMap(i_idx);
      }
      int packed_j;
      if(_equal<RightDilutionType,StandardIndexDilution>::value || full_j < nlr) packed_j = full_j; //rindexdilution.getModeType() == StandardIndex
      else{ //* W
	StandardIndexDilution rfulldil(parent->rindexdilution);
	modeIndexSet j_idx; rfulldil.indexUnmap(full_j-nlr, j_idx);
	if(j_idx.time != tblockr) return zero;
	else packed_j = nlr + parent->rindexdilution.indexMap(j_idx);
      }
      return this->operator()(packed_i,packed_j);
    }

    //Version of the above that returns a pointer so that the element can be modified. A NULL pointer will be returned for elements that are enforced to be zero by the index packing
    inline ScalarComplexType* elem_ptr(const int full_i, const int full_j){
      StaticAssert< _equal<LeftDilutionType,StandardIndexDilution>::value || _equal<LeftDilutionType,TimePackedIndexDilution>::value >();
      StaticAssert< _equal<RightDilutionType,StandardIndexDilution>::value || _equal<RightDilutionType,TimePackedIndexDilution>::value >();
    
      int nll = parent->lindexdilution.getNl();
      int nlr = parent->rindexdilution.getNl();
      int tblockl = parent->lindexdilution.tblock(tl);
      int tblockr = parent->rindexdilution.tblock(tr);

      int packed_i;
      if(_equal<LeftDilutionType,StandardIndexDilution>::value || full_i < nll) packed_i = full_i; //  lindexdilution.getModeType() == StandardIndex
      else{ // W *
	StandardIndexDilution lfulldil(parent->lindexdilution);
	modeIndexSet i_idx; lfulldil.indexUnmap(full_i-nll, i_idx);
	if(i_idx.time != tblockl) return NULL; //delta function in time
	else packed_i = nll + parent->lindexdilution.indexMap(i_idx);
      }
      int packed_j;
      if(_equal<RightDilutionType,StandardIndexDilution>::value || full_j < nlr) packed_j = full_j; //rindexdilution.getModeType() == StandardIndex
      else{ //* W
	StandardIndexDilution rfulldil(parent->rindexdilution);
	modeIndexSet j_idx; rfulldil.indexUnmap(full_j-nlr, j_idx);
	if(j_idx.time != tblockr) return NULL;
	else packed_j = nlr + parent->rindexdilution.indexMap(j_idx);
      }
      return &this->operator()(packed_i,packed_j);
    }


    void free();
  };

  //Accepts HostRead, HostWrite, HostReadWrite, DeviceRead,   *NOT* DeviceWrite
  inline View view(ViewMode mode) const{ return View(mode, *this); }


  
  //Call this when you use the default constructor if not automatically called (it is called automatically in ::compute)
  void setup(const A2Aparams &lp, const A2Aparams &rp, const int _tl, const int _tr){
    tl = _tl; tr = _tr;
    lindexdilution = lp; rindexdilution = rp;

    nmodes_l = lindexdilution.getNmodes();
    nmodes_r = rindexdilution.getNmodes();

    int old_fsize = fsize;
    fsize = nmodes_l*nmodes_r;

    if(this->data() != NULL && old_fsize != fsize ){
      this->freeMem();
      this->alloc(128, fsize * sizeof(ScalarComplexType)); 
    }else if(this->data() == NULL){
      this->alloc(128, fsize * sizeof(ScalarComplexType)); 
    }
    zero();
  }

  //Return size in bytes
  size_t byte_size() const{ return fsize * sizeof(ScalarComplexType); }

  //Can compute the byte size without an instance as long as we know the params
  static size_t byte_size(const A2Aparams &lp, const A2Aparams &rp){
    LeftDilutionType lindexdilution = lp;
    RightDilutionType rindexdilution = rp;
    int nmodes_l = lindexdilution.getNmodes();
    int nmodes_r = rindexdilution.getNmodes();
    size_t fsize = nmodes_l*nmodes_r;
    return fsize * sizeof(ScalarComplexType);
  }

  void free_mem(){
    this->freeMem();
  }

  bool equals(const A2AmesonField &r, const double tolerance = 1e-10, bool verbose = false) const{
    if(r.nmodes_l != nmodes_l || r.nmodes_r != nmodes_r) return false;
    
    CPSautoView(t_v,(*this),HostRead);
    CPSautoView(r_v,r,HostRead);
    
    for(int i=0;i<nmodes_l;i++){
      for(int j=0;j<nmodes_r;j++){
	const ScalarComplexType &lval = t_v(i,j);
	const ScalarComplexType &rval = r_v(i,j);
	
	if( fabs(lval.real() - rval.real()) > tolerance || fabs(lval.imag() - rval.imag()) > tolerance ){
	  if(verbose && !UniqueID()){
	    printf("Err: (%d,%d) : this[%g,%g] vs that[%g,%g] : diff [%g,%g]\n",i,j,
		   lval.real(),lval.imag(),rval.real(),rval.imag(),fabs(lval.real()-rval.real()), fabs(lval.imag()-rval.imag()) ); fflush(stdout);
	  }
	  return false;
	}
      }
    }
    return true;
  }    
  
  A2AmesonField &operator=(const A2AmesonField &r){    
    nmodes_l = r.nmodes_l; nmodes_r = r.nmodes_r; fsize = r.fsize;
    lindexdilution = r.lindexdilution;  rindexdilution = r.rindexdilution;
    tl = r.tl; tr = r.tr;
    ((MesonFieldDistributedStorageType*)this)->operator=(r);
    return *this;
  }

  void move(A2AmesonField &from){
    nmodes_l = from.nmodes_l; nmodes_r = from.nmodes_r; fsize = from.fsize;
    lindexdilution = from.lindexdilution; rindexdilution = from.rindexdilution; 
    tl = from.tl; tr = from.tr;
    ((MesonFieldDistributedStorageType*)this)->move(from);
  }
  
  //Size in complex
  inline int size() const{ return fsize; }

  inline double norm2() const{
    CPSautoView(t_v,(*this),HostRead);
    double out = 0.;
    for(int i=0;i<size();i++) out += norm(t_v.ptr()[i]);
    return out;
  }
  
  inline int getRowTimeslice() const{ return tl; }
  inline int getColTimeslice() const{ return tr; } 

  //Convert the meson field from the packed default format into an unpacked format
  //(i,j) element of 'into' =  j + getNcolsFull() * i
  //size of 'into' = getNrowsFull()*getNcolsFull()
  void unpack(ScalarComplexType* into) const;  

  //Convert the meson field from the packed default format into an unpacked format *on the device*
  //If a view is not provided it will be created internally and the host->device copy performed
  //(i,j) element of 'into' =  j + getNcolsFull() * i
  //size of 'into' = getNrowsFull()*getNcolsFull()
  //** into must be allocated on the device! **
  void unpack_device(ScalarComplexType* into, View const* view = nullptr) const;
  
  //Convert the meson field from the unpacked format into the packed format
  //(i,j) element of 'from' =  j + getNcolsFull() * i
  //size of 'from' = getNrowsFull()*getNcolsFull()
  //Note: the left and right timeslices of "this" must be set to match the time row/column that is non zero (if appropriate)  
  void pack(ScalarComplexType const* from);

  //Convert the meson field from the unpacked format into the packed format *on the device*
  //The packing is performed on the device and the result is copied to the host
  //(i,j) element of 'from' =  j + getNcolsFull() * i
  //size of 'from' = getNrowsFull()*getNcolsFull()
  //Note: the left and right timeslices of "this" must be set to match the time row/column that is non zero (if appropriate)
  //** from must be allocated on the device! **
  void pack_device(ScalarComplexType const* from);
  
  inline void zero(const bool parallel = true){
    CPSautoView(t_v,(*this),HostWrite);
    memset(t_v.ptr(), 0, sizeof(ScalarComplexType) * fsize);      
  }
  //For all mode indices l_i and r_j, compute the meson field  V^-1 \sum_p l_i^\dagger(p,t) X(p,t) r_j(p,t)
  //It is assumed that A2AfieldL and A2AfieldR are Fourier transformed field containers
  //X(p,t) is a completely general momentum-space spin/color/flavor matrix per temporal slice
  //do_setup allows you to disable the reassignment of the memory (it will still reset to zero). Use wisely!
  //InnerProduct M must accumulate into ScalarComplexType
  template<typename InnerProduct>
  void compute(const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r,const int &t, bool do_setup = true);

  //This version is more efficient on multi-nodes
  template<typename InnerProduct, typename Allocator>
  static void compute(std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator > &mf_t, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup = true);

  //Version of the above for multi-src inner products (output vector indexed by [src idx][t]
  template<typename InnerProduct, typename Allocator>
  static void compute(std::vector< std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>, Allocator >* > &mf_st, const A2AfieldL<mf_Policies> &l, const InnerProduct &M, const A2AfieldR<mf_Policies> &r, bool do_setup = true);

  //These functions return the number of *packed* modes not the full number of modes
  inline int getNrows() const{ return nmodes_l; }
  inline int getNcols() const{ return nmodes_r; }

  //These functions return the number of *full* modes
  inline int getNrowsFull() const{ return lindexdilution.getNv(); }
  inline int getNcolsFull() const{ return rindexdilution.getNv(); }

  //Return the full set of dilution parameters
  inline const LeftDilutionType & getRowParams() const{ return lindexdilution; }
  inline const RightDilutionType & getColParams() const{ return rindexdilution; }
  
  void plus_equals(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel = true);
  void times_equals(const ScalarComplexType f, const bool parallel = true);

  //Replace this meson field with the average of this and a second field, 'with'
  void average(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &with, const bool parallel = true);

  //Set each float to a uniform random number in the specified range
  //Uses a fixed-seed uniform RNG that every node has an identical copy of
  void testRandom(const Float hi=0.5, const Float lo=-0.5);

  //Reorder the rows so that all the elements in idx_map are sequential. Indices not in map may be written over. Use at your own risk
  void rowReorder(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel = true) const;
  void colReorder(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel = true) const;

#ifdef USE_GRID
  //Do a column reorder but where we pack the row indices to exclude those not used (as indicated by input bool array)
  //Output to a linearized matrix of Grid SIMD vectors where we have splatted the scalar onto all SIMD lanes
  //Option not to resize the output vector, allowing reuse of a previously allocated vector providing it's large enough
  void splatPackedColReorder(typename AlignedVector<typename mf_Policies::ComplexType>::type &into, const int idx_map[], int map_size, bool rowidx_used[], bool do_resize = true) const;
  void scalarPackedColReorder(typename AlignedVector<typename mf_Policies::ComplexType>::type &into, const int idx_map[], int map_size, bool rowidx_used[], bool do_resize = true) const;
#endif
  
  //Transpose the meson field! (parallel)
  void transpose(A2AmesonField<mf_Policies,A2AfieldR,A2AfieldL> &into) const;

  //Take the complex conjugate of the meson field (parallel)
  void conj(A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &into) const;

  //Take the hermitian conjugate of the meson field (parallel)
  void hconj(A2AmesonField<mf_Policies,A2AfieldR,A2AfieldL> &into) const;

  //Delete all the data associated with this meson field apart from on node with UniqueID 'node'. The node index is saved so that the data can be later retrieved.
  //The memory will be distributed according to a global index that cycles between 0... nodes-1 (with looping) to ensure even distribution
  void nodeDistribute();
  //Get back the data. After the call, all nodes will have a complete copy
  //All nodes must call nodeGet simultaneously. If 'require' is false the comms will be performed but the data will not be kept on this node
  void nodeGet(bool require = true);

  void write(std::ostream *file_ptr, FP_FORMAT fileformat = FP_AUTOMATIC) const;
  void write(const std::string &filename, FP_FORMAT fileformat = FP_AUTOMATIC) const;
  void read(std::istream *file_ptr); //istream pointer should only be open on node 0 - should be NULL otherwise
  void read(const std::string &filename);

  static void write(const std::string &filename, const std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs, FP_FORMAT fileformat = FP_AUTOMATIC);
  static void write(std::ostream *file_ptr, const std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs, FP_FORMAT fileformat = FP_AUTOMATIC);
  static void read(const std::string &filename, std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs);
  static void read(std::istream *file_ptr, std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &mfs);

  void nodeSum(){ //don't call unless you know what you're doing
    CPSautoView(t_v,(*this),HostReadWrite);
    globalSum( (typename ScalarComplexType::value_type*)t_v.ptr(),2*fsize);
  }
};



#include "implementation/mesonfield_io.tcc"
#include "implementation/mesonfield_compute_impl.tcc"
#include "implementation/mesonfield_impl.tcc"

CPS_END_NAMESPACE





#endif
