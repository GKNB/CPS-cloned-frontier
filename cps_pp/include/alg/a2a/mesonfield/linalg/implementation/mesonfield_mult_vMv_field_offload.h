#ifndef _MULT_VMV_FIELD_OFFLOAD_H_
#define _MULT_VMV_FIELD_OFFLOAD_H_

#include<set>

//Not currently defined in Grid for CPU targets
#ifndef accelerator_for2dNB
#define accelerator_for2dNB(iter1, num1, iter2, num2, nsimd, ... ) thread_for2d(iter1,num1,iter2,num2,{ __VA_ARGS__ });
#endif

CPS_START_NAMESPACE

template<typename mf_Policies,
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR,
	 typename ComplexClass>
class _mult_vMv_field_offload_v{};

template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
using mult_vMv_field = _mult_vMv_field_offload_v<mf_Policies, lA2AfieldL, lA2AfieldR, rA2AfieldL, rA2AfieldR, typename ComplexClassify<typename mf_Policies::ComplexType>::type >;


#ifdef USE_GRID

template<typename mf_Policies, int isGparity>
struct _mult_vMv_field_offload_fields{};

template<typename mf_Policies>
struct _mult_vMv_field_offload_fields<mf_Policies,1>{
  typedef CPSspinColorFlavorMatrix<typename mf_Policies::ComplexType> VectorMatrixType;
  typedef CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy> PropagatorField;
  static accelerator_inline typename mf_Policies::ComplexType & access(const int s1, const int c1, const int f1,
							   const int s2, const int c2, const int f2,
							   VectorMatrixType &M){
    return M(s1,s2)(c1,c2)(f1,f2);
  }
    
};
template<typename mf_Policies>
struct _mult_vMv_field_offload_fields<mf_Policies,0>{
  typedef CPSspinMatrix<CPScolorMatrix<typename mf_Policies::ComplexType> > VectorMatrixType;
  typedef CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy> PropagatorField;
  static accelerator_inline typename mf_Policies::ComplexType & access(const int s1, const int c1, const int f1,
							   const int s2, const int c2, const int f2,
							   VectorMatrixType &M){
    return M(s1,s2)(c1,c2);
  }
};

struct mult_vMv_field_offload_timers{
  struct timers{
    double init;
    double vaprime;
    double Mprime;
    double lMr;
    size_t calls;

    timers(): init(0), vaprime(0), Mprime(0), lMr(0), calls(0){}

    void reset(){
      init=vaprime=Mprime=lMr=0;
      calls = 0;
    }
    void average(){
      init/=calls;
      vaprime/=calls;
      lMr/=calls;
    }
    void print(){
      average();
      printf("calls=%zu init=%g va'=%g M'=%g lMr=%g\n", calls, init, vaprime, Mprime, lMr);
    }

  };
  static timers & get(){ static timers t; return t; }
};


//For A,B,C... \in { A2AvectorW, A2AvectorV, A2AvectorWfftw, A2AvectorVfftw }
//Compute   A (BC) D    where  (BC) is a meson field, A, D are A2A vectors
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
struct _mult_vMv_field_offload_v<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,grid_vector_complex_mark>{
  typedef _mult_vMv_field_offload_fields<mf_Policies, mf_Policies::GPARITY> fdef;
  typedef typename fdef::VectorMatrixType VectorMatrixType;
  typedef typename fdef::PropagatorField PropagatorField;

  typedef lA2AfieldL<mf_Policies> lA2AfieldType;
  typedef rA2AfieldR<mf_Policies> rA2AfieldType;
  typedef A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> MesonFieldType;

  typedef typename lA2AfieldType::DilutionType iLeftDilutionType;
  typedef typename MesonFieldType::LeftDilutionType iRightDilutionType;
  
  typedef typename MesonFieldType::RightDilutionType jLeftDilutionType;    
  typedef typename rA2AfieldType::DilutionType jRightDilutionType;

  typedef typename mf_Policies::ComplexType VectorComplexType;
  typedef typename SIMT<VectorComplexType>::value_type ScalarComplexType;

  //A slow but simple implementation ignoring the index compression
  static void simple(PropagatorField &into,
		     const lA2AfieldType &l,
		     const MesonFieldType &M,
		     const rA2AfieldType &r,
		     bool conj_l, bool conj_r){
    into.zero();

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);
    
    A2Aparams i_params(l), j_params(r);
    StandardIndexDilution idil(i_params), jdil(j_params);
    
    int ni = idil.getNmodes();
    int nj = jdil.getNmodes();
    
    size_t nsimd = VectorComplexType::Nsimd();
    
    size_t vol4d = into.size();

    int nf = GJP.Gparity() ? 2:1;

    typedef SIMT<VectorComplexType> ACC;

    using namespace Grid;
    thread_for(x4d, vol4d,
		    {
		      VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d);
		      size_t xop; int top;
		      into.fourToThree(xop, top, x4d);

		      for(int sl=0;sl<4;sl++){
			for(int cl=0;cl<3;cl++){
			  for(int fl=0;fl<nf;fl++){	  
			    
			    for(int sr=0;sr<4;sr++){
			      for(int cr=0;cr<3;cr++){
				for(int fr=0;fr<nf;fr++){
				  VectorComplexType &out = vsite_mat(sl,sr)(cl,cr)(fl,fr);

				  for(int i=0;i<ni;i++){
				    
				    ScalarComplexType lval_tmp = ACC::read(l.elem(i,xop,top,cl+3*sl,fl));
				    ScalarComplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
		  
				    for(int j=0;j<nj;j++){
				      ScalarComplexType rval_tmp = ACC::read(r.elem(j,xop,top,cr+3*sr,fr));
				      ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
				      
				      ScalarComplexType Mval = M.elem(i,j);

				      ScalarComplexType val = ACC::read(out) + lval * Mval * rval;
				      ACC::write(out, val);
				    }
				  }
				}
			      }
			    }
			  }
			}
		      }
		    }
		    );
    
  }


  template<typename T>
  static accelerator_inline T min_value(const T&a, const T&b){ return a < b ? a : b; }

  static void setup_index_maps(ManagedVector< std::pair<int,int> > &il_ir_pairs, ManagedVector< std::pair<int,int> > &jl_jr_pairs,
			       std::map<std::pair<int,int>, int> &il_ir_pairs_index_map, std::map<std::pair<int,int>, int> &jl_jr_pairs_index_map,
			       const MesonFieldType &M, const int nf, int ntblocks,
			       const ModeContractionIndices<iLeftDilutionType,iRightDilutionType> &i_ind,
			       const ModeContractionIndices<jLeftDilutionType,jRightDilutionType> &j_ind){
			       
    std::set<std::pair<int,int> > il_ir_pairs_s;
    std::set<std::pair<int,int> > jl_jr_pairs_s;

    {
      modeIndexSet ilp, irp, jlp, jrp;
      irp.time = M.getRowTimeslice();
      jlp.time = M.getColTimeslice();
      for(int tv=0;tv<ntblocks;tv++){
	ilp.time = jrp.time = tv;
	for(int f=0;f<nf;f++){
	  ilp.flavor = irp.flavor = jlp.flavor = jrp.flavor = f;
	  for(int sc=0;sc<12;sc++){
	    ilp.spin_color = irp.spin_color = jlp.spin_color = jrp.spin_color = sc;
	
	    const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
	    for(int i=0;i<i_ind_pairs.size();i++) il_ir_pairs_s.insert(i_ind_pairs[i]);	    

	    const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
	    for(int j=0;j<j_ind_pairs.size();j++) jl_jr_pairs_s.insert(j_ind_pairs[j]);
	  }
	}
      }
    }
    
    il_ir_pairs.resize(il_ir_pairs_s.size());
    int ii=0;
    for(auto it=il_ir_pairs_s.begin(); it != il_ir_pairs_s.end(); it++){
      il_ir_pairs[ii] = *it;
      il_ir_pairs_index_map[*it] = ii;
      ++ii;
    }
    jl_jr_pairs.resize(jl_jr_pairs_s.size());
    ii=0;
    for(auto it=jl_jr_pairs_s.begin(); it != jl_jr_pairs_s.end(); it++){
      jl_jr_pairs[ii] = *it;
      jl_jr_pairs_index_map[*it] = ii;
      ++ii;
    }
  }
  
  static void setup_alpha_beta(ManagedVector<uint8_t> &alpha, ManagedVector<uint8_t> &beta,
			       const MesonFieldType &M, const int nf, const int ntblocks,
			       const ModeContractionIndices<iLeftDilutionType,iRightDilutionType> &i_ind,
			       const ModeContractionIndices<jLeftDilutionType,jRightDilutionType> &j_ind,
			       std::map<std::pair<int,int>, int> &il_ir_pairs_index_map, std::map<std::pair<int,int>, int> &jl_jr_pairs_index_map){
    modeIndexSet ilp, irp, jlp, jrp;
    irp.time = M.getRowParams().tblock(M.getRowTimeslice());
    jlp.time = M.getColParams().tblock(M.getColTimeslice());
    for(int tv=0;tv<ntblocks;tv++){
      ilp.time = jrp.time = tv;
      for(int f=0;f<nf;f++){
	ilp.flavor = irp.flavor = jlp.flavor = jrp.flavor = f;
	for(int sc=0;sc<12;sc++){
	  ilp.spin_color = irp.spin_color = jlp.spin_color = jrp.spin_color = sc;

	  const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
	  for(int i=0;i<i_ind_pairs.size();i++){
	    const std::pair<int, int> &pair = i_ind_pairs[i];
	    int pair_idx = il_ir_pairs_index_map[pair];
	    alpha[sc + 12*(f+ nf*(tv + ntblocks*pair_idx))] = 1;
	  }
	  const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
	  for(int j=0;j<j_ind_pairs.size();j++){
	    const std::pair<int, int> &pair = j_ind_pairs[j];
	    int pair_idx = jl_jr_pairs_index_map[pair];
	    beta[sc + 12*(f+ nf*(tv + ntblocks*pair_idx))] = 1;
	  }
	}
      }
    }
  }  
  
  //Create va'
  static void create_vaprime(VectorComplexType* vaprime, typename PropagatorField::View &into_v, typename lA2AfieldType::View &l_v,
			     ManagedVector<uint8_t>::View &alpha_v, ManagedVector< std::pair<int,int> >::View &il_ir_pairs_v,
			     int t_off, int src_width, size_t iprimestart, int nf, int ntblocks, size_t vol3d_node, hostDeviceMirroredContainer<int> &local_timeslices, size_t niprime_block, size_t nsimd, bool conj_l){
    size_t nt = local_timeslices.size();
    size_t nsites4d = vol3d_node * nt;
    int const* local_timeslices_v = local_timeslices.getDeviceReadPtr();
    typedef SIMT<VectorComplexType> ACC;
    using namespace Grid;
    accelerator_for2d(xx, nsites4d, iprimeb, niprime_block, nsimd,
		      {
			size_t xop = xx % vol3d_node;
			int top = local_timeslices_v[xx / vol3d_node];
			size_t x4d = xop + vol3d_node * top; //actual local 4d offset
			int t_glob = top + t_off; 
			int t_glob_block = t_glob / src_width; //time block for a2a index
			size_t iprime = iprimeb + iprimestart;
			for(int f=0;f<nf;f++){
			  for(int sc=0;sc<12;sc++){
			    VectorComplexType *into = vaprime +  iprimeb + niprime_block*( xx + nsites4d*(sc + 12*f) ); //contiguous in summed index
			    
			    auto val = ACC::read(l_v.nativeElem(il_ir_pairs_v[iprime].first, x4d, sc, f));
			    val = conj_l ? Grid::conjugate(val) : val;
			    val = val * double(alpha_v[sc + 12*(f+ nf*(t_glob_block + ntblocks*iprime))]);
			    ACC::write(*into, val);
			  }
			}
		      });
    
  }



  
  //Create Mprime
  static void create_Mprime(typename MesonFieldType::ScalarComplexType *Mprime, const MesonFieldType &M,
			    size_t iprimestart, size_t iprimelessthan, size_t jprimestart, size_t jprimelessthan,
			    ManagedVector< std::pair<int,int> > &il_ir_pairs, ManagedVector< std::pair<int,int> > &jl_jr_pairs){
    typedef typename MesonFieldType::ScalarComplexType MFcomplexType;
    
    MFcomplexType *Mptr = Mprime;
    for(size_t iprime = iprimestart; iprime < iprimelessthan; iprime++){
      size_t iprimeb = iprime - iprimestart;
      size_t ir = il_ir_pairs[iprime].second;
      for(size_t jprime = jprimestart; jprime < jprimelessthan; jprime++){
	size_t jprimeb = jprime - jprimestart;
	size_t jl = jl_jr_pairs[jprime].first;
	*Mptr++ = M(ir, jl);
      }
    }
  }    
  
  static void create_vbprime(VectorComplexType* vbprime, typename PropagatorField::View &into_v, typename rA2AfieldType::View &r_v,
			     ManagedVector<uint8_t>::View &beta_v, ManagedVector< std::pair<int,int> >::View &jl_jr_pairs_v,
			     int t_off, int src_width, size_t jprimestart, int nf, int ntblocks, size_t vol3d_node, hostDeviceMirroredContainer<int> &local_timeslices,
			     size_t njprime_block, size_t nsimd, bool conj_r,
			     int scr, int fr){
    size_t nt = local_timeslices.size();
    size_t nsites4d = vol3d_node * nt;
    int const* local_timeslices_v = local_timeslices.getDeviceReadPtr();
    typedef SIMT<VectorComplexType> ACC;
    using namespace Grid;
    accelerator_for2dNB(xx, nsites4d, jprimeb, njprime_block, nsimd,
			{
			  size_t jprime = jprimeb + jprimestart;
			  size_t xop = xx % vol3d_node;
			  int top = local_timeslices_v[xx / vol3d_node];
			  size_t x4d = xop + vol3d_node * top; //actual local 4d offset
			  int t_glob = top + t_off;
			  int t_glob_block = t_glob / src_width; //tblock for a2a index
			  
			  VectorComplexType *into = vbprime + jprimeb + njprime_block*xx;  //contiguous in summed index
			  auto val = ACC::read(r_v.nativeElem(jl_jr_pairs_v[jprime].second, x4d, scr, fr));
			  val = conj_r ? Grid::conjugate(val) : val;
			  val = val * double(beta_v[scr + 12*(fr+ nf*(t_glob_block + ntblocks*jprime))]);
			  ACC::write(*into, val);
			});
  }



  
  //Mprime * vbprime
  static void Mprime_vbprime(VectorComplexType* Mvbprime, typename MesonFieldType::ScalarComplexType *Mprime, VectorComplexType* vbprime, 
			     size_t vol4d_node, size_t niprime_block, size_t njprime_block, size_t nsimd, bool verbose = false){
    typedef SIMT<VectorComplexType> ACC;
    typedef typename MesonFieldType::ScalarComplexType MFcomplexType;
    using namespace Grid;
#ifdef GRID_CUDA
    uint32_t orig_t = Grid::acceleratorThreads();
    Grid::acceleratorThreads(16);
    //std::cout << "Changed number of GPU threads from " << orig_t << " to " << Grid::acceleratorThreads() << std::endl;
#endif

    int nxblocks = BlockedvMvOffloadArgs::bb;
    assert(vol4d_node % nxblocks == 0);
    size_t xblocksz = vol4d_node / nxblocks;
    
    double block_size_MB = double(njprime_block*xblocksz*sizeof(VectorComplexType))/1024./1024.;
    double Mprime_size_MB = double(niprime_block*njprime_block*sizeof(MFcomplexType))/1024./1024.;
    if(verbose) std::cout << "Solving M'* vb' with nxblocks=" << nxblocks << ": block volume " << xblocksz << " = " << block_size_MB << " MB and M' size " << niprime_block << "*" << njprime_block << " = " << Mprime_size_MB << " MB" << std::endl;
    
    accelerator_for2dNB(x4d_block, xblocksz, iprimeb_xb, niprime_block*nxblocks, nsimd,
			{
			  size_t iprimeb = iprimeb_xb % niprime_block;
			  int xblock_idx = iprimeb_xb / niprime_block;
			  size_t x4d = x4d_block + xblocksz*xblock_idx;		  
			  VectorComplexType *into = Mvbprime + iprimeb + niprime_block*x4d;
			  
			  typename ACC::value_type sum(0);
			  VectorComplexType *rptr = vbprime + njprime_block*x4d; //jprimeb=0
			  MFcomplexType *Mptr = Mprime + njprime_block * iprimeb; //jprimeb=0
			  
			  for(int jprimeb=0;jprimeb<njprime_block; jprimeb++){
			    auto rval = ACC::read(*rptr++);
			    auto Mval = *Mptr++; //not vectorized
			    sum = sum + Mval * rval;
			  }
			  ACC::write(*into, sum);				      
			});
    
#ifdef GRID_CUDA    
    Grid::acceleratorThreads(orig_t);
#endif
  }
  
  //va' (M' vb')
  static void vaprime_Mprime_vbprime(typename PropagatorField::View &into_v, VectorComplexType* vaprime, VectorComplexType* Mvbprime, int sr, int cr, int fr, size_t vol3d_node, hostDeviceMirroredContainer<int> &local_timeslices, size_t niprime_block, int nf, size_t nsimd){
    size_t nt = local_timeslices.size();
    size_t nsites4d = vol3d_node * nt;
    int const* local_timeslices_v = local_timeslices.getDeviceReadPtr();
    typedef SIMT<VectorComplexType> ACC;
    using namespace Grid;
    accelerator_for2dNB(xx, nsites4d, scfl, 12*nf, nsimd,
			{
			  size_t xop = xx % vol3d_node;
			  int top = local_timeslices_v[xx / vol3d_node];
			  size_t x4d = xop + vol3d_node * top; //actual local 4d offset
			  
			  VectorMatrixType &vsite_mat = *into_v.fsite_ptr(x4d);
			  int scl = scfl % 12;
			  int fl = scfl / 12;
			  int cl = scl % 3;
			  int sl = scl / 3;			    

			  VectorComplexType &out = fdef::access(sl,cl,fl, sr,cr,fr, vsite_mat);
			  auto sum = ACC::read(out);
			  
			  VectorComplexType *lptr = vaprime + niprime_block*( xx + nsites4d*scfl );		  
			  VectorComplexType *Mrptr = Mvbprime + niprime_block*xx;
			  
			  for(size_t iprimeb=0; iprimeb < niprime_block; iprimeb++){
			    auto lval = ACC::read(*lptr++); 
			    auto Mrval = ACC::read(*Mrptr++);
			    sum = sum + lval * Mrval;
			  }
			  ACC::write(out, sum);
			});
  }


  
  static void prefetch_r(size_t jprimeblock, size_t njprime_blocks, size_t iprimeblock, size_t niprime_blocks,
			 size_t blocksize, size_t njprime, ManagedVector< std::pair<int,int> > &jl_jr_pairs, const rA2AfieldType &r,
			 size_t vol3d_node, hostDeviceMirroredContainer<int> &local_timeslices){
    size_t nt = local_timeslices.size();
    int const* local_timeslices_v = local_timeslices.getHostReadPtr();
    
#ifdef GRID_CUDA
    int device;
    assert(cudaGetDevice(&device) == cudaSuccess);
    
    if(jprimeblock < njprime_blocks-1 || (jprimeblock == njprime_blocks-1 && iprimeblock != niprime_blocks-1)  ){
      size_t jprimeblock_nxt = (jprimeblock+1) % njprime_blocks; //loops back to 0 on last iteration so as to prefetch memory for next iblock
      size_t jprimestart_nxt = jprimeblock_nxt * blocksize;
      size_t jprimelessthan_nxt = std::min(jprimestart_nxt + blocksize, njprime);
      size_t njprime_block_nxt = jprimelessthan_nxt - jprimestart_nxt;
      
      for(size_t jprimeb = 0 ; jprimeb < njprime_block_nxt; jprimeb++){
	size_t jprime = jprimeb + jprimestart_nxt;
	size_t jr = jl_jr_pairs[jprime].second;
	for(int tt=0;tt<nt;tt++){
	  int t = local_timeslices_v[tt];	  
	  VectorComplexType const* v0;
	  VectorComplexType const* v1;
	  size_t sz;
	  r.getModeTimesliceData(v0,v1,sz,jr,t);
	  assert( cudaMemPrefetchAsync( (void const*)v0, sz * sizeof(VectorComplexType), device, Grid::copyStream ) == cudaSuccess );
	  if(GJP.Gparity()) assert( cudaMemPrefetchAsync( (void const*)v1, sz * sizeof(VectorComplexType), device, Grid::copyStream ) == cudaSuccess );
	}
      }
    }
#endif
  }

  static void optimized(PropagatorField &into,
			const lA2AfieldType &l,
			const MesonFieldType &M,
			const rA2AfieldType &r,
			bool conj_l, bool conj_r,
			const int t_start, const int t_end){
    if(!UniqueID()) std::cout << "Starting field vMv multiplication between t=" << t_start << " and " << t_end << std::endl;
#ifdef GRID_CUDA
    cudaFuncCache cache_default;
    assert(cudaDeviceGetCacheConfig(&cache_default) == cudaSuccess );
    assert(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) == cudaSuccess );

    for(int i=0;i<l.getNlowModes();i++) l.getLowMode(i).deviceSetAdviseUVMreadOnly(true);
    for(int i=0;i<l.getNhighModes();i++) l.getHighMode(i).deviceSetAdviseUVMreadOnly(true);
    for(int i=0;i<r.getNlowModes();i++) r.getLowMode(i).deviceSetAdviseUVMreadOnly(true);
    for(int i=0;i<r.getNhighModes();i++) r.getHighMode(i).deviceSetAdviseUVMreadOnly(true);
    
#endif   
    
    mult_vMv_field_offload_timers::timers &time = mult_vMv_field_offload_timers::get();

    ++time.calls;

    time.init -= dclock();

    into.zero();
    
    int Lt = l.getLt();
    assert(into.nodeSites(3) == GJP.TnodeSites()); //cannot be SIMD packed in t-direction

    //Which local timeslices do we need?
    std::vector<int> local_timeslices_v;
    {
      int tsep_start_end = (t_end - t_start + 10*Lt) % Lt; //wrap periodically
      for(int tlcl = 0; tlcl < GJP.TnodeSites(); tlcl++){
	int tglb = tlcl + GJP.TnodeCoor() * GJP.TnodeSites();
	if( (tglb - t_start + 3*Lt) % Lt <= tsep_start_end )
	  local_timeslices_v.push_back(tlcl);
      }
      std::cout << "t_start=" << t_start << " t_end=" << t_end << " tsep=" << tsep_start_end << ". This node doing timeslices: ";
      for(auto v: local_timeslices_v) std::cout << v << " ";
      std::cout << std::endl;
    }
    int nt_do = local_timeslices_v.size();
    if(nt_do == 0) return;
    
    hostDeviceMirroredContainer<int> local_timeslices(nt_do);
    memcpy( local_timeslices.getHostWritePtr(), local_timeslices_v.data(), nt_do * sizeof(int));
        
    CPSautoView(l_v, l);
    CPSautoView(r_v, r);
    auto into_v = into.view(); //doesn't require free

    //This version is designed for l, r with the same temporal src_width
    int ntblocks = l.getNtBlocks();
    assert(r.getNtBlocks() == ntblocks);
       
    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);

    int nf = GJP.Gparity() + 1;
    size_t nsimd = VectorComplexType::Nsimd();
    size_t vol4d_node = into.size();
    int t_off = GJP.TnodeSites() * GJP.TnodeCoor();
    size_t blocksize = BlockedvMvOffloadArgs::b;
    //size_t inner_blocksize = BlockedvMvOffloadArgs::bb;

    size_t vol3d_node = vol4d_node / GJP.TnodeSites();
    
    typedef SIMT<VectorComplexType> ACC;
    typedef typename MesonFieldType::ScalarComplexType MFcomplexType;

    //Need to compute \sum_i\sum_j v(il)_{scl,fl}(x)  M(ir, jl) * v(jr)_{scr,fr}(x)
    
    //Transform into   \sum_i' \sum_j'  v'(i')_{scl, fl}(x) M'(i',j') v'(j')_{scr, fr}(x)  alpha(scl,fl,t,i')   beta(scr,fr,t,j')
    //i' and j' run over the full set of allowed index pairs
    //alpha, beta are boolean masks that zero out pairs that are not valid for a particular sc,f,t
    //v'(i')_{scl, fl}(x) = v(il[i'])_{scl, fl}(x)
    //M'(i',j') = M(ir[i'],jl[j'])


    //Then define
    //va'(i')_{scl, fl}(x) =  alpha(scl,fl,t,i')  v'(i')_{scl, fl}(x)
    //vb'(j')_{scr, fr}(x) =  beta(scr,fr,t,j')  v'(j')_{scr, fr}(x)
    //Such that
    //\sum_i' \sum_j'  va'(i')_{scl, fl}(x) M'(i',j') vb'(j')_{scr, fr}(x)  

    ManagedVector< std::pair<int,int> > il_ir_pairs, jl_jr_pairs; //pairs of il,il  and jl, jr  that must be contracted
    std::map<std::pair<int,int>, int> il_ir_pairs_index_map, jl_jr_pairs_index_map; //map of a pair to its offset in the arrays above
    setup_index_maps(il_ir_pairs, jl_jr_pairs, il_ir_pairs_index_map, jl_jr_pairs_index_map, M, nf, ntblocks, i_ind, j_ind);

    int nil_ir_pairs = il_ir_pairs.size(), njl_jr_pairs = jl_jr_pairs.size();
    auto il_ir_pairs_v = il_ir_pairs.view(), jl_jr_pairs_v = jl_jr_pairs.view();
    
    //Construct the masks
    ManagedVector<uint8_t> alpha(12*nf*ntblocks*nil_ir_pairs,0), beta(12*nf*ntblocks*njl_jr_pairs);
    setup_alpha_beta(alpha, beta, M, nf, ntblocks, i_ind, j_ind, il_ir_pairs_index_map, jl_jr_pairs_index_map);
    auto alpha_v = alpha.view(), beta_v = beta.view();

    //Prepare for blocked vMv
    size_t niprime = nil_ir_pairs, njprime = njl_jr_pairs;

    size_t field_size = 12 * nf * vol3d_node * nt_do;
    size_t blocked_fermfields_bytes = field_size * blocksize * sizeof(VectorComplexType);
    size_t blocked_cmplxfields_bytes = vol3d_node * nt_do * blocksize * sizeof(VectorComplexType);    
    size_t Mprime_bytes = blocksize * blocksize * sizeof(MFcomplexType);

    if(!UniqueID()){
      std::cout << "Outer block size is blocksize=" << blocksize  << std::endl;  //<< " inner blocksize " << inner_blocksize << std::endl;
      std::cout << "Fermion field size is F=" << double(field_size * sizeof(VectorComplexType)) / 1024./1024. << " MB and complex filed Z="
		<< double(vol4d_node*sizeof(VectorComplexType)) / 1024./1024. << "MB, for vector temporaries require blocksize * (F + 2*Z) MB total" << std::endl;
      std::cout << "vaprime " << double(blocked_fermfields_bytes)/1024./1024. << " MB" << std::endl;
      std::cout << "vbprime " << double(blocked_cmplxfields_bytes)/1024./1024. << " MB" << std::endl;
      std::cout << "Mprime " << double(Mprime_bytes)/1024./1024. << " MB" << std::endl;
      std::cout << "Mvbprime " << double(blocked_cmplxfields_bytes)/1024./1024. << " MB" << std::endl;
    }
    
    VectorComplexType* vaprime = (VectorComplexType*)device_alloc_check(blocked_fermfields_bytes);
    VectorComplexType* vbprime = (VectorComplexType*)device_alloc_check(blocked_cmplxfields_bytes); //only one spin,color,flavor    
    MFcomplexType* Mprime = (MFcomplexType*)managed_alloc_check(Mprime_bytes);
    VectorComplexType* Mvbprime = (VectorComplexType*)device_alloc_check(blocked_cmplxfields_bytes);
    
    //Do in blocks over i',j' to avoid taking too much space
    size_t niprime_blocks = (niprime + blocksize-1)/blocksize;
    size_t njprime_blocks = (njprime + blocksize-1)/blocksize;

    time.init += dclock();
    
    for(size_t iprimeblock =0; iprimeblock < niprime_blocks; iprimeblock++){

      time.vaprime -= dclock();
      size_t iprimestart = iprimeblock * blocksize;
      size_t iprimelessthan = std::min(iprimestart + blocksize, niprime);
      size_t niprime_block = iprimelessthan - iprimestart;

      std::cout << "iprimeblock:" << iprimeblock << " iprimestart:" << iprimestart << " iprimelessthan:" << iprimelessthan << " niprime_block:"<< niprime_block << std::endl;
      //std::cout << "Create va'" << std::endl;

      create_vaprime(vaprime, into_v, l_v, alpha_v, il_ir_pairs_v, t_off, l.getArgs().src_width, iprimestart, nf, ntblocks, vol3d_node, local_timeslices, niprime_block, nsimd, conj_l);
      time.vaprime += dclock();

      for(size_t jprimeblock =0; jprimeblock < njprime_blocks; jprimeblock++){
	size_t jprimestart = jprimeblock * blocksize;
	size_t jprimelessthan = std::min(jprimestart + blocksize, njprime);
	size_t njprime_block = jprimelessthan - jprimestart;	

	std::cout << "jprimeblock:" << jprimeblock << " jprimestart:" << jprimestart << " jprimelessthan:" << jprimelessthan << " njprime_block:"<< njprime_block << std::endl;

	time.Mprime -= dclock();
	create_Mprime(Mprime, M, iprimestart, iprimelessthan, jprimestart, jprimelessthan, il_ir_pairs, jl_jr_pairs);
	time.Mprime += dclock();
	
	time.lMr -= dclock();

	for(int fr=0;fr<nf;fr++){
	  for(int sr=0;sr<4;sr++){
	    for(int cr=0;cr<3;cr++){
	      int scr = cr+3*sr;
	      int scfr = cr+3*(sr + 4*fr);	      

	      //if(cr == 0 && sr == 0 && fr == 0 && jprimeblock == 0 && iprimeblock == 0) cudaProfilerStart();
	      
	      //Create vb'
	      create_vbprime(vbprime, into_v, r_v, beta_v, jl_jr_pairs_v, t_off, r.getArgs().src_width, jprimestart, nf, ntblocks, vol3d_node, local_timeslices, njprime_block, nsimd, conj_r, scr, fr);

	      //Mprime * vbprime
	      Mprime_vbprime(Mvbprime, Mprime, vbprime, vol3d_node * nt_do, niprime_block, njprime_block, nsimd, scfr==0);

	      //va' (M' vb')	      
	      vaprime_Mprime_vbprime(into_v, vaprime, Mvbprime, sr, cr, fr, vol3d_node, local_timeslices, niprime_block, nf, nsimd);

	      //if(cr == 0 && sr == 0 && fr == 0 && jprimeblock == 0 && iprimeblock == 0) cudaProfilerStop();
	      
	    }//cr
	  }//sr      
	}//fr

	//The kernels below takes a while so we may as well prefetch r for the next cycle
	//Note: these are issued after the kernels are launched because they lock up the CPU; the kernels are still executing at this time
	prefetch_r(jprimeblock, njprime_blocks, iprimeblock, niprime_blocks, blocksize, njprime, jl_jr_pairs, r, vol3d_node, local_timeslices);
	
	{
	  using namespace Grid;
	  accelerator_barrier(dummy);
	}

	time.lMr += dclock();
      }//jprimeblock

    }//iprimeblock

    device_free(vaprime);
    device_free(vbprime);
    managed_free(Mprime);
    device_free(Mvbprime);

#ifdef GRID_CUDA
    assert(cudaDeviceSetCacheConfig(cache_default) == cudaSuccess );

    for(int i=0;i<l.getNlowModes();i++) l.getLowMode(i).deviceSetAdviseUVMreadOnly(false);
    for(int i=0;i<l.getNhighModes();i++) l.getHighMode(i).deviceSetAdviseUVMreadOnly(false);
    for(int i=0;i<r.getNlowModes();i++) r.getLowMode(i).deviceSetAdviseUVMreadOnly(false);
    for(int i=0;i<r.getNhighModes();i++) r.getHighMode(i).deviceSetAdviseUVMreadOnly(false);

#endif   

  }//end of func

  
  static void implementation(PropagatorField &into,
			     const lA2AfieldType &l,
			     const MesonFieldType &M,
			     const rA2AfieldType &r,
			     bool conj_l, bool conj_r){
    optimized(into, l, M, r, conj_l, conj_r, 0, l.getLt()-1);
    //simple(into, l, M, r, conj_l, conj_r);
  }

  static void implementation(PropagatorField &into,
			     const lA2AfieldType &l,
			     const MesonFieldType &M,
			     const rA2AfieldType &r,
			     bool conj_l, bool conj_r,
			     const int t_start, const int t_end){
    optimized(into, l, M, r, conj_l, conj_r, t_start, t_end);
  }

  

  
};



#endif //USE_GRID


CPS_END_NAMESPACE

#endif
