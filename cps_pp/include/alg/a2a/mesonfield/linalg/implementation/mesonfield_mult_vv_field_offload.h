#ifndef _MULT_VV_FIELD_OFFLOAD_H_
#define _MULT_VV_FIELD_OFFLOAD_H_

#include<set>
#include "mesonfield_mult_vMv_field_offload.h"
#include <alg/a2a/utils/utils_gpu.h>

CPS_START_NAMESPACE

template<typename mf_Policies, 
	 template <typename> class lA2Afield,  template <typename> class rA2Afield,
	 typename PropagatorField,
	 typename ComplexClass>
class _mult_vv_field_offload_v{};

template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class rA2Afield,
	 typename PropagatorField>
using mult_vv_field = _mult_vv_field_offload_v<mf_Policies, lA2Afield, rA2Afield, PropagatorField, typename ComplexClassify<typename mf_Policies::ComplexType>::type >;


#ifdef USE_GRID


struct mult_vv_field_offload_timers{
  struct timers{
    double init1;
    double init2;
    double vv;
    size_t calls;

    timers(): init1(0), init2(0), vv(0), calls(0){}

    void reset(){
      init1=init2=vv=0;
      calls = 0;
    }
    void average(){
      init1/=calls;
      init2/=calls;
      vv/=calls;
    }
    void print(){
      average();
      a2a_printf("calls=%zu init1=%g init2=%g vv=%g\n", calls, init1, init2, vv);
    }

  };
  static timers & get(){ static timers t; return t; }
};


//For A2A vector A,B \in { A2AvectorW, A2AvectorV, A2AvectorWfftw, A2AvectorVfftw }
//Compute   AB  contracting over mode indices to produce a spin-color(-flavor) matrix field
template<typename mf_Policies, 
	 template <typename> class lA2Afield,  template <typename> class rA2Afield,
	 typename PropagatorField>
struct _mult_vv_field_offload_v<mf_Policies,lA2Afield,rA2Afield,PropagatorField,grid_vector_complex_mark>{
  typedef _mult_vMv_field_offload_fields<mf_Policies, mf_Policies::GPARITY> fdef;
  typedef typename PropagatorField::FieldSiteType VectorMatrixType;
  
  //typedef typename fdef::VectorMatrixType VectorMatrixType;
  //typedef typename fdef::PropagatorField PropagatorField;

  typedef lA2Afield<mf_Policies> lA2AfieldType;
  typedef rA2Afield<mf_Policies> rA2AfieldType;

  typedef typename lA2AfieldType::DilutionType leftDilutionType;
  typedef typename rA2AfieldType::DilutionType rightDilutionType;
  
  typedef typename mf_Policies::ComplexType VectorComplexType;
  typedef typename VectorComplexType::scalar_type ScalarComplexType;

  //A slow but simple implementation ignoring the index compression
  static void simple(PropagatorField &into,
		     const lA2AfieldType &l,
		     const rA2AfieldType &r,
		     bool conj_l, bool conj_r){
    into.zero();

    A2Aparams params(l);
    StandardIndexDilution dil(params);
    
    size_t n = dil.getNmodes();
    
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

				  for(size_t i=0;i<n;i++){				    
				    ScalarComplexType lval_tmp = ACC::read(l.elem(i,xop,top,cl+3*sl,fl));
				    ScalarComplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;

				    ScalarComplexType rval_tmp = ACC::read(r.elem(i,xop,top,cr+3*sr,fr));
				    ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
				    
				    ScalarComplexType val = ACC::read(out) + lval * rval;
				    ACC::write(out, val);				    
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


#if defined(GRID_CUDA) || defined(GRID_HIP)
  static void run_VV_kernel_GPU(VectorComplexType* vaprime,
				VectorComplexType* vbprime,
				typename ManagedVector<uint8_t>::View &alpha,
				PropagatorField &into,
				size_t niprime, size_t niprime_block,
				size_t iprimestart, size_t iprimelessthan,
				size_t vol4d, int t_off, int nf, int src_width, size_t nsimd){
    device_UVM_prefetch_device(alpha.data(), alpha.byte_size());
    if(PropagatorField::FieldAllocPolicy::UVMenabled){
      CPSautoView(into_v,into,HostRead);
      device_UVM_prefetch_device(into_v.ptr(), into.byte_size());
    }

    //Divide up into two matrices of size   3 * shmem_iblock_size   requiring  bytes =  2* 3*shmem_iblock_size *sizeof(ScalarComplexType)
    int shmem_max = maxDeviceShmemPerBlock() / 4; //if we use all the available shared memory we get less blocks running at once. Tuneable
    int gpu_threads = Grid::acceleratorThreads();
    int shmem_max_per_thread = shmem_max / nsimd / gpu_threads;
    int shmem_iblock_size = shmem_max_per_thread/3/2/sizeof(ScalarComplexType);  // two  3 * shmem_iblock_size complex matrices (fixed fl,sl fr,sr)
    LOGA2A << "Shared memory " << shmem_max/1024 << " kB with block sizes nsimd*gpu_threads="<< nsimd << "*"<<gpu_threads <<"=" << nsimd*gpu_threads 
	      << " allows " << shmem_max_per_thread << " B/thread which allows for iblock size " << shmem_iblock_size << " for sizeof(ScalarComplexType)=" << sizeof(ScalarComplexType) << std::endl;
    if(shmem_iblock_size == 0) assert(0);

    using namespace Grid;
    CPSautoView(into_v,into,DeviceReadWrite);
    accelerator_for_shmem(x4d, 
			  vol4d, 
			  nsimd, 
			  shmem_max,
			{
			  typedef SIMT<VectorComplexType> ACC; //this will use scalar data as on device
			  typedef typename ACC::value_type SIMTcomplexType; //=ScalarComplexType on GPU
			  
			  extern __shared__ SIMTcomplexType shared_all[];
                          //ScalarComplexType* shared_t = shared_all + (threadIdx.z + nsimd * threadIdx.x)*2*3*shmem_iblock_size; //OLD MAPPING IN GRID
                          SIMTcomplexType* shared_t = shared_all + (threadIdx.x + nsimd * threadIdx.y)*2*3*shmem_iblock_size; //New mapping uses threadIdx.x for simd lan
			  
			  SIMTcomplexType *matA = shared_t;
			  SIMTcomplexType *matB = shared_t + 3*shmem_iblock_size;

			  size_t xop; int top;
			  into_v.fourToThree(xop, top, x4d);
			  int t_glob = top + t_off;
			  int t_glob_block = t_glob / src_width;

			  VectorMatrixType &vsite_mat = *into_v.fsite_ptr(x4d);
			  size_t niprimeb_subblocks = (niprime_block + shmem_iblock_size - 1)/shmem_iblock_size;
			  for(int iprimeb_subblock=0; iprimeb_subblock < niprimeb_subblocks; iprimeb_subblock++){
			    size_t iprimeb_start = iprimeb_subblock * shmem_iblock_size;
			    size_t iprimeb_lessthan = min_value( iprimeb_start + shmem_iblock_size, niprime_block );
			    size_t iprimeb_subblock_size = iprimeb_lessthan - iprimeb_start;
			    
			    for(int fl=0;fl<nf;fl++){
			      for(int sl=0;sl<4;sl++){

				//Populate matA
				for(int cl=0;cl<3;cl++){
				  int scl = cl + 3*sl;
				  for(int isb=0; isb < iprimeb_subblock_size; isb++){
				    size_t iprimeb = isb + iprimeb_start;
				    VectorComplexType const* lptr = vaprime + iprimeb + niprime_block*(scl + 12*(fl + nf*x4d) );
				    matA[isb + iprimeb_subblock_size*cl] = ACC::read(*lptr);				  
				  }
				}
				for(int fr=0;fr<nf;fr++){
				  for(int sr=0;sr<4;sr++){

				    //Populate matB
				    for(int cr=0;cr<3;cr++){
				      int scr = cr + 3*sr;
				      for(int isb=0; isb < iprimeb_subblock_size; isb++){
					size_t iprimeb = isb + iprimeb_start;
					VectorComplexType const* rptr = vbprime + iprimeb + niprime_block*(scr + 12*(fr + nf*x4d) );
					matB[isb + iprimeb_subblock_size*cr] = ACC::read(*rptr);
				      }
				    }

				    //Multiply matA * matB
				    for(int cl=0;cl<3;cl++){
				      int scl = cl+3*sl;
				      for(int cr=0;cr<3;cr++){
					int scr = cr+3*sr;

					VectorComplexType &out = fdef::access(sl,cl,fl, sr,cr,fr, vsite_mat);
					SIMTcomplexType sum = ACC::read(out);
					uint8_t const* alptr = alpha.data() + iprimestart + iprimeb_start + niprime * ( scr + 12*(fr + nf*( scl + 12*( fl + nf*t_glob_block) ) ) );

					for(int isb=0; isb < iprimeb_subblock_size; isb++){
					  if(*alptr++ == 1){
					    SIMTcomplexType lval = matA[isb + iprimeb_subblock_size*cl];
					    SIMTcomplexType rval = matB[isb + iprimeb_subblock_size*cr];
					    sum = sum + lval * rval;
					  }
					}
					ACC::write(out, sum);

				      }//cr
				    }//cl


				  }//sr
				}//fr			      
			      }//sl
			    }//fl
			  }//iprimeb_subblock
			});
  }
#endif


#if !defined(GRID_CUDA) && !defined(GRID_HIP)
  static void run_VV_kernel_base(VectorComplexType* vaprime,
				 VectorComplexType* vbprime,
				 typename ManagedVector<uint8_t>::View &alpha,
				 PropagatorField &into,
				 size_t niprime, size_t niprime_block,
				 size_t iprimestart, size_t iprimelessthan,
				 size_t vol4d, int t_off, int nf, int src_width, size_t nsimd
				 ){
    static const int shmem_iblock_size = 4;
    
    using namespace Grid;
    CPSautoView(into_v,into,DeviceReadWrite);    
    accelerator_for(x4d, 
		    vol4d, 
		    nsimd, 
		    {
		      typedef SIMT<VectorComplexType> ACC; //this will use scalar data as on device
		      typedef typename ACC::value_type SIMTcomplexType; //=ScalarComplexType on GPU

		      SIMTcomplexType shared_t[2*3*shmem_iblock_size]; //thread stack

		      SIMTcomplexType *matA = shared_t;
		      SIMTcomplexType *matB = shared_t + 3*shmem_iblock_size;

		      size_t xop; int top;
		      into.fourToThree(xop, top, x4d);
		      int t_glob = top + t_off;
		      int t_glob_block = t_glob / src_width;

		      VectorMatrixType &vsite_mat = *into_v.fsite_ptr(x4d);
		      size_t niprimeb_subblocks = (niprime_block + shmem_iblock_size - 1)/shmem_iblock_size;
		      for(int iprimeb_subblock=0; iprimeb_subblock < niprimeb_subblocks; iprimeb_subblock++){
			size_t iprimeb_start = iprimeb_subblock * shmem_iblock_size;
			size_t iprimeb_lessthan = min_value( iprimeb_start + shmem_iblock_size, niprime_block );
			size_t iprimeb_subblock_size = iprimeb_lessthan - iprimeb_start;
			    
			for(int fl=0;fl<nf;fl++){
			  for(int sl=0;sl<4;sl++){

			    //Populate matA
			    for(int cl=0;cl<3;cl++){
			      int scl = cl + 3*sl;
			      for(int isb=0; isb < iprimeb_subblock_size; isb++){
				size_t iprimeb = isb + iprimeb_start;
				VectorComplexType const* lptr = vaprime + iprimeb + niprime_block*(scl + 12*(fl + nf*x4d) );
				matA[isb + iprimeb_subblock_size*cl] = ACC::read(*lptr);				  
			      }
			    }
			    for(int fr=0;fr<nf;fr++){
			      for(int sr=0;sr<4;sr++){

				//Populate matB
				for(int cr=0;cr<3;cr++){
				  int scr = cr + 3*sr;
				  for(int isb=0; isb < iprimeb_subblock_size; isb++){
				    size_t iprimeb = isb + iprimeb_start;
				    VectorComplexType const* rptr = vbprime + iprimeb + niprime_block*(scr + 12*(fr + nf*x4d) );
				    matB[isb + iprimeb_subblock_size*cr] = ACC::read(*rptr);
				  }
				}

				//Multiply matA * matB
				for(int cl=0;cl<3;cl++){
				  int scl = cl+3*sl;
				  for(int cr=0;cr<3;cr++){
				    int scr = cr+3*sr;

				    VectorComplexType &out = fdef::access(sl,cl,fl, sr,cr,fr, vsite_mat);
				    SIMTcomplexType sum = ACC::read(out);
				    uint8_t const* alptr = alpha.data() + iprimestart + iprimeb_start + niprime * ( scr + 12*(fr + nf*( scl + 12*( fl + nf*t_glob_block) ) ) );

				    for(int isb=0; isb < iprimeb_subblock_size; isb++){
				      if(*alptr++ == 1){
					SIMTcomplexType lval = matA[isb + iprimeb_subblock_size*cl];
					SIMTcomplexType rval = matB[isb + iprimeb_subblock_size*cr];
					sum = sum + lval * rval;
				      }
				    }
				    ACC::write(out, sum);

				  }//cr
				}//cl


			      }//sr
			    }//fr			      
			  }//sl
			}//fl
		      }//iprimeb_subblock
		    });
  }



  static void run_VV_kernel_base_simple(VectorComplexType* vaprime,
					VectorComplexType* vbprime,
					typename ManagedVector<uint8_t>::View &alpha,
					PropagatorField &into,
					size_t niprime, size_t niprime_block,
					size_t iprimestart, size_t iprimelessthan,
					size_t vol4d, int t_off, int nf, int src_width, size_t nsimd
					){
    using namespace Grid;
    CPSautoView(into_v,into,DeviceReadWrite);    
    accelerator_for(x4d, 
		    vol4d, 
		    nsimd, 
		    {
		      typedef SIMT<VectorComplexType> ACC; //this will use scalar data as on device
		      typedef typename ACC::value_type SIMTcomplexType; //=ScalarComplexType on GPU

		      size_t xop; int top;
		      into.fourToThree(xop, top, x4d);
		      int t_glob = top + t_off;
		      int t_glob_block = t_glob / src_width;

		      VectorMatrixType &vsite_mat = *into_v.fsite_ptr(x4d);

			for(int fl=0;fl<nf;fl++){
			  for(int sl=0;sl<4;sl++){
			    for(int cl=0;cl<3;cl++){
			      int scl = cl + 3*sl;			      
			      
			      for(int fr=0;fr<nf;fr++){
				for(int sr=0;sr<4;sr++){
				  for(int cr=0;cr<3;cr++){
				    int scr = cr + 3*sr;

				    VectorComplexType &out = fdef::access(sl,cl,fl, sr,cr,fr, vsite_mat);
				    SIMTcomplexType sum = ACC::read(out);

				    uint8_t const* alptr = alpha.data() + iprimestart + niprime * ( scr + 12*(fr + nf*( scl + 12*( fl + nf*t_glob_block) ) ) );
				    
				    for(int iprimeb=0; iprimeb < niprime_block; iprimeb++){  
				      VectorComplexType const* lptr = vaprime + iprimeb + niprime_block*(scl + 12*(fl + nf*x4d) );
				      VectorComplexType const* rptr = vbprime + iprimeb + niprime_block*(scr + 12*(fr + nf*x4d) );
				    
				      if(*alptr++ == 1){
					sum = sum + (*lptr) * (*rptr);
				      }
				    }
				    
				    ACC::write(out, sum);
				  }
				}
			      }
			    }
			  }
			}
		    });

  }




#endif

  static void prefetch_start(size_t iprimeblock, size_t niprime_blocks, size_t blocksize, size_t niprime, const lA2AfieldType &l, const rA2AfieldType &r, const hostDeviceMirroredContainer< std::pair<int,int> > &il_ir_pairs){
    size_t iprimeblock_nxt = iprimeblock + 1;
    if(iprimeblock_nxt < niprime_blocks){
      size_t iprimestart_nxt = iprimeblock_nxt * blocksize;
      size_t iprimelessthan_nxt = std::min(iprimestart_nxt + blocksize, niprime);
     
      std::vector<bool> lmodes_used(l.getNmodes(),false), rmodes_used(r.getNmodes(),false);
      {
	CPSautoView(il_ir_pairs_v, il_ir_pairs, HostRead);
	for(size_t iprime = iprimestart_nxt; iprime < iprimelessthan_nxt; iprime++){
	  lmodes_used[il_ir_pairs_v[iprime].first] = true;
	  rmodes_used[il_ir_pairs_v[iprime].second] = true;
	}
      }
      l.enqueuePrefetch(DeviceRead,lmodes_used);
      r.enqueuePrefetch(DeviceRead,rmodes_used);
      lA2AfieldType::startPrefetches();
    }
  }
  static void prefetch_wait(){
    lA2AfieldType::waitPrefetches();
  }    
   
  static void optimized(PropagatorField &into,
		 const lA2AfieldType &l,
		 const rA2AfieldType &r,
		 bool conj_l, bool conj_r){
    LOGA2A << "Starting field vv multiplication" << std::endl;

    mult_vv_field_offload_timers::timers &time = mult_vv_field_offload_timers::get();

    ++time.calls;

    time.init1 -= dclock();

    into.zero();

    ModeContractionIndices<leftDilutionType,rightDilutionType> i_ind(l);

    assert(into.nodeSites(3) == GJP.TnodeSites()); //cannot be SIMD packed in t-direction
    int nf = GJP.Gparity() + 1;
    int nsimd = VectorComplexType::Nsimd();
    size_t vol4d = into.size();
    int t_off = GJP.TnodeSites() * GJP.TnodeCoor();
    size_t blocksize = BlockedvMvOffloadArgs::b;

    //This version is designed for l, r with the same temporal src_width
    int ntblocks = l.getNtBlocks();
    assert(r.getNtBlocks() == ntblocks);

    int src_width = l.getArgs().src_width;
    assert(r.getArgs().src_width == src_width);

    //Need to compute \sum_i v(il)_{scl,fl}(x) v(ir)_{scr,fr}(x)
    //The map of i -> il, ir depends on scl,fl, scr,fr, t
    
    //Transform into   \sum_i'  v'(i')_{scl, fl}(x) v'(i')_{scr, fr}(x)  alpha(scl,fl,scr,fr,t,i')
    //i' runs over the full set of allowed index pairs
    //alpha is a boolean mask that zero out pairs that are not valid for a particular scl,fl,scr,fr,t
    //v'(i')_{scl, fl}(x) = v(il[i'])_{scl, fl}(x)
    //v'(i')_{scr, fr}(x) = v(ir[i'])_{scr, fr}(x)

    int lmodes = l.getNmodes();
    int rmodes = r.getNmodes();
    std::vector<std::vector<bool> > il_ir_used(lmodes, std::vector<bool>(rmodes,false));  

    {
      modeIndexSet ilp, irp;
      for(int tv=0;tv<ntblocks;tv++){
	ilp.time = irp.time = tv;
	for(ilp.flavor=0; ilp.flavor<nf; ilp.flavor++){
	  for(ilp.spin_color=0; ilp.spin_color<12; ilp.spin_color++){
	    for(irp.flavor=0; irp.flavor<nf; irp.flavor++){
	      for(irp.spin_color=0; irp.spin_color<12; irp.spin_color++){
		
		const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
		for(int i=0;i<i_ind_pairs.size();i++)
		  il_ir_used[ i_ind_pairs[i].first ][ i_ind_pairs[i].second ] = true;
		
	      }
	    }
	  }
	}
      }
    }
    //Count
    int nil_ir_pairs = 0;
    for(int ll=0;ll<lmodes;ll++)
      for(int rr=0;rr<rmodes;rr++)
	if(il_ir_used[ll][rr])
	  ++nil_ir_pairs;

    //Map
    hostDeviceMirroredContainer< std::pair<int,int> > il_ir_pairs(nil_ir_pairs); //map of pair index to pair (ll,rr) 
    std::vector<std::vector<int> > il_ir_pairs_index_map(lmodes, std::vector<int>(rmodes)); //inverse map of (ll,rr) -> pair index  (if ll,rr in use)
    {
      CPSautoView(il_ir_pairs_v, il_ir_pairs, HostWrite);
      int ii=0;
      for(int ll=0;ll<lmodes;ll++){
	for(int rr=0;rr<rmodes;rr++){
	  if(il_ir_used[ll][rr]){
	    il_ir_pairs_v[ii] = std::pair<int,int>(ll,rr);
	    il_ir_pairs_index_map[ll][rr] = ii;
	    ++ii;
	  }
	}
      }
    }

    //Construct the mask
    ManagedVector<uint8_t> alpha(nil_ir_pairs*12*nf*12*nf*ntblocks,0); //map as  i' + ni' * (scr + 12*(fr + nf*( scl + 12*(fl + nf*tblock)  ) ) )
    auto alpha_v = alpha.view();
    {
      modeIndexSet ilp, irp;
      for(int tv=0;tv<ntblocks;tv++){
	ilp.time = irp.time = tv;
	for(ilp.flavor=0; ilp.flavor<nf; ilp.flavor++){
	  for(ilp.spin_color=0; ilp.spin_color<12; ilp.spin_color++){
	    for(irp.flavor=0; irp.flavor<nf; irp.flavor++){
	      for(irp.spin_color=0; irp.spin_color<12; irp.spin_color++){

		const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
		for(int i=0;i<i_ind_pairs.size();i++){
		  const std::pair<int, int> &pair = i_ind_pairs[i];
		  int pair_idx = il_ir_pairs_index_map[pair.first][pair.second];
		  size_t alpha_idx = pair_idx + nil_ir_pairs * (irp.spin_color + 12*(irp.flavor + nf*( ilp.spin_color + 12*(ilp.flavor + nf*tv)  ) ) );
		  alpha[alpha_idx] = 1;
		}
	      }
	    }
	  }
	}
      }
    }

    size_t niprime = nil_ir_pairs;

    size_t field_size = 12 * nf * vol4d;
    size_t vprime_bytes = field_size * blocksize * sizeof(VectorComplexType);

    VectorComplexType* vaprime = (VectorComplexType*)device_alloc_check(vprime_bytes);
    VectorComplexType* vbprime = (VectorComplexType*)device_alloc_check(vprime_bytes);
    
    LOGA2A << "Outer block size " << blocksize << std::endl
	   << "vaprime " << double(vprime_bytes)/1024./1024. << " MB" << std::endl
	   << "vbprime " << double(vprime_bytes)/1024./1024. << " MB" << std::endl;

    
    //Do in blocks over i' to avoid taking too much space
    size_t niprime_blocks = (niprime + blocksize-1)/blocksize;

    time.init1 += dclock();
    
    for(size_t iprimeblock =0; iprimeblock < niprime_blocks; iprimeblock++){

      time.init2 -= dclock();
      size_t iprimestart = iprimeblock * blocksize;
      size_t iprimelessthan = std::min(iprimestart + blocksize, niprime);
      size_t niprime_block = iprimelessthan - iprimestart;

      LOGA2A << "iprimeblock:" << iprimeblock << " iprimestart:" << iprimestart << " iprimelessthan:" << iprimelessthan << " niprime_block:"<< niprime_block << std::endl;
      LOGA2A << "Create va'/vb'" << std::endl;
     
      std::vector<bool> lmodes_used(l.getNmodes(),false), rmodes_used(r.getNmodes(),false);
      {
	CPSautoView(il_ir_pairs_v, il_ir_pairs, HostRead);
	for(size_t iprime = iprimestart; iprime < iprimelessthan; iprime++){
	  lmodes_used[il_ir_pairs_v[iprime].first] = true;
	  rmodes_used[il_ir_pairs_v[iprime].second] = true;
	}
      }
    
      //Create va' and vb'
      {
	auto l_v = l.view(DeviceRead,lmodes_used);
	auto r_v = r.view(DeviceRead,rmodes_used);	
       
	CPSautoView(il_ir_pairs_v, il_ir_pairs, DeviceRead);

	prefetch_start(iprimeblock, niprime_blocks, blocksize, niprime, l, r, il_ir_pairs);
	
	using namespace Grid;	
	accelerator_for2d(x4d, vol4d, iprimeb, niprime_block, nsimd,
			{
			  typedef SIMT<VectorComplexType> ACC;
			  size_t xop; int top;
			  into.fourToThree(xop, top, x4d);
			  int t_glob = top + t_off;
			  size_t iprime = iprimeb + iprimestart;
			  
			  for(int f=0;f<nf;f++){
			    for(int sc=0;sc<12;sc++){

			      {
				VectorComplexType *into = vaprime +  iprimeb + niprime_block*( sc + 12*(f + nf*x4d) ); //contiguous in summed index
				auto val = ACC::read(l_v.nativeElem(il_ir_pairs_v[iprime].first, x4d, sc, f));
				val = conj_l ? Grid::conjugate(val) : val;
				ACC::write(*into, val);
			      }
			      {
				VectorComplexType *into = vbprime + iprimeb + niprime_block*( sc + 12*(f + nf*x4d) );  //contiguous in summed index
				auto val = ACC::read(r_v.nativeElem(il_ir_pairs_v[iprime].second, x4d, sc, f));
				val = conj_r ? Grid::conjugate(val) : val;
				ACC::write(*into, val);
			      }

			    }
			  }
			});
	l_v.free(); r_v.free();
      }
      
      time.init2 += dclock();

      LOGA2A << "Compute VV" << std::endl;
      time.vv -= dclock();
         
#if defined(GRID_CUDA) || defined(GRID_HIP)
      run_VV_kernel_GPU(vaprime, vbprime, alpha_v, into, niprime, niprime_block, iprimestart, iprimelessthan, vol4d, t_off, nf, src_width, nsimd);
#else
      run_VV_kernel_base(vaprime, vbprime, alpha_v, into, niprime, niprime_block, iprimestart, iprimelessthan, vol4d, t_off, nf, src_width, nsimd);
      //run_VV_kernel_base_simple(vaprime, vbprime, alpha_v, into, niprime, niprime_block, iprimestart, iprimelessthan, vol4d, t_off, nf, src_width, nsimd);
#endif

      prefetch_wait();
      
      time.vv += dclock();
    }
    
    device_free(vaprime);
    device_free(vbprime);
  }//end of func

  static void implementation(PropagatorField &into,
			     const lA2AfieldType &l,
			     const rA2AfieldType &r,
			     bool conj_l, bool conj_r){
    return optimized(into, l, r, conj_l, conj_r);
  }

};



#endif //USE_GRID


CPS_END_NAMESPACE

#endif
