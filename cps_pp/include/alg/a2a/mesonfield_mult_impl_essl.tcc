#ifndef _MESONFIELD_MULTI_IMPL_ESSL
#define _MESONFIELD_MULTI_IMPL_ESSL

CPS_END_NAMESPACE
#include<essl_interface.h>
CPS_START_NAMESPACE

//Implementations for meson field contractions
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
class _mult_impl{ //necessary to avoid an annoying ambigous overload when mesonfield friends mult
public:
  typedef gsl_wrapper<typename mf_Policies::ScalarComplexType::value_type> gw;

  //Matrix product of meson field pairs
  //out(t1,t4) = l(t1,t2) * r(t3,t4)     (The stored timeslices are only used to unpack TimePackedIndex so it doesn't matter if t2 and t3 are thrown away; their indices are contracted over hence the times are not needed)

  static int next_divisor(const int of, const int start){
    if(start == of) return start;
    
    int ret = start + 1;
    while(of % ret !=0){
      ++ret;
      if(ret == of) break;
    }
    return ret;
  }
  
  
  static void mult(A2AmesonField<mf_Policies,lA2AfieldL,rA2AfieldR> &out, const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r, const bool node_local){
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    typedef typename ScalarComplexType::value_type mf_Float;
    
    assert( (void*)&out != (void*)&l || (void*)&out != (void*)&r );

    if(! l.getColParams().paramsEqual( r.getRowParams() ) ){
      if(!UniqueID()){
	printf("mult():  Illegal matrix product: underlying vector parameters must match\n"); fflush(stdout);
	std::cout << "left-column: " << l.getColParams().print() << "\n";
	std::cout << "right-row: " << r.getRowParams().print() << "\n";
	std::cout.flush();
      }
      exit(-1);
    }

    out.setup(l.getRowParams(),r.getColParams(), l.getRowTimeslice(), r.getColTimeslice() ); //zeroes output, so safe to re-use
  
    const int ni = l.getNrows();
    const int nk = r.getNcols();

    typedef typename A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR>::RightDilutionType LeftDilutionType;
    typedef typename A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR>::LeftDilutionType RightDilutionType;

    ModeContractionIndices<LeftDilutionType,RightDilutionType> j_ind2(l.getColParams()); //these maps could be cached somewhere
    
    modeIndexSet lmodeparams; lmodeparams.time = l.getColTimeslice();
    modeIndexSet rmodeparams; rmodeparams.time = r.getRowTimeslice();
    
    const int nj = j_ind2.getNindices(lmodeparams,rmodeparams);

    int jlmap[nj], jrmap[nj];
    for(int j = 0; j < nj; j++)
      j_ind2.getBothIndices(jlmap[j],jrmap[j],j,lmodeparams,rmodeparams);

    //Try a blocked matrix multiply
    //Because ni, nk are different and not necessarily multiples of a common blocking we need to dynamically choose the block size
    int nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);

    const int compute_elements = omp_get_max_threads() * ( node_local ? 1 : nodes );

    //Want the total number of blocks to be close to the number of compute elements = (number of nodes)*(number of threads)

    //ni0 = ni/bi   number of i blocks
    //nk0 = nk/bk   number of k blocks

    //Require  ni0 * nk0 ~ compute_elements
    //=>  ni * nk ~ compute_elements * bi * bk
    //Require ni % bi == 0,  nk % bk == 0

    int bi = 1, bk = 1;
    
    int cycle = 0;
    while(compute_elements * bi * bk < ni * nk){
      //printf("bi %d bk %d,  compute_elements*bi*bk %d,  ni*nk %d\n",bi,bk,compute_elements*bi*bk,ni*nk);
      if(cycle % 2 == 0) bi = next_divisor(ni,bi);
      else bk = next_divisor(nk,bk);
      ++cycle;
    }

    assert(ni % bi == 0);
    const int ni0 = ni/bi;
  
    assert(nk % bk == 0);
    const int nk0 = nk/bk;

    
    //parallelize ik blocks
    const int work = ni0 * nk0;
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work,node_local);

    if(do_work){    
      ScalarComplexType* lreord_tbuf[omp_get_max_threads()];
      ScalarComplexType* rreord_tbuf[omp_get_max_threads()];
      ScalarComplexType* prod_tmp_tbuf[omp_get_max_threads()];
      for(int t=0;t<omp_get_max_threads();t++){
	lreord_tbuf[t] = (ScalarComplexType*)memalign_check(128, bi*nj*sizeof(ScalarComplexType));
	rreord_tbuf[t] = (ScalarComplexType*)memalign_check(128, nj*bk*sizeof(ScalarComplexType));
	prod_tmp_tbuf[t] = (ScalarComplexType*)memalign_check(128, bi*bk*sizeof(ScalarComplexType));
      }
     
#pragma omp parallel for
      for(int i0k0 = node_off; i0k0 < node_off + node_work; ++i0k0){
	int rem = i0k0;
	int k0 = rem % nk0; rem /= nk0;
	int i0 = rem;
	i0 *= bi; k0 *= bk;

	//Get the sub-matrix of l of size bi*nj and starting at coordinate (i0, 0) of the col-reordered matrix
	//These should be *column-major*
	ScalarComplexType* lreord = lreord_tbuf[omp_get_thread_num()];
	for(int i=i0;i<bi+i0;i++){
	  for(int j=0;j<nj;j++){
	    int j_actual = jlmap[j];
	    int off = (i-i0) + bi*j;
	    lreord[off] = l(i,j_actual);
	  }
	}
	//Get the sub-matrix of r of size nj*bk and starting at coordinate (0,k0) of the row-reordered matrix
	//These should be *column-major*
	ScalarComplexType* rreord = rreord_tbuf[omp_get_thread_num()];
	for(int j=0;j<nj;j++){
	  int j_actual = jrmap[j];
	  for(int k=k0;k<bk+k0;k++){
	    int off = j + nj*(k-k0);
	    rreord[off] = r(j_actual,k);
	  }
	}

#ifndef MEMTEST_MODE
	ScalarComplexType* prod_tmp = prod_tmp_tbuf[omp_get_thread_num()];
	memset(prod_tmp, 0, bi*bk*sizeof(ScalarComplexType));
	essl_interface::essl_gemul_colmajor(prod_tmp, lreord, rreord, bi, nj, bk, false, false);
	
	for(int i=0;i<bi;i++) 
	  for(int k=0;k<bk;k++)
	    out(i0+i,k0+k) = prod_tmp[i + bi*k]; //column-major
#endif	
      }

      for(int t=0;t<omp_get_max_threads();t++){
	free(lreord_tbuf[t]);
	free(rreord_tbuf[t]);
	free(prod_tmp_tbuf[t]);
      }      
    }
    if(!node_local) out.nodeSum();
  }  
};

#endif
