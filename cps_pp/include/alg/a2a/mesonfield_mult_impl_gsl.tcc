#ifndef _MULT_IMPL_GSL_H
#define _MULT_IMPL_GSL_H

CPS_END_NAMESPACE
#include<alg/a2a/gsl_wrapper.h>
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

    //if(!UniqueID()) printf("mult sizes %d %d %d block sizes %d %d %d, num blocks %d %d %d, total blocks %d, compute elements %d\n",ni,nj,nk,bi,nj,bk,ni0,1,nk0,ni0*nk0,compute_elements);
    
    if(do_work){    
      Float t1 = dclock();

      //complex mult  re = re*re - im*im, im = re*im + im*re   //6 flops
      //complex add   2 flops

      const Float flops_total = Float(ni)*Float(nk)*Float(nj)*8.;

      A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> lreord;
      A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> rreord;
#ifndef MEMTEST_MODE
      r.rowReorder(rreord,jrmap,nj);
      l.colReorder(lreord,jlmap,nj);
#endif
      
      typename gw::matrix_complex *lreord_gsl = gw::matrix_complex_alloc(ni,nj);
      typename gw::matrix_complex *rreord_gsl = gw::matrix_complex_alloc(nj,nk);
      
#ifndef MEMTEST_MODE
      
#pragma omp parallel for
      for(int i=0;i<ni;i++)
	for(int j=0;j<nj;j++){
	  const ScalarComplexType & el = lreord(i, j);
	  mf_Float *el_gsl = (mf_Float*)gw::matrix_complex_ptr(lreord_gsl,i,j);
	  *(el_gsl++) = std::real(el);
	  *(el_gsl) = std::imag(el);
	}

#pragma omp parallel for
      for(int j=0;j<nj;j++)
	for(int k=0;k<nk;k++){
	  const ScalarComplexType & el = rreord(j, k);
	  mf_Float *el_gsl = (mf_Float*)gw::matrix_complex_ptr(rreord_gsl,j,k);
	  *(el_gsl++) = std::real(el);
	  *(el_gsl) = std::imag(el);
	}
      
#endif
      
      static const int lcol_stride = 1;      
      const int rrow_stride = rreord.getNcols();

      typename gw::complex one; GSL_SET_COMPLEX(&one,1.0,0.0);
      typename gw::complex zero; GSL_SET_COMPLEX(&zero,0.0,0.0);
      
#pragma omp parallel for
      for(int i0k0 = node_off; i0k0 < node_off + node_work; ++i0k0){
	int rem = i0k0;
	int k0 = rem % nk0; rem /= nk0;
	int i0 = rem;
	i0 *= bi; k0 *= bk;

	typename gw::matrix_complex *tmp_out = gw::matrix_complex_alloc(bi,bk);

	// if(i0 >= ni) ERR.General("_mult_impl","mult","i0 out of range\n");
	// if(i0 + bi > ni) ERR.General("_mult_impl","mult","i0+bi overflows matrix\n");
	// if(k0 >= nk) ERR.General("_mult_impl","mult","k0 out of range\n");
	// if(k0 + bk > nk) ERR.General("_mult_impl","mult","k0+bk overflows matrix\n");
	
	typename gw::matrix_complex_const_view ijblock_view = gw::matrix_complex_const_submatrix(lreord_gsl,i0,0,bi,nj);
	typename gw::matrix_complex_const_view jkblock_view = gw::matrix_complex_const_submatrix(rreord_gsl,0,k0,nj,bk);

	const typename gw::matrix_complex *const ijblock = &ijblock_view.matrix; //gw::matrix_complex_alloc(bi,bj);
	const typename gw::matrix_complex *const jkblock = &jkblock_view.matrix;  //gw::matrix_complex_alloc(bj,bk);

#ifndef MEMTEST_MODE
	gw::matrix_complex_set_zero(tmp_out);
	gw::blas_gemm(CblasNoTrans, CblasNoTrans, one, ijblock, jkblock, zero, tmp_out);
	
	for(int i=0;i<bi;i++) 
	  for(int k=0;k<bk;k++){
	    mf_Float const* el = (mf_Float const*)gw::matrix_complex_ptr(tmp_out,i,k);
	    out(i0+i,k0+k) = ScalarComplexType(el[0],el[1]);
	  }
#endif	
	gw::matrix_complex_free(tmp_out);
      }

      Float t2 = dclock();
      Float flops_per_sec = flops_total/(t2-t1);
      //if(!UniqueID()) printf("node mult flops/s %g  (time %f total flops %g)\n",flops_per_sec,t2-t1,flops_total);

      gw::matrix_complex_free(lreord_gsl);
      gw::matrix_complex_free(rreord_gsl);
    }
    Float time = -dclock();
    if(!node_local) out.nodeSum();
    time += dclock();
    //if(!UniqueID()) printf("mult comms time %g s\n",time);
  }
  
};

#endif
