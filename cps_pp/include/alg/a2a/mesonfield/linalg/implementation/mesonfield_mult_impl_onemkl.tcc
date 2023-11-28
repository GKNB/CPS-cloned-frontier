#ifndef _MULT_IMPL_CUBLASXT_H
#define _MULT_IMPL_CUBLASXT_H

class _mult_impl_base{
public:
  struct timers{
    int calls;
    double t_init;
    double t_alloc;
    double t_reord;
    double t_compute;
    double t_write;
    double t_globalsum;

    void reset(){
      calls = 0;
      t_init = t_alloc = t_reord = t_compute = t_write = t_globalsum = 0;
    }
    timers(){
      reset();
    }
    void print() const{
      a2a_printf("Timers: calls=%d init=%g alloc=%g reord=%g compute=%g write=%g gsum=%g\n", calls, t_init/calls, t_alloc/calls, t_reord/calls, t_compute/calls, t_write/calls, t_globalsum/calls);
    }
  };

  static timers & getTimers(){ static timers t; return t; }
};

//Implementations for meson field contractions
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
class _mult_impl: public _mult_impl_base{ //necessary to avoid an annoying ambigous overload when mesonfield friends mult
public:
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

  static void mult_onemkl(A2AmesonField<mf_Policies,lA2AfieldL,rA2AfieldR> &out, const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r, const bool node_local){
    getTimers().calls++;
    getTimers().t_init -= dclock();

    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    typedef typename ScalarComplexType::value_type mf_Float;
    
    assert( (void*)&out != (void*)&l || (void*)&out != (void*)&r );

    if(! l.getColParams().paramsEqual( r.getRowParams() ) ){
      LOGA2A << "mult():  Illegal matrix product: underlying vector parameters must match\n"
	     << "left-column: " << l.getColParams().print() << "\n"
	     << "right-row: " << r.getRowParams().print() << std::endl;
      exit(-1);
    }
    
    out.setup(l.getRowParams(),r.getColParams(), l.getRowTimeslice(), r.getColTimeslice() ); //zeroes output, so safe to re-use
  
    const int ni = l.getNrows();
    const int nk = r.getNcols();

    typedef typename A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR>::RightDilutionType LeftDilutionType;
    typedef typename A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR>::LeftDilutionType RightDilutionType;

    ModeContractionIndices<LeftDilutionType,RightDilutionType> j_ind2(l.getColParams()); //these maps could be cached somewhere

    modeIndexSet lmodeparams; lmodeparams.time = l.getColParams().tblock(l.getColTimeslice());
    modeIndexSet rmodeparams; rmodeparams.time = r.getRowParams().tblock(r.getRowTimeslice());
    
    const int nj = j_ind2.getNindices(lmodeparams,rmodeparams);

    int jlmap[nj], jrmap[nj];
    for(int j = 0; j < nj; j++)
      j_ind2.getBothIndices(jlmap[j],jrmap[j],j,lmodeparams,rmodeparams);


    //Find consecutive blocks in jlmap
    std::vector<std::pair<int,int> > jlmap_blocks;
    find_contiguous_blocks(jlmap_blocks, jlmap, nj);

    //We're going to let cublas decide how to "thread" the matrix multiplication internally, thus we just have to make the number of blocks equal to the number of nodes
    //Because ni, nk are different and not necessarily multiples of a common blocking we need to dynamically choose the block size
    int nodes = 1;
    for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);

    const int compute_elements = ( node_local ? 1 : nodes );

    //Want the total number of blocks to be close to the number of compute elements = (number of nodes)

    //niblk = ni/bi   number of i blocks
    //nkblk = nk/bk   number of k blocks

    //Require  niblk * nkblk ~ compute_elements
    //=>  ni * nk ~ compute_elements * bi * bk
    //Require ni % bi == 0,  nk % bk == 0

    int bi = 1, bk = 1;
    
    int cycle = 0;
    while(compute_elements * bi * bk < ni * nk){
      if(cycle % 2 == 0) bi = next_divisor(ni,bi);
      else bk = next_divisor(nk,bk);
      ++cycle;
    }

    assert(ni % bi == 0);
    const int niblk = ni/bi; //number of iblocks
  
    assert(nk % bk == 0);
    const int nkblk = nk/bk; //number of k blocks
    
    //parallelize ik blocks over ndoes
    const int work = niblk * nkblk;
    int node_work, node_off; bool do_work;
    getNodeWork(work,node_work,node_off,do_work,node_local);

    getTimers().t_init += dclock();

    if(do_work){    
      //Pull out the submatrices
      getTimers().t_alloc -= dclock();
      oneMKLmatrix<ScalarComplexType> lreord(bi,nj);
      oneMKLmatrix<ScalarComplexType> rreord(nj,bk);
      oneMKLmatrix<ScalarComplexType> lr(bi, bk);
      getTimers().t_alloc += dclock();	

      CPSautoView(l_v,l,HostRead);
      CPSautoView(r_v,r,HostRead);
      CPSautoView(out_v,out,HostWrite);
      
      //Compute which iblock,kblock index this node is responsible for
      //Some nodes might have to do >1 block depending on geometry
      for(int i0k0 = node_off; i0k0 < node_off + node_work; ++i0k0){
	int rem = i0k0;
	int k0 = rem % nkblk; rem /= nkblk;
	int i0 = rem;
	i0 *= bi; k0 *= bk;

	getTimers().t_reord -= dclock();
#ifndef MEMTEST_MODE

#pragma omp parallel for
	for(int i=i0;i<i0+bi;i++){
	  for(int b=0;b<jlmap_blocks.size();b++){
	    int j_start = jlmap_blocks[b].first;
	    int j_start_actual = jlmap[j_start];
	    int sz = jlmap_blocks[b].second;

	    ScalarComplexType *p = &lreord(i-i0,j_start);
	    const ScalarComplexType & el = l_v(i, j_start_actual);
	    memcpy(p, &el, sz*sizeof(ScalarComplexType));
	  }
	}

#pragma omp parallel for
	for(int j=0;j<nj;j++){
	  int j_actual = jrmap[j];
	  ScalarComplexType const* e = &r_v(j_actual, k0);	  
	  ScalarComplexType *p = &rreord(j,0);
	  memcpy(p, e, bk*sizeof(ScalarComplexType));
	}

	getTimers().t_reord += dclock();
	
	getTimers().t_compute -= dclock();

	mult_offload_oneMKL(lr, lreord, rreord);

	getTimers().t_compute += dclock();

	getTimers().t_write -= dclock();

#pragma omp parallel for
	for(int i=i0;i<i0+bi;i++){
	  ScalarComplexType *p = &out_v(i,k0);
	  ScalarComplexType const* e = &lr(i-i0,0);
	  memcpy(p, e, bk*sizeof(ScalarComplexType));
	}

	getTimers().t_write += dclock();

#endif //MEMTEST_MODE

      }//block loop
    } //do_work
    
    if(!node_local && nodes > 1){
      getTimers().t_globalsum -= dclock();
      out.nodeSum();
      getTimers().t_globalsum += dclock();
    }
  }
  
  
  inline static void mult(A2AmesonField<mf_Policies,lA2AfieldL,rA2AfieldR> &out, const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r, const bool node_local){
    mult_onemkl(out, l,r,node_local);
  }
};

#endif
