struct mesonfield_trace_prod_timings{
  struct _data{
    size_t count;
    double init;
    double table;
    double mf_copy;
    double prod_tmp_alloc_free;
    double prod;    
    double reduction;

    void reset(){
      count = 0;
      init = table = mf_copy = prod = reduction = prod_tmp_alloc_free = 0;
    }
    _data(){
      reset();
    }    
    void report(){
      std::cout << "Trace-prod " << count << " calls. Avg:  Init=" << init/count << "s Table=" << table/count << "s MF-copy=" << mf_copy/count << "s Prod-tmp-alloc-free=" << prod_tmp_alloc_free/count << "s Prod=" << prod/count << " Reduction=" << reduction/count << "s" << std::endl;
    }
  };
  
  static _data &data(){
    static _data d;
    return d;
  }
};
    

//Compute   l^ij(t1,t2) r^ji(t3,t4)
//Threaded but *node local*
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace(const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r){
  mesonfield_trace_prod_timings::_data &timings = mesonfield_trace_prod_timings::data();
  ++timings.count;
  timings.init -= dclock();
  
  //Check the indices match
  if(! l.getRowParams().paramsEqual( r.getColParams() ) || ! l.getColParams().paramsEqual( r.getRowParams() ) )
    ERR.General("","trace(const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &)","Illegal matrix product: underlying vector parameters must match\n");
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  ScalarComplexType into(0,0);

  typedef typename A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR>::LeftDilutionType DilType0;
  typedef typename A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR>::RightDilutionType DilType1;
  typedef typename A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR>::LeftDilutionType DilType2;
  typedef typename A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR>::RightDilutionType DilType3;

  ModeContractionIndices<DilType0,DilType3> i_ind(l.getRowParams());
  ModeContractionIndices<DilType1,DilType2> j_ind(r.getRowParams());

  const int times[4] = { l.getRowTimeslice(), l.getColTimeslice(), r.getRowTimeslice(), r.getColTimeslice() };

  //W * W is only non-zero when the timeslice upon which we evaluate them are equal
  modeIndexSet lip; lip.time = times[0];
  modeIndexSet rip; rip.time = times[3];

  modeIndexSet ljp; ljp.time = times[1];
  modeIndexSet rjp; rjp.time = times[2];

  const int ni = i_ind.getNindices(lip,rip); //how many indices to loop over
  const int nj = j_ind.getNindices(ljp,rjp);

  timings.init += dclock();
  
#ifndef MEMTEST_MODE

#ifdef GPU_VEC
  //std::cout << Grid::GridLogMessage << "Trace table create" << std::endl;
  timings.table -= dclock();
  
  hostDeviceMirroredContainer<int> tables(2*ni + 2*nj, false); //{li,ri}{lj,rj}   //Dont use pinned memory because of allocation time
  int* tab_host_write = tables.getHostWritePtr();

  //std::cout << Grid::GridLogMessage << "Trace table setup" << std::endl;
#pragma omp parallel for  
  for(int i=0;i<ni;i++){
    int li = i_ind.getLeftIndex(i,lip,rip);
    int ri = i_ind.getRightIndex(i,lip,rip);
    tab_host_write[2*i] = li;
    tab_host_write[2*i + 1] = ri;
  }
#pragma omp parallel for  
  for(int j = 0; j < nj; j++){
    int lj = j_ind.getLeftIndex(j,ljp,rjp);
    int rj = j_ind.getRightIndex(j,ljp,rjp);
    tab_host_write[2*ni + 2*j] = lj;
    tab_host_write[2*ni + 2*j + 1] = rj;
  }
  int const* tab_device_read = tables.getDeviceReadPtr();

  timings.table += dclock();

  timings.prod_tmp_alloc_free -= dclock();
  ScalarComplexType *tmp = (ScalarComplexType *)device_alloc_check(ni*nj*sizeof(ScalarComplexType));
  timings.prod_tmp_alloc_free += dclock();
  {
    //std::cout << Grid::GridLogMessage << "Trace view create" << std::endl;
    timings.mf_copy -= dclock();
    CPSautoView(l_v,l);
    CPSautoView(r_v,r);  
    timings.mf_copy += dclock();

    timings.prod -= dclock();
    using namespace Grid;
    //std::cout << GridLogMessage << "Trace step 1" << std::endl;
    accelerator_for2d(j,nj,i,ni,1,{ 
      const int li = tab_device_read[2*i];
      const int ri = tab_device_read[2*i+1];
      const int lj = tab_device_read[2*ni + 2*j];
      const int rj = tab_device_read[2*ni + 2*j + 1];
      tmp[j + nj*i] = l_v(li,lj) * r_v(rj,ri);
      });
    timings.prod += dclock();

    timings.reduction -= dclock();
    //Reduce over j at fixed i
    //std::cout << GridLogMessage << "Trace step 2" << std::endl;
    accelerator_for(i,ni,1,{
      ScalarComplexType red = 0;
      for(int j=0;j<nj;j++)
    	red += tmp[j+nj*i];
      tmp[i] = red; //each thread owns a different i so it's ok to write here
      });
    accelerator_for(i,1,1,{
    	ScalarComplexType red = 0;
    	for(int ii=0;ii<ni;ii++) red += tmp[ii];
    	tmp[0] = red;
      });      
    //std::cout << GridLogMessage << "Trace copy back" << std::endl;
    copy_device_to_host(&into, tmp, sizeof(ScalarComplexType));
    timings.reduction += dclock();
  }
  timings.prod_tmp_alloc_free -= dclock();		
  device_free(tmp);
  timings.prod_tmp_alloc_free += dclock();
    
#else
  const int n_threads = omp_get_max_threads();
  std::vector<ScalarComplexType, BasicAlignedAllocator<ScalarComplexType> > ret_vec(n_threads,ScalarComplexType(0.,0.));
  
  
#pragma omp parallel for schedule(static)
  for(int i = 0; i < ni; i++){
    const int id = omp_get_thread_num();
    const int li = i_ind.getLeftIndex(i,lip,rip);
    const int ri = i_ind.getRightIndex(i,lip,rip);

    for(int j = 0; j < nj; j++){
      const int lj = j_ind.getLeftIndex(j,ljp,rjp);
      const int rj = j_ind.getRightIndex(j,ljp,rjp);
      
      ret_vec[id] += l(li,lj) *  r(rj,ri);
    }
  }

  for(int i=0;i<n_threads;i++) into += ret_vec[i];
#endif //GRID_GPU
  
#endif //MEMTEST_MODE
  
  return into;
}


//Slow implementation for testing
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace_slow(const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r){
  assert(l.getNrowsFull() == r.getNcolsFull());
  assert(l.getNcolsFull() == r.getNrowsFull());
  
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  ScalarComplexType out = 0;
  for(int i=0;i<l.getNrowsFull();i++){
    for(int j=0;j<l.getNcolsFull();j++){
      out += l.elem(i,j) * r.elem(j,i);
    }
  }
  return out;
}

  

//Compute   l^ij(t1,t2) r^ji(t3,t4) for all t1, t4  and place into matrix element t1,t4
//This is both threaded and distributed over nodes
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
void trace(fMatrix<typename mf_Policies::ScalarComplexType> &into, const std::vector<A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> > &l, const std::vector<A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> > &r){
  //Distribute load over all nodes
  int lsize = l.size();
  int rsize = r.size();

  into.resize(lsize,rsize);

  const int work = lsize*rsize;

  bool do_work;
  int node_work, node_off;
  getNodeWork(work, node_work, node_off, do_work);

#ifndef MEMTEST_MODE
  if(do_work){
    for(int tt=node_off; tt<node_off + node_work; tt++){
      int rem = tt;
      int tsnk = rem % lsize; rem /= lsize; //sink time
      int tsrc = rem; //source time

      into(tsnk,tsrc) = trace(l[tsnk],r[tsrc]);
    }
  }
  into.nodeSum(); //give all nodes a copy
#endif
}


//Compute   m^ii(t1,t2)
//Threaded but *node local*
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &m){
  //Check the indices match
  if(! m.getRowParams().paramsEqual( m.getColParams() ) )
    ERR.General("","trace(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &)","Illegal trace: underlying index parameters must match\n");
  
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  ScalarComplexType into(0,0);

  typedef typename A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::LeftDilutionType DilType0;
  typedef typename A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR>::RightDilutionType DilType1;

  ModeContractionIndices<DilType0,DilType1> i_ind(m.getRowParams());

  const int times[2] = { m.getRowTimeslice(), m.getColTimeslice() };

  const int n_threads = omp_get_max_threads();
  std::vector<ScalarComplexType, BasicAlignedAllocator<ScalarComplexType> > ret_vec(n_threads,(0.,0.));
    
  modeIndexSet lip; lip.time = times[0];
  modeIndexSet rip; rip.time = times[1];

  const int ni = i_ind.getNindices(lip,rip); //how many indices to loop over

#ifndef MEMTEST_MODE

#pragma omp parallel for schedule(static)
  for(int i = 0; i < ni; i++){
    const int id = omp_get_thread_num();
    const int li = i_ind.getLeftIndex(i,lip,rip);
    const int ri = i_ind.getRightIndex(i,lip,rip);

    ret_vec[id] += m(li,ri);
  }

  for(int i=0;i<n_threads;i++) into += ret_vec[i];
	 
#endif
  
  return into;
}




//Basic implementation for testing
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace_slow(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &m){
  //Check the indices match
  if(! m.getRowParams().paramsEqual( m.getColParams() ) )
    ERR.General("","trace(const A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> &)","Illegal trace: underlying index parameters must match\n");
  
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  ScalarComplexType into(0,0);

  const int nv = m.getRowParams().getNv();

  const int n_threads = omp_get_max_threads();
  std::vector<ScalarComplexType, BasicAlignedAllocator<ScalarComplexType> > ret_vec(n_threads,(0.,0.));
  
#pragma omp parallel for schedule(static)
  for(int i = 0; i < nv; i++){
    const int id = omp_get_thread_num();
    ret_vec[id] += m.elem(i,i);
  }
  for(int i=0;i<n_threads;i++) into += ret_vec[i];  
  return into;
}




//Compute   m^ii(t1,t2)  for an arbitrary vector of meson fields
//This is both threaded and distributed over nodes
template<typename mf_Policies, 
	 template <typename> class A2AfieldL,  template <typename> class A2AfieldR
	 >
void trace(std::vector<typename mf_Policies::ScalarComplexType> &into, const std::vector<A2AmesonField<mf_Policies,A2AfieldL,A2AfieldR> > &m){
  //Distribute load over all nodes
  const int nmf = m.size();
  into.resize(nmf);
  for(int i=0;i<nmf;i++) into[i] = typename mf_Policies::ScalarComplexType(0);

  const int work = nmf;
  bool do_work;
  int node_work, node_off;
  getNodeWork(work, node_work, node_off, do_work);

#ifndef MEMTEST_MODE
  if(do_work){
    for(int t=node_off; t<node_off + node_work; t++){
      into[t] = trace(m[t]);
    }
  }
  globalSum( (typename mf_Policies::ScalarComplexType::value_type*)&into[0],2*nmf); //give all nodes a copy
#endif
}
