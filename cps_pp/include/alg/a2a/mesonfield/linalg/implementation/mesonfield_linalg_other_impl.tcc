//Compute   l^ij(t1,t2) r^ji(t3,t4)
//Threaded but *node local*
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
typename mf_Policies::ScalarComplexType trace(const A2AmesonField<mf_Policies,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Policies,rA2AfieldL,rA2AfieldR> &r){
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
  const int n_threads = omp_get_max_threads();
  std::vector<ScalarComplexType, BasicAlignedAllocator<ScalarComplexType> > ret_vec(n_threads,(0.,0.));
    
  modeIndexSet lip; lip.time = times[0];
  modeIndexSet rip; rip.time = times[3];

  modeIndexSet ljp; ljp.time = times[1];
  modeIndexSet rjp; rjp.time = times[2];

  const int ni = i_ind.getNindices(lip,rip); //how many indices to loop over
  const int nj = j_ind.getNindices(ljp,rjp);

#ifndef MEMTEST_MODE

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
	 
#endif
  
  return into;
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
  QMP_sum_array( (typename mf_Policies::ScalarComplexType::value_type*)&into[0],2*nmf); //give all nodes a copy
#endif
}
