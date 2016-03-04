#ifndef MESON_FIELD_IMPL
#define MESON_FIELD_IMPL

#include<alg/a2a/mult_impl.h>
#include<alg/a2a/mult_vMv_split.h>
#include<alg/a2a/mult_vMv_impl.h>
#include<alg/a2a/mult_vv_impl.h>

template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::plus_equals(const A2AmesonField<mf_Float,A2AfieldL,A2AfieldR> &with, const bool &parallel){
  if(nmodes_l != with.nmodes_l || nmodes_r != with.nmodes_r || 
     !lindexdilution.paramsEqual(with.lindexdilution) || !rindexdilution.paramsEqual(with.rindexdilution) ){
    ERR.General("A2AmesonField","plus_equals(..)","Second meson field must have the same underlying parameters\n");
  }
  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) mf[i] += with.mf[i];
  }else{
    for(int i=0;i<fsize;i++) mf[i] += with.mf[i];
  }
}

template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::times_equals(const mf_Float f,const bool &parallel){
  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) mf[i] *= f;			       
  }else{
    for(int i=0;i<fsize;i++) mf[i] *= f;
  }
}



//Replace this meson field with the average of this and a second field, 'with'
template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::average(const A2AmesonField<mf_Float,A2AfieldL,A2AfieldR> &with, const bool &parallel){
  if(nmodes_l != with.nmodes_l || nmodes_r != with.nmodes_r || 
     !lindexdilution.paramsEqual(with.lindexdilution) || !rindexdilution.paramsEqual(with.rindexdilution) ){
    ERR.General("A2AmesonField","average(..)","Second meson field must have the same underlying parameters\n");
  }
  if(parallel){
#pragma omp_parallel for
    for(int i=0;i<fsize;i++) mf[i] = (mf[i] + with.mf[i])/2.0;
  }else{
    for(int i=0;i<fsize;i++) mf[i] = (mf[i] + with.mf[i])/2.0;
  }
}

//Reorder the rows so that all the elements in idx_map are sequential. Indices not in map are ignored. Use at your own risk
template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::rowReorder(A2AmesonField<mf_Float,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel) const{
  into.setup(lindexdilution, rindexdilution, tl, tr);

#define DOIT \
    int irow = idx_map[i]; \
    for(int j=0;j<nmodes_r;j++) \
      into(i,j) = (*this)(irow,j);

    //mf_Float const* p = mf + 2*( j + nmodes_r*i );
  if(parallel){
#pragma omp parallel for
    for(int i=0;i<map_size;i++){
      DOIT;
    }
  }else{
    for(int i=0;i<map_size;i++){
      DOIT;
    }
  }
#undef DOIT

}
template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::colReorder(A2AmesonField<mf_Float,A2AfieldL,A2AfieldR> &into, const int idx_map[], int map_size, bool parallel) const{
  into.setup(lindexdilution, rindexdilution, tl, tr);

#define DOIT \
    for(int j=0;j<map_size;j++){ \
      int jcol = idx_map[j]; \
      into(i,j) = (*this)(i,jcol); \
    }

  if(parallel){
#pragma omp parallel for
    for(int i=0;i<nmodes_l;i++){
      DOIT;
    }
  }else{
    for(int i=0;i<nmodes_l;i++){
      DOIT;
    } 
  }
}



//Do a column reorder but where we pack the row indices to exclude those not used (as indicated by input bool array)
//Output as a GSL matrix
template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
typename gsl_wrapper<mf_Float>::matrix_complex * A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::GSLpackedColReorder(const int idx_map[], int map_size, bool rowidx_used[], typename gsl_wrapper<mf_Float>::matrix_complex *reuse ) const{
  typedef gsl_wrapper<mf_Float> gw;
  assert(sizeof(typename gw::complex) == sizeof(std::complex<mf_Float>));
  int rows = nmodes_l;
  int cols = nmodes_r;

  int nrows_used = 0;
  for(int i_full=0;i_full<rows;i_full++) if(rowidx_used[i_full]) nrows_used++;

  typename gw::matrix_complex *M_packed;
  if(reuse!=NULL){
    M_packed = reuse;
    M_packed->size1 = nrows_used;
    M_packed->size2 = M_packed->tda = map_size;
  }else M_packed = gw::matrix_complex_alloc(nrows_used,map_size);

  //Look for contiguous blocks in the idx_map we can take advantage of
  std::vector<std::pair<int,int> > blocks;
  find_contiguous_blocks(blocks,idx_map,map_size);

  int i_packed = 0;

  for(int i_full=0;i_full<rows;i_full++){
    if(rowidx_used[i_full]){
      mf_Float const* mf_row_base = mf + 2*nmodes_r*i_full; //meson field are row major so columns are contiguous
      typename gw::complex* row_base = gw::matrix_complex_ptr(M_packed,i_packed,0); //GSL matrix are also row major
      for(int b=0;b<blocks.size();b++){
	mf_Float const* block_ptr = mf_row_base + 2*idx_map[blocks[b].first];
	memcpy((void*)row_base,(void*)block_ptr,2*blocks[b].second*sizeof(mf_Float));
	row_base += blocks[b].second;
      }
      i_packed++;
    }
  }

  return M_packed;
}



template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::transpose(A2AmesonField<mf_Float,A2AfieldR,A2AfieldL> &into) const{
  assert( (void*)this != (void*)&into );
  into.setup(rindexdilution, lindexdilution, tr, tl);
#pragma omp parallel for
  for(int i=0;i<nmodes_l;i++)
    for(int j=0;j<nmodes_r;j++)
      into(j,i) = (*this)(i,j);
}

//For all mode indices l_i and r_j, compute the meson field  \sum_p l_i^\dagger(p,t) M(p,t) r_j(p,t)
//It is assumed that A2AfieldL and A2AfieldR are Fourier transformed field containers
//M(p,t) is a completely general momentum-space spin/color/flavor matrix per temporal slice

template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::compute(const A2AfieldL<mf_Float> &l, const InnerProduct &M, const A2AfieldR<mf_Float> &r, const int &t, bool do_setup){
  if(do_setup) setup(l,r,t,t); //both vectors have same timeslice
  else zero();
  
  if(!UniqueID()) printf("Starting A2AmesonField::compute timeslice %d with %d threads\n",t, omp_get_max_threads());

  double time = -dclock();

  //For W vectors we dilute out the flavor index in-place while performing this contraction
  int size_3d = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites();
  int nl_l = lindexdilution.getNl();
  int nl_r = rindexdilution.getNl();

  int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();
  if(t_lcl >= 0 && t_lcl < GJP.TnodeSites()){ //if timeslice is on-node
    const int size_4d = GJP.VolNodeSites();
    const int flav_offset = size_4d * SPINOR_SIZE;

#pragma omp parallel for
    for(int i = 0; i < nmodes_l; i++){
      std::complex<double> mf_accum;

      modeIndexSet i_high_unmapped; if(i>=nl_l) lindexdilution.indexUnmap(i-nl_l,i_high_unmapped);

      for(int j = 0; j < nmodes_r; j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) rindexdilution.indexUnmap(j-nl_r,j_high_unmapped);

	mf_accum = 0.;

	for(int p_3d = 0; p_3d < size_3d; p_3d++) {

	  int site_offset = SPINOR_SIZE*(p_3d + size_3d*t_lcl);

	  SCFvectorPtr<mf_Float> lscf = l.getFlavorDilutedVect(i,i_high_unmapped,site_offset,flav_offset); //dilute flavor in-place if it hasn't been already
	  SCFvectorPtr<mf_Float> rscf = r.getFlavorDilutedVect(j,j_high_unmapped,site_offset,flav_offset);

	  mf_accum += M(lscf,rscf,p_3d,t); //produces double precision output by spec
	}
	(*this)(i,j) = mf_accum; //downcast after accumulate      
      }
    }
  }
  sync();
  print_time("A2AmesonField","local compute",time + dclock());
  time = -dclock();

  //Sum over all nodes so all nodes have a copy
  nodeSum();
  print_time("A2AmesonField","nodeSum",time + dclock());
}

//This version is more efficient on multi-nodes
template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
template<typename InnerProduct, typename Allocator>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::compute(std::vector<A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>, Allocator > &mf_t, const A2AfieldL<mf_Float> &l, const InnerProduct &M, const A2AfieldR<mf_Float> &r, bool do_setup){
  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  if(!UniqueID()) printf("Starting A2AmesonField::compute for %d timeslices with %d threads\n",Lt, omp_get_max_threads());

  double time = -dclock();
  mf_t.resize(Lt);
  for(int t=0;t<Lt;t++) 
    if(do_setup) mf_t[t].setup(l,r,t,t); //both vectors have same timeslice (zeroes the starting matrix)
    else mf_t[t].zero();

  //For W vectors we dilute out the flavor index in-place while performing this contraction
  const int size_3d = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites();
  const int size_4d = GJP.VolNodeSites();
  //const int flav_offset_l = l.getMode(0).fsiteFlavorOffset() * l.getMode(0).siteSize();
  //const int flav_offset_r = r.getMode(0).fsiteFlavorOffset() * r.getMode(0).siteSize();

  //Each node only works on its time block
  for(int t=GJP.TnodeCoor()*GJP.TnodeSites(); t<(GJP.TnodeCoor()+1)*GJP.TnodeSites(); t++){
    double ttime = -dclock();

    const int nl_l = mf_t[t].lindexdilution.getNl();
    const int nl_r = mf_t[t].rindexdilution.getNl();

    int t_lcl = t-GJP.TnodeCoor()*GJP.TnodeSites();

#pragma omp parallel for
    for(int i = 0; i < mf_t[t].nmodes_l; i++){
      std::complex<double> mf_accum;

      modeIndexSet i_high_unmapped; if(i>=nl_l) mf_t[t].lindexdilution.indexUnmap(i-nl_l,i_high_unmapped);

      for(int j = 0; j < mf_t[t].nmodes_r; j++) {
	modeIndexSet j_high_unmapped; if(j>=nl_r) mf_t[t].rindexdilution.indexUnmap(j-nl_r,j_high_unmapped);

	mf_accum = 0.;

	for(int p_3d = 0; p_3d < size_3d; p_3d++) {

	  //int site_offset = SPINOR_SIZE*(p_3d + size_3d*t_lcl);

	  //SCFvectorPtr<mf_Float> lscf = l.getFlavorDilutedVect(i,i_high_unmapped,site_offset,flav_offset_l); //dilute flavor in-place if it hasn't been already
	  //SCFvectorPtr<mf_Float> rscf = r.getFlavorDilutedVect(j,j_high_unmapped,site_offset,flav_offset_r);

	  SCFvectorPtr<mf_Float> lscf = l.getFlavorDilutedVect2(i,i_high_unmapped,p_3d,t_lcl); //dilute flavor in-place if it hasn't been already
	  SCFvectorPtr<mf_Float> rscf = r.getFlavorDilutedVect2(j,j_high_unmapped,p_3d,t_lcl);

	  mf_accum += M(lscf,rscf,p_3d,t); //produces double precision output by spec
	}
	mf_t[t](i,j) = mf_accum; //downcast after accumulate      
      }
    }
    std::ostringstream os; os << "timeslice " << t << " from range " << GJP.TnodeCoor()*GJP.TnodeSites() << " to " << (GJP.TnodeCoor()+1)*GJP.TnodeSites()-1;

    print_time("A2AmesonField",os.str().c_str(),ttime + dclock());
  }
  print_time("A2AmesonField","local compute",time + dclock());

  time = -dclock();
  sync();
  print_time("A2AmesonField","sync",time + dclock());

  //Accumulate
  time = -dclock();
  for(int t=0; t<Lt; t++) mf_t[t].nodeSum();
  print_time("A2AmesonField","nodeSum",time + dclock());
}




//Compute   l^ij(t1,t2) r^ji(t3,t4)
//Threaded but *node local*
template<typename mf_Float, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
std::complex<mf_Float> trace(const A2AmesonField<mf_Float,lA2AfieldL,lA2AfieldR> &l, const A2AmesonField<mf_Float,rA2AfieldL,rA2AfieldR> &r){
  //Check the indices match
  if(! l.getRowParams().paramsEqual( r.getColParams() ) || ! l.getColParams().paramsEqual( r.getRowParams() ) )
    ERR.General("","trace(mf_Float &into, const int &, const int &, const A2AmesonField &, const A2AmesonField &)","Illegal matrix product: underlying vector parameters must match\n");

  std::complex<mf_Float> into = std::complex<mf_Float>(0.,0.);

  typedef typename A2AmesonField<mf_Float,lA2AfieldL,lA2AfieldR>::LeftDilutionType DilType0;
  typedef typename A2AmesonField<mf_Float,lA2AfieldL,lA2AfieldR>::RightDilutionType DilType1;
  typedef typename A2AmesonField<mf_Float,rA2AfieldL,rA2AfieldR>::LeftDilutionType DilType2;
  typedef typename A2AmesonField<mf_Float,rA2AfieldL,rA2AfieldR>::RightDilutionType DilType3;

  ModeContractionIndices<DilType0,DilType3> i_ind(l.getRowParams());
  ModeContractionIndices<DilType1,DilType2> j_ind(r.getRowParams());

  int times[4] = { l.getRowTimeslice(), l.getColTimeslice(), r.getRowTimeslice(), r.getColTimeslice() };

  //W * W is only non-zero when the timeslice upon which we evaluate them are equal
  const int n_threads = omp_get_max_threads();
  std::vector< std::complex<mf_Float> > ret_vec(n_threads,(0.,0.));
    
  modeIndexSet lip; lip.time = times[0];
  modeIndexSet rip; rip.time = times[3];

  modeIndexSet ljp; ljp.time = times[1];
  modeIndexSet rjp; rjp.time = times[2];

  const int &ni = i_ind.getNindices(lip,rip); //how many indices to loop over
  const int &nj = j_ind.getNindices(ljp,rjp);

#pragma omp parallel for
  for(int i = 0; i < ni; i++){
    int id = omp_get_thread_num();
    int li = i_ind.getLeftIndex(i,lip,rip);
    int ri = i_ind.getRightIndex(i,lip,rip);

    for(int j = 0; j < nj; j++){
      int lj = j_ind.getLeftIndex(j,ljp,rjp);
      int rj = j_ind.getRightIndex(j,ljp,rjp);
      
      ret_vec[id] += l(li,lj) *  r(rj,ri);
    }
  }
  for(int i=0;i<n_threads;i++) into += ret_vec[i];

  return into;
}




//Compute   l^ij(t1,t2) r^ji(t3,t4) for all t1, t4  and place into matrix element t1,t4
//This is both threaded and distributed over nodes
template<typename mf_Float, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR
	 >
void trace(fMatrix<mf_Float> &into, const std::vector<A2AmesonField<mf_Float,lA2AfieldL,lA2AfieldR> > &l, const std::vector<A2AmesonField<mf_Float,rA2AfieldL,rA2AfieldR> > &r){
  //Distribute load over all nodes
  int lsize = l.size();
  int rsize = r.size();

  into.resize(lsize,rsize);

  int nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
  int work = lsize*rsize;

  bool do_work = true;
  if(nodes > work){
    nodes = work; if(UniqueID() >= work) do_work = false; //too many nodes, at least for this parallelization. Might want to consider parallelizing in a different way!
  }

  int node_work = work/nodes;
  if(node_work * nodes < work && !UniqueID()) node_work += work - node_work * nodes; //node 0 mops up remainder

  int node_off = UniqueID()*node_work;

  if(do_work){
    for(int tt=node_off; tt<node_off + node_work; tt++){
      int rem = tt;
      int tsnk = rem % lsize; rem /= lsize; //sink time
      int tsrc = rem; //source time

      into(tsnk,tsrc) = trace(l[tsnk],r[tsrc]);
    }
  }
  into.nodeSum(); //give all nodes a copy
}


struct nodeDistributeCounter{
  //Get a rank idx that cycles between 0... nodes-1
  static int getNext(){
    static int cur = 0;
    static int nodes = -1;
    if(nodes == -1){
      nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
    }
    int out = cur;
    cur = (cur + 1) % nodes;
    return out;
  }
};


//Delete all the data associated with this meson field apart from on node with UniqueID 'node'. The node index is saved so that the data can be later retrieved.
template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::nodeDistribute(int node_uniqueid){
  if(node_uniqueid == -1) node_uniqueid = nodeDistributeCounter::getNext(); //draw the next node index from the pool

  int nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
  if(node_uniqueid < 0 || node_uniqueid >= nodes) ERR.General("A2AmesonField","nodeDistribute","Invalid node rank %d\n", node_uniqueid);

#ifndef USE_MPI
  if(nodes > 1) ERR.General("A2AmesonField","nodeDistribute","Implementation requires MPI\n");
  //For one node, don't do anything
#else
  //if(!UniqueID()){ printf("nodeDistribute start with destination node UniqueID %d\n",node_uniqueid); fflush(stdout); }
  MPI_Barrier(MPI_COMM_WORLD);

  //MPI rank and CPS UniqueID may not be the same. Translate
  int my_rank;
  int ret = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","nodeDistribute","Comm_rank failed\n");

  if(mf == NULL){ printf("Error nodeDistribute: Rank %d has null pointer!\n",my_rank); fflush(stdout); exit(-1); }

  //printf("UniqueID %d -> MPI rank %d\n",UniqueID(),my_rank); fflush(stdout);

  int rank_tmp = (UniqueID() == node_uniqueid ? my_rank : 0);
  ret = MPI_Allreduce(&rank_tmp,&node_mpi_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); //node is now the MPI rank corresponding to UniqueID == _node
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","nodeDistribute","Reduce failed\n");
  MPI_Barrier(MPI_COMM_WORLD);

  //printf("UniqueID %d (MPI rank %d) got storage rank %d\n",UniqueID(),my_rank,node_mpi_rank); fflush(stdout);

  if(UniqueID() != node_uniqueid){
    //printf("UniqueID %d (MPI rank %d) free'd memory\n",UniqueID(),my_rank); fflush(stdout);
    free(mf); mf = NULL;
  }//else{ printf("UniqueID %d (MPI rank %d) is storage node, not freeing\n",UniqueID(),my_rank); fflush(stdout); }

  if(!UniqueID()) printf("A2AmesonField::nodeDistribute %f MB stored on node %d (MPI rank %d)\n",(double)byte_size()/(1024.*1024.), node_uniqueid, node_mpi_rank);
  //if(my_rank == node_mpi_rank) printf("A2AmesonField::nodeDistribute I am node with MPI rank %d and I have UniqueID %d, my first elem remains %f\n",my_rank,UniqueID(),mf[0]);
#endif
}

#ifdef USE_MPI

template<typename mf_Float>
struct getMPIdataType{
};
template<>
struct getMPIdataType<double>{
  static MPI_Datatype doit(){ return MPI_DOUBLE; }
};
template<>
struct getMPIdataType<float>{
  static MPI_Datatype doit(){ return MPI_FLOAT; }
};

#endif


//Get back the data. After the call, all nodes will have a complete copy
template<typename mf_Float, template <typename> class A2AfieldL,  template <typename> class A2AfieldR>
void A2AmesonField<mf_Float,A2AfieldL,A2AfieldR>::nodeGet(){
  if(node_mpi_rank == -1) return; //already on all nodes

#ifndef USE_MPI
  int nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);
  if(nodes > 1) ERR.General("A2AmesonField","nodeGet","Implementation requires MPI\n");
#else
  int mpi_rank; //get this node's MPI rank
  int ret = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if(ret != MPI_SUCCESS) ERR.General("A2AmesonField","nodeGet","Comm_rank failed\n");

  if(mpi_rank != node_mpi_rank){
    //if(mf != NULL) printf("rank %d pointer should be NULL but it isn't!\n",mpi_rank); fflush(stdout);
    mf = (mf_Float*)malloc(fsize * sizeof(mf_Float));  
    if(mf == NULL){ printf("rank %d failed to allocate memory!\n",mpi_rank); fflush(stdout); exit(-1); }
    //printf("rank %d allocated memory\n",mpi_rank); fflush(stdout);
  }//else{ printf("rank %d is root, first element of data %f\n",mpi_rank,mf[0]); fflush(stdout); }
  
  MPI_Datatype dtype = getMPIdataType<mf_Float>::doit();
  int dsize;
  MPI_Type_size(dtype,&dsize);
  if(sizeof(mf_Float) != dsize){
    if(!UniqueID()){ printf("MPI size mismatch, want %d, got %d\n",sizeof(mf_Float),dsize); fflush(stdout); }
    MPI_Barrier(MPI_COMM_WORLD);
    exit(-1);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ret = MPI_Bcast((void*)mf, fsize, dtype, node_mpi_rank, MPI_COMM_WORLD);
  if(ret != MPI_SUCCESS){
    printf("rank %d Bcast fail\n",mpi_rank,mf[0]); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    exit(-1);
  }
  //printf("rank %d completed Bcast, first element %f\n",mpi_rank,mf[0]); fflush(stdout);

  if(mpi_rank == node_mpi_rank) printf("A2AmesonField::nodeGet %f MB gathered from node %d (MPI rank %d)\n",(double)byte_size()/(1024.*1024.), UniqueID(), node_mpi_rank);

  node_mpi_rank = -1; //now on all nodes
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}



#endif








