template<typename S>
struct _nodeGetManyPerf{ 
  static void reset(){}
  static void print(){}; 
};

template<>
struct _nodeGetManyPerf<DistributedMemoryStorage>{ 
  static void reset(){ DistributedMemoryStorage::perf().reset(); }; 
  static void print(){ 
    DistributedMemoryStorage::perf().print(); 
#ifdef DISTRIBUTED_MEMORY_STORAGE_REUSE_MEMORY
    if(!UniqueID()) DistributedMemoryStorage::block_allocator().stats(std::cout);
#endif
  }; 
};


//Handy helpers for gather and distribute of length Lt vectors of meson fields
template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, ...){
  _nodeGetManyPerf<typename T::MesonFieldDistributedStorageType>::reset();
  cps::sync();
  
  double time = -dclock();

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    a->operator[](t).nodeGet();
  }

  va_list vl;
  va_start(vl,a);
  for(int i=1; i<n; i++){
    std::vector<T>* val=va_arg(vl,std::vector<T>*);

    for(int t=0;t<Lt;t++){
      val->operator[](t).nodeGet();
    }
  }
  va_end(vl);

  print_time("nodeGetMany","Meson field gather",time+dclock());
  _nodeGetManyPerf<typename T::MesonFieldDistributedStorageType>::print();
}


template<typename T>
void nodeDistributeMany(const int n, std::vector<T> *a, ...){
  cps::sync();
  
  double time = -dclock();

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    a->operator[](t).nodeDistribute();
  }

  va_list vl;
  va_start(vl,a);
  for(int i=1; i<n; i++){
    std::vector<T>* val=va_arg(vl,std::vector<T>*);

    for(int t=0;t<Lt;t++){
      val->operator[](t).nodeDistribute();
    }
  }
  va_end(vl);

  print_time("nodeDistributeMany","Meson field distribute",time+dclock());
}


//Same as above but the user can pass in a set of bools that tell the gather whether the MF on that timeslice is required. If not it is internally deleted, freeing memory
template<typename T>
void nodeGetMany(const int n, std::vector<T> *a, std::vector<bool> const* a_timeslice_mask,  ...){
  _nodeGetManyPerf<typename T::MesonFieldDistributedStorageType>::reset();
  cps::sync();

  double time = -dclock();

  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  for(int t=0;t<Lt;t++){
    a->operator[](t).nodeGet(a_timeslice_mask->at(t));
  }

  va_list vl;
  va_start(vl,a_timeslice_mask);
  for(int i=1; i<n; i++){
    std::vector<T>* val=va_arg(vl,std::vector<T>*);
    std::vector<bool> const* timeslice_mask = va_arg(vl,std::vector<bool> const*);
    
    for(int t=0;t<Lt;t++){
      val->operator[](t).nodeGet(timeslice_mask->at(t));
    }
  }
  va_end(vl);

  print_time("nodeGetMany","Meson field gather",time+dclock());
  _nodeGetManyPerf<typename T::MesonFieldDistributedStorageType>::print();
}


//Distribute only meson fields in 'from' that are *not* present in any of the sets 'notina' and following
template<typename T>
void nodeDistributeUnique(std::vector<T> &from, const int n, std::vector<T> const* notina, ...){
  cps::sync();
  
  double time = -dclock();
  
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  
  std::set<T const*> exclude;

  for(int t=0;t<Lt;t++)
    exclude.insert(& notina->operator[](t) );

  va_list vl;
  va_start(vl,notina);
  for(int i=1; i<n; i++){
    std::vector<T> const* val=va_arg(vl,std::vector<T> const*);

    for(int t=0;t<Lt;t++)
      exclude.insert(& val->operator[](t) );
  }
  va_end(vl);

  for(int t=0;t<Lt;t++){
    if(exclude.count(&from[t]) == 0) from[t].nodeDistribute();
  }

  print_time("nodeDistributeUnique","Meson field distribute",time+dclock());
}


//Distribute all meson fields in to_distribute which are not present in to_keep
template<typename T>
void nodeDistributeUnique(const std::vector< std::vector<T>* > &to_distribute, const std::vector< std::vector<T> const*> &to_keep){
  cps::sync();
  
  double time = -dclock();

  std::set<T const*> keep_p;
  for(auto tv: to_keep)
    for(const T & p: *tv)
      keep_p.insert(&p);
  
  for(auto tv: to_distribute)
    for(T & p: *tv)
      if(!keep_p.count(&p)) p.nodeDistribute();

  print_time("nodeDistributeUnique(vector edition)","Meson field distribute",time+dclock());
}






