#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void benchmarkMesonFieldUnpack(const A2AArg &a2a_args,const int ntest){
  std::cout << "Timing mesonfield unpack" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);

  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t into_size = rows_full * cols_full * sizeof(Complex);
  Complex* into = (Complex*)malloc(into_size);
  
  double time = 0;
  for(int i=0;i<ntest;i++){
    mf1.testRandom();
    time -= dclock();
    mf1.unpack(into);
    time += dclock();
  }

  std::cout << "Cold " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;

  time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    mf1.unpack(into);
    time += dclock();
  }

  std::cout << "Hot " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;
  free(into);
}



template<typename A2Apolicies>
void benchmarkMesonFieldPack(const A2AArg &a2a_args,const int ntest){
  std::cout << "Timing mesonfield pack" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);

  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t into_size = rows_full * cols_full * sizeof(Complex);
  Complex* from = (Complex*)malloc(into_size);  
  
  double time = 0;
  for(int i=0;i<ntest;i++){
    memset(from, i, into_size);    
    time -= dclock();
    mf1.pack(from);
    time += dclock();
  }

  std::cout << "Cold " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;

  time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    mf1.pack(from);
    time += dclock();
  }

  std::cout << "Hot " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;
  free(from);
}


template<typename A2Apolicies>
void benchmarkMesonFieldUnpackDevice(const A2AArg &a2a_args,const int ntest){
#ifdef GPU_VEC
  std::cout << "Timing mesonfield unpack device version" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);
  mf1.testRandom();
  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t into_size = rows_full * cols_full * sizeof(Complex);
  Complex* into = (Complex*)device_alloc_check(into_size);  
  
  double time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    mf1.unpack_device(into); //involves device copy
    time += dclock();
  }

  std::cout << "Cold " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;

  CPSautoView(mf1_v,mf1);
  
  time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    mf1.unpack_device(into, &mf1_v);
    time += dclock();
  }

  std::cout << "Hot " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;
  device_free(into);
#endif
}

template<typename A2Apolicies>
void benchmarkMesonFieldPackDevice(const A2AArg &a2a_args,const int ntest){
#ifdef GPU_VEC
  std::cout << "Timing mesonfield pack device version" << std::endl;

  LRG.AssignGenerator(0); //always uses the RNG at coord 0 on node 0 - should always be the same one!

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1;
  mf1.setup(a2a_args,a2a_args,0,0);

  typedef typename A2Apolicies::ScalarComplexType Complex;

  int rows_full = mf1.getNrowsFull();
  int cols_full = mf1.getNcolsFull();

  size_t unpacked_size = rows_full * cols_full * sizeof(Complex);
  Complex* u = (Complex*)device_alloc_check(unpacked_size);  

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf1_p;
  mf1_p.setup(a2a_args,a2a_args,0,0);
  
  double time = 0;
  for(int i=0;i<ntest;i++){
    mf1.testRandom();
    mf1.unpack_device(u);    
    time -= dclock();
    mf1_p.pack_device(u);
    time += dclock();
  }

  std::cout << "Cold " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;

  time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    mf1_p.pack_device(u);
    time += dclock();
  }

  std::cout << "Hot " << ntest << " iterations, avg time " << time / ntest << "s" << std::endl;
  device_free(u);
#endif
}


void benchmarkMesonFieldGather(const A2AArg &a2a_args,const int ntest){
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  int nodes = 1;
  for(int i=0;i<4;i++) nodes *= GJP.Nodes(i);

  //Distributed storage
  if(nodes > 1){
    std::cout << "Benchmarking distributed storage" << std::endl;
    
    A2APOLICIES_TEMPLATE(A2ApoliciesTmp, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_AUTOMATIC_ALLOC, SET_MFSTORAGE_DISTRIBUTED);
    typedef A2AmesonField<A2ApoliciesTmp,A2AvectorWfftw,A2AvectorVfftw> MfType;
    std::vector<MfType> mf(nodes); //arrange for 1 per node

    std::vector<MfType*> owner_map(nodes);
    for(int f=0;f<nodes;f++){
      int t = f % Lt;
      mf[f].setup(a2a_args,a2a_args,t,t);
      mf[f].testRandom();
      mf[f].nodeDistribute();
      owner_map[mf[f].masterUID()] = &mf[f]; //only need one per partner node
    }
    
    for(int f=0;f<nodes;f++){
      double avg_time = 0;
      double var_time = 0;

      DistributedMemoryStorage::perf().reset();
      
      for(int t=0;t<ntest;t++){
	double time = -dclock();
	owner_map[f]->nodeGet();
	time += dclock();

	owner_map[f]->nodeDistribute();
	
	avg_time += time;
	var_time += time*time;
      }
      avg_time /= ntest;
      var_time = var_time/ntest - avg_time*avg_time;

      if(!UniqueID()) printf("Owner node uid %d,  this node uid %d, time  %f +- %f\n", f, UniqueID(), avg_time, sqrt(var_time));

      DistributedMemoryStorage::perf().print();
    }
    std::cout << "-----------------------------------" << std::endl;
  }

  //One-sided distributed storage
  if(nodes > 1){
    std::cout << "Benchmarking one-sided distributed storage" << std::endl;
    
    A2APOLICIES_TEMPLATE(A2ApoliciesTmp, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_AUTOMATIC_ALLOC, SET_MFSTORAGE_DISTRIBUTEDONESIDED);
    typedef A2AmesonField<A2ApoliciesTmp,A2AvectorWfftw,A2AvectorVfftw> MfType;
    std::vector<MfType> mf(nodes); //arrange for 1 per node

    std::vector<MfType*> owner_map(nodes);
    for(int f=0;f<nodes;f++){
      int t = f % Lt;
      mf[f].setup(a2a_args,a2a_args,t,t);
      mf[f].testRandom();
      mf[f].nodeDistribute();
      owner_map[mf[f].masterUID()] = &mf[f]; //only need one per partner node
    }
    
    for(int f=0;f<nodes;f++){
      double avg_time = 0;
      double var_time = 0;

      DistributedMemoryStorageOneSided::perf().reset();
      
      for(int t=0;t<ntest;t++){
	double time = -dclock();
	owner_map[f]->nodeGet();
	time += dclock();

	owner_map[f]->nodeDistribute();
	
	avg_time += time;
	var_time += time*time;
      }
      avg_time /= ntest;
      var_time = var_time/ntest - avg_time*avg_time;

      if(!UniqueID()) printf("Owner node uid %d,  this node uid %d, time  %f +- %f\n", f, UniqueID(), avg_time, sqrt(var_time));

      DistributedMemoryStorageOneSided::perf().print();
    }
    std::cout << "-----------------------------------" << std::endl;
  }

  //Burst buffer storage (only one head node:  assumes shared scratch)
  {
    std::cout << "Benchmarking burst-buffer distributed storage" << std::endl;
    
    A2APOLICIES_TEMPLATE(A2ApoliciesTmp, 1, BaseGridPoliciesGparity, SET_A2AVECTOR_AUTOMATIC_ALLOC, SET_MFSTORAGE_BURSTBUFFER);
    typedef A2AmesonField<A2ApoliciesTmp,A2AvectorWfftw,A2AvectorVfftw> MfType;
    std::vector<MfType> mf(nodes); //arrange for 1 per node

    for(int f=0;f<nodes;f++){
      int t = f % Lt;
      mf[f].setup(a2a_args,a2a_args,t,t);
      mf[f].testRandom();
      mf[f].nodeDistribute();
    }
    
    //only need to do this once but time from the perspective of all nodes (recall only head node writes)
    //this first test is "hot", i.e. with the same data, so caching will possibly come into play
    double avg_time = 0;
    double var_time = 0;
    
    BurstBufferMemoryStorage::perf().reset();
    
    for(int t=0;t<ntest;t++){
      double time = -dclock();
      mf[0].nodeGet();
      time += dclock();
      
      mf[0].nodeDistribute();
      
      avg_time += time;
      var_time += time*time;
    }
    avg_time /= ntest;
    var_time = var_time/ntest - avg_time*avg_time;

    for(int f=0;f<nodes;f++){
      if(UniqueID() == f) printf("Node uid %d, time  %f +- %f (hot)\n", UniqueID(), avg_time, sqrt(var_time));
      cps::sync();
    }
    
    BurstBufferMemoryStorage::perf().print();


    avg_time = 0;
    var_time = 0;
    
    BurstBufferMemoryStorage::perf().reset();
    
    for(int t=0;t<ntest;t++){
      mf[0].nodeGet();	    
      mf[0].testRandom();
      mf[0].nodeDistribute(); //checksum will differ so it forces a rewrite
      
      double time = -dclock();
      mf[0].nodeGet();
      time += dclock();
      
      avg_time += time;
      var_time += time*time;
    }
    avg_time /= ntest;
    var_time = var_time/ntest - avg_time*avg_time;

    for(int f=0;f<nodes;f++){
      if(UniqueID() == f) printf("Node uid %d, time  %f +- %f (cold)\n", UniqueID(), avg_time, sqrt(var_time));
      cps::sync();
    }
    
    BurstBufferMemoryStorage::perf().print();    
    
    std::cout << "-----------------------------------" << std::endl;
  }
}







CPS_END_NAMESPACE
