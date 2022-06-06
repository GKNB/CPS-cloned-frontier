#pragma once

CPS_START_NAMESPACE

void benchmarkCPSfieldIO(){
  const int nfield_tests[7] = {1,10,50,100,250,500,1000};

  for(int n=0;n<7;n++){
    const int nfield = nfield_tests[n];
    
    std::vector<CPSfermion4D<cps::ComplexD> > a(nfield);
    for(int i=0;i<nfield;i++) a[i].testRandom();
    const double mb_written = double(a[0].byte_size())/1024/1024*nfield;
    
    const int ntest = 10;

    double avg_rate = 0;
    
    for(int i=0;i<ntest;i++){
      std::ostringstream fname; fname << "field.test" << i << ".node" << UniqueID();
      std::ofstream f(fname.str().c_str());
      double time = -dclock();
      for(int j=0;j<nfield;j++) a[j].writeParallel(f);
      f.close();
      time += dclock();
      
      const double rate = mb_written/time;
      avg_rate += rate;
      if(!UniqueID()) printf("Test %d, wrote %f MB in %f s: rate %f MB/s\n",i,mb_written,time,rate);
    }
    avg_rate /= ntest;
    
    if(!UniqueID()) printf("Data size %f MB, avg rate %f MB/s\n",mb_written,avg_rate);
    
  }
}

void markit(){
  static int v = 0;
  v++;
}
  

void benchmarkMmapMemoryStorage(int ntest, int nval){
  MmapMemoryStorage store;

  size_t sz = nval*sizeof(int);
  double sz_MB = double(sz)/1024./1024.;
  double sz_GB = double(sz)/1024./1024./1024.;
  std::cout << "Data size " << sz_MB << " MB" << std::endl;
  store.alloc(0,sz);
  int* iptr = (int*)store.data();
  
  for(int i=0;i<nval;i++){
    iptr[i] = i;
  }    
  store.flush();
  

  
  //memset(store.data(), 0, sz);

  //Hot read
  int v = 0;

  for(int i=0;i<nval;i++){
    v = iptr[i];
  }    
 
  double time = 0;
  for(int i=0;i<ntest;i++){
    time -= dclock();
    for(int j=0;j<nval;j++){
      v += iptr[j];
    }    
    time += dclock();
  }
  
  std::cout << "Hot read: " << ntest*sz_GB/time << " GB/s, read time: " << time << "s" << std::endl;

  //Cold read 
  time = 0;
  double free_time = 0;
  double alloc_time = 0;
  double flush_time = 0;
  double write_time = 0;
  
  for(int i=0;i<ntest;i++){
    free_time -= dclock();
    store.freeMem();
    free_time += dclock();

    alloc_time -= dclock();
    store.alloc(0,sz);
    alloc_time += dclock();

    write_time -= dclock();
    iptr = (int*)store.data();
    for(int j=0;j<nval;j++){
      iptr[j] = j;
    }    
    write_time += dclock();
    
    flush_time -= dclock();
    store.flush();
    flush_time += dclock();
       
    time -= dclock();
    markit();
    for(int j=0;j<nval;j++){
      v += iptr[j];
    }
    markit();
    time += dclock();
  }
  
  std::cout << "Cold read: " << ntest*sz_GB/time << " GB/s,  read time: " << time << ", free time: " << free_time << "s, alloc time: " << alloc_time << "s, write time: " << write_time << ", flush time: " << flush_time << "s" << std::endl;

  

}





CPS_END_NAMESPACE
