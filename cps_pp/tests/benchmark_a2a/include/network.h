#pragma once

CPS_START_NAMESPACE

void timeAllReduce(bool huge_pages){
#ifdef USE_MPI

  size_t bytes = 1024;
  size_t scale = 2;
  size_t max_bytes = 1024*1024*1024 + 1; //512MB max comm
  
  while(bytes < max_bytes){
    assert(bytes % sizeof(double) == 0);
    size_t ndouble = bytes / sizeof(double);

    assert( (size_t) ((int)ndouble) == ndouble ); //make sure it can be downcast to int

    double* buf;
    
    if(huge_pages){
      char shm_name [1024];
      struct passwd *pw = getpwuid (getuid());
      sprintf(shm_name,"/shm_%s",pw->pw_name);

      shm_unlink(shm_name);
      int fd=shm_open(shm_name,O_RDWR|O_CREAT,0666);
      if ( fd < 0 ) {   perror("failed shm_open");      assert(0);      }
      ftruncate(fd, bytes);
      
      int mmap_flag = MAP_SHARED;
#ifdef MAP_POPULATE
      mmap_flag |= MAP_POPULATE;
#endif
      //if (huge) mmap_flag |= MAP_HUGETLB;
      mmap_flag |= MAP_HUGETLB;
      buf =  (double*)mmap(NULL,bytes, PROT_READ | PROT_WRITE, mmap_flag, fd, 0);

      //std::cout << "SHM "<<ptr<< "("<< size<< "bytes)"<<std::endl;
      if ( (void*)buf == (void*)MAP_FAILED ) {
        perror("failed mmap");
        assert(0);
      }

    }else{
      buf = (double*)malloc(bytes);
    }

    int nrpt = 10;

    Float time = -dclock();
    {
      for(int n=0;n<nrpt;n++)
	MPI_Allreduce(MPI_IN_PLACE, buf, ndouble, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    time += dclock();

    time /= nrpt;

    Float MB =  ( (double)bytes )/1024./1024.;

    Float MB_per_s = MB / time;
    std::cout << "Size " << MB << " MB  time " << time << " secs, " << MB_per_s << " MB/s" << std::endl;

    if(huge_pages) munmap(buf,bytes);
    else free(buf);

    bytes *= scale;
  }
#endif
}


CPS_END_NAMESPACE
