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


CPS_END_NAMESPACE
