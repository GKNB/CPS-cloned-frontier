#pragma once

#include <chrono>
#include <thread>

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
  int base = dclock();
 
  size_t sz = nval*sizeof(int);
  double sz_MB = double(sz)/1024./1024.;
  double sz_GB = double(sz)/1024./1024./1024.;
  std::cout << "Data size " << sz_MB << " MB" << std::endl;

  int v = 0;

  {
    //Plain memory bandwidth
    int* p = (int*)malloc(sz);
    for(int i=0;i<nval;i++){
      p[i] = base+i;
    }    
    
    for(int i=0;i<nval;i++){
      v += p[i];
    }    
  
    for(int t=0;t<20;t++){
      double time = 0;
      for(int i=0;i<ntest;i++){
	//int volatile *rp = p;

	markit();
	time -= dclock();
	for(int j=0;j<nval;j++){
	  //(void)rp[j] ; //*base;
	  v += p[j];
	}    
	time += dclock();
	markit();
      }
      
      std::cout << "Hot memory read: " << ntest*sz_GB/time << " GB/s, read time: " << time << "s" << std::endl;
      std::cout << v << std::endl;
    }
    free(p);
  }


  store.alloc(0,sz);
  {
    CPSautoView(store_v,store,HostWrite); 
    int* iptr = (int*)store_v();
    
    for(int i=0;i<nval;i++){
      iptr[i] = base+i;
    } 
  }
  store.flush();
 
  //memset(store.data(), 0, sz);

  //Hot read
  {
    CPSautoView(store_v,store,HostRead);
    int const* iptr = (int const*)store_v();
    for(int i=0;i<nval;i++){
      v += iptr[i];
    }    

    for(int t=0;t<20;t++){
      double time = 0;
      for(int i=0;i<ntest;i++){
	time -= dclock();
	for(int j=0;j<nval;j++){
	  v += iptr[j] ; //*base;
	}    
	time += dclock();
      }
    
      std::cout << "Hot read: " << ntest*sz_GB/time << " GB/s, read time: " << time << "s" << std::endl;

      std::cout << v << std::endl;
    }
  }


  //Cold read 
  for(int t=0;t<20;t++){
    double time = 0;
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
      CPSautoView(store_v,store,HostRead);
      int const *iptr = (int const*)store_v();

      time -= dclock();
      for(int j=0;j<nval;j++){
	v += iptr[j];   //*base;
      }
      time += dclock();
    }
  
    std::cout << "Cold read: " << ntest*sz_GB/time << " GB/s,  read time: " << time << ", free time: " << free_time << "s, alloc time: " << alloc_time << "s, write time: " << write_time << ", flush time: " << flush_time << "s" << std::endl;

    std::cout << v << std::endl;
  }


  //Cold read madvise distribute
  for(int t=0;t<20;t++){
    store.alloc(0,sz);
    {
      CPSautoView(store_v,store,HostWrite);
      int* iptr = (int*)store_v();
    
      for(int i=0;i<nval;i++){
	iptr[i] = base+i;
      }
    }
    store.flush();
    store.distribute();

    double time = 0;
    double distribute_time = 0;
  
    for(int i=0;i<ntest;i++){
      {
	CPSautoView(store_v,store,HostRead);
	int const* iptr = (int const*)store_v();
	time -= dclock();
	for(int j=0;j<nval;j++){
	  v += iptr[j];   //*base;
	}
	time += dclock();
      }

      distribute_time -= dclock();
      store.distribute();
      distribute_time += dclock();
    }
  
    std::cout << "Cold read madvice distribute: " << ntest*sz_GB/time << " GB/s,  read time: " << time << ", distribute time: " << distribute_time << "s" << std::endl;

    std::cout << v << std::endl;
  }



  //Cold read with gather
  for(int t=0;t<20;t++){
    double time = 0;
    double free_time = 0;
    double alloc_time = 0;
    double gather_time = 0;
  
    for(int i=0;i<ntest;i++){
      free_time -= dclock();
      store.freeMem();
      free_time += dclock();

      alloc_time -= dclock();
      store.alloc(0,sz);
      alloc_time += dclock();

      //Pages are initially not host resident so reading a newly allocated region is performed from disk
      gather_time -= dclock();
      store.gather(true);
      gather_time += dclock();

      {
	CPSautoView(store_v,store,HostRead);
	int const* iptr = (int const*)store_v();
	time -= dclock();
	for(int j=0;j<nval;j++){
	  v += iptr[j];   //*base;
	}
	time += dclock();
      }
    }
  
    std::cout << "Cold read with gather: " << ntest*sz_GB/time << " GB/s,  read time: " << time << ", free time: " << free_time << "s, alloc time: " << alloc_time << "s, gather time: " << gather_time << "s" << std::endl;

    std::cout << v << std::endl;
  }


  //Cold read madvise distribute and gather
  for(int t=0;t<20;t++){
    store.alloc(0,sz);
    {
      CPSautoView(store_v,store,HostWrite);
      int* iptr = (int*)store_v();   
      for(int i=0;i<nval;i++){
	iptr[i] = base+i;
      }
    }
    store.flush();
    store.distribute();

    double time = 0;
    double distribute_time = 0;
    double gather_time = 0;

    for(int i=0;i<ntest;i++){
      gather_time -= dclock();
      store.gather(true);
      gather_time += dclock();

      {
	CPSautoView(store_v,store,HostRead);
	int const* iptr = (int const*)store_v();
	time -= dclock();
	for(int j=0;j<nval;j++){
	  v += iptr[j];   //*base;
	}
	time += dclock();
      }

      distribute_time -= dclock();
      store.distribute();
      distribute_time += dclock();
    }
  
    std::cout << "Cold read madvice distribute/gather: " << ntest*sz_GB/time << " GB/s,  read time: " << time << ", distribute time: " << distribute_time << "s, gather time: " << gather_time << "s" << std::endl;

    std::cout << v << std::endl;
  }


}

template<typename vtype> using iDataType2 = Grid::iVector<Grid::iVector<Grid::iVector<Grid::iVector<vtype, Grid::Nc>, Grid::Ns>, Grid::Ngp>, 2 >;
template<typename vtype> using iDataType4 = Grid::iVector<Grid::iVector<Grid::iVector<Grid::iVector<vtype, Grid::Nc>, Grid::Ns>, Grid::Ngp>, 4 >;
template<typename vtype> using iDataType8 = Grid::iVector<Grid::iVector<Grid::iVector<Grid::iVector<vtype, Grid::Nc>, Grid::Ns>, Grid::Ngp>, 8 >;
template<typename vtype> using iDataType16 = Grid::iVector<Grid::iVector<Grid::iVector<Grid::iVector<vtype, Grid::Nc>, Grid::Ns>, Grid::Ngp>, 16 >;

template<typename A2Apolicies>
void benchmarkPackGridFieldIO(typename A2Apolicies::FgridGFclass &lattice, int nfield=32){
#ifndef HAVE_LIME
  return;
#else
  using namespace Grid;
  assert(nfield % 16 == 0);
  
  GridCartesian *UGrid = lattice.getUGrid();
  GridRedBlackCartesian *UrbGrid = lattice.getUrbGrid();
  GridCartesian *FGrid = lattice.getFGrid();
  GridRedBlackCartesian *FrbGrid = lattice.getFrbGrid();
  LatticeGaugeFieldD *Umu = lattice.getUmu();

  int nodes = 1;
  for(int i=0;i<4;i++) nodes *= GJP.Nodes(i);

  std::string data_file = "tmp";

  {
    typedef iGparitySpinColourVector<vComplexD> GparitySpinColourVectorD;
    typedef Grid::Lattice<GparitySpinColourVectorD> GparityLatticeFermionD;
    
    std::vector<GparityLatticeFermionD> data(nfield, UGrid); //4D full-grid fermions
    size_t field_bytes = nodes*data[0].oSites()*sizeof(GparitySpinColourVectorD);
    std::cout << "Field size " << field_bytes << " bytes" << std::endl;

    Grid::emptyUserRecord record;
    Grid::ScidacWriter WR(UGrid->IsBoss());
    double time = -dclock();
    WR.open(data_file);
    for(int k=0; k<nfield;k++){
      WR.writeScidacFieldRecord(data[k],record);
    }
    WR.close();
    time += dclock();
    std::cout << "Wrote " << nfield << " fields of size " << double(field_bytes)/1024./1024. << "MB in " << time << "s: rate " << nfield*double(field_bytes)/1024./1024./time << "MB/s" << std::endl;
  }
  std::cout << "-----------------------------------------------" << std::endl;
  {
    typedef iDataType2<vComplexD> DataTypeD;
    typedef Grid::Lattice<DataTypeD> LatticeTypeD;
    
    int nfieldt = nfield/2;

    std::vector<LatticeTypeD> data(nfieldt, UGrid); //4D full-grid fermions
    size_t field_bytes = nodes*data[0].oSites()*sizeof(DataTypeD);
    std::cout << "Field size " << field_bytes << " bytes" << std::endl;

    Grid::emptyUserRecord record;
    Grid::ScidacWriter WR(UGrid->IsBoss());
    double time = -dclock();
    WR.open(data_file);
    for(int k=0; k<nfieldt;k++){
      WR.writeScidacFieldRecord(data[k],record);
    }
    WR.close();
    time += dclock();
    std::cout << "Wrote " << nfieldt << " fields of size " << double(field_bytes)/1024./1024. << "MB in " << time << "s: rate " << nfieldt*double(field_bytes)/1024./1024./time << "MB/s" << std::endl;
  }
  std::cout << "-----------------------------------------------" << std::endl;
  {
    typedef iDataType4<vComplexD> DataTypeD;
    typedef Grid::Lattice<DataTypeD> LatticeTypeD;
    
    int nfieldt = nfield/4;

    std::vector<LatticeTypeD> data(nfieldt, UGrid); //4D full-grid fermions
    size_t field_bytes = nodes*data[0].oSites()*sizeof(DataTypeD);
    std::cout << "Field size " << field_bytes << " bytes" << std::endl;

    Grid::emptyUserRecord record;
    Grid::ScidacWriter WR(UGrid->IsBoss());
    double time = -dclock();
    WR.open(data_file);
    for(int k=0; k<nfieldt;k++){
      WR.writeScidacFieldRecord(data[k],record);
    }
    WR.close();
    time += dclock();
    std::cout << "Wrote " << nfieldt << " fields of size " << double(field_bytes)/1024./1024. << "MB in " << time << "s: rate " << nfieldt*double(field_bytes)/1024./1024./time << "MB/s" << std::endl;
  }
  std::cout << "-----------------------------------------------" << std::endl;
  {
    typedef iDataType8<vComplexD> DataTypeD;
    typedef Grid::Lattice<DataTypeD> LatticeTypeD;
    
    int nfieldt = nfield/8;

    std::vector<LatticeTypeD> data(nfieldt, UGrid); //4D full-grid fermions
    size_t field_bytes = nodes*data[0].oSites()*sizeof(DataTypeD);
    std::cout << "Field size " << field_bytes << " bytes" << std::endl;

    Grid::emptyUserRecord record;
    Grid::ScidacWriter WR(UGrid->IsBoss());
    double time = -dclock();
    WR.open(data_file);
    for(int k=0; k<nfieldt;k++){
      WR.writeScidacFieldRecord(data[k],record);
    }
    WR.close();
    time += dclock();
    std::cout << "Wrote " << nfieldt << " fields of size " << double(field_bytes)/1024./1024. << "MB in " << time << "s: rate " << nfieldt*double(field_bytes)/1024./1024./time << "MB/s" << std::endl;
  }
  std::cout << "-----------------------------------------------" << std::endl;
  {
    typedef iDataType16<vComplexD> DataTypeD;
    typedef Grid::Lattice<DataTypeD> LatticeTypeD;
    
    int nfieldt = nfield/16;

    std::vector<LatticeTypeD> data(nfieldt, UGrid); //4D full-grid fermions
    size_t field_bytes = nodes*data[0].oSites()*sizeof(DataTypeD);
    std::cout << "Field size " << field_bytes << " bytes" << std::endl;

    Grid::emptyUserRecord record;
    Grid::ScidacWriter WR(UGrid->IsBoss());
    double time = -dclock();
    WR.open(data_file);
    for(int k=0; k<nfieldt;k++){
      WR.writeScidacFieldRecord(data[k],record);
    }
    WR.close();
    time += dclock();
    std::cout << "Wrote " << nfieldt << " fields of size " << double(field_bytes)/1024./1024. << "MB in " << time << "s: rate " << nfieldt*double(field_bytes)/1024./1024./time << "MB/s" << std::endl;
  }
#endif
}


CPS_END_NAMESPACE
