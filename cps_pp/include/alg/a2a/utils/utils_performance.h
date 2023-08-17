#ifndef _CK_UTILS_PERFORMANCE_H__
#define _CK_UTILS_PERFORMANCE_H__

#include <util/lattice.h>
#include "utils_generic.h"
#ifdef USE_GRID
#include<util/lattice/fgrid.h>
#endif

CPS_START_NAMESPACE

#ifdef USE_GRID
//Returns per-node performance in Mflops of double/double Dirac op
template<typename GridPolicies>
double gridBenchmark(Lattice &lat){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  typedef typename GridDirac::GaugeField GridGaugeField;
  
  FgridFclass & lgrid = dynamic_cast<FgridFclass &>(lat);
  
  Grid::GridCartesian *FGrid = lgrid.getFGrid();
  Grid::GridCartesian *UGrid = lgrid.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lgrid.getUrbGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lgrid.getFrbGrid();

  GridGaugeField & Umu = *lgrid.getUmu();    

  //Setup Dirac operator
  const double mass = 0.1;
  const double mob_b = lgrid.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  
  typename GridDirac::ImplParams params;
  lgrid.SetParams(params);
    
  GridDirac Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_c, params);

  //Run benchmark
  LOGA2A << "gridBenchmark: Running Grid Dhop benchmark\n";
  LOGA2A << "Dirac operator type is : " << printType<GridDirac>() << std::endl;
  
  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  
  std::cout << Grid::GridLogMessage << "Initialising 4d RNG" << std::endl;
  Grid::GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);
  std::cout << Grid::GridLogMessage << "Initialising 5d RNG" << std::endl;
  Grid::GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  std::cout << Grid::GridLogMessage << "Initialised RNGs" << std::endl;
  
  GridFermionField src   (FGrid); random(RNG5,src);
  Grid::RealD N2 = 1.0/::sqrt(norm2(src));
  src = src*N2;
  
  GridFermionField result(FGrid); result = Grid::Zero();
  GridFermionField    ref(FGrid); ref = Grid::Zero();
  GridFermionField    tmp(FGrid);
  GridFermionField    err(FGrid);
  
  Grid::RealD NP = UGrid->_Nprocessors;
  Grid::RealD NN = UGrid->NodeCount();
  
  std::cout << Grid::GridLogMessage<< "* Vectorising space-time by "<<Grid::vComplexF::Nsimd()<<std::endl;
#ifdef GRID_OMP
  if ( Grid::WilsonKernelsStatic::Comms == Grid::WilsonKernelsStatic::CommsAndCompute ) std::cout << Grid::GridLogMessage<< "* Using Overlapped Comms/Compute" <<std::endl;
  if ( Grid::WilsonKernelsStatic::Comms == Grid::WilsonKernelsStatic::CommsThenCompute) std::cout << Grid::GridLogMessage<< "* Using sequential comms compute" <<std::endl;
#endif
  if ( Grid::WilsonKernelsStatic::Opt == Grid::WilsonKernelsStatic::OptGeneric   ) std::cout << Grid::GridLogMessage<< "* Using GENERIC Nc WilsonKernels" <<std::endl;
  if ( Grid::WilsonKernelsStatic::Opt == Grid::WilsonKernelsStatic::OptHandUnroll) std::cout << Grid::GridLogMessage<< "* Using Nc=3       WilsonKernels" <<std::endl;
  if ( Grid::WilsonKernelsStatic::Opt == Grid::WilsonKernelsStatic::OptInlineAsm ) std::cout << Grid::GridLogMessage<< "* Using Asm Nc=3   WilsonKernels" <<std::endl;
  std::cout << Grid::GridLogMessage<< "*****************************************************************" <<std::endl;
  
  int ncall =1000;

  FGrid->Barrier();
  //Ddwf.ZeroCounters(); no longer supported
  Ddwf.Dhop(src,result,0);
  std::cout<<Grid::GridLogMessage<<"Called warmup"<<std::endl;
  double t0=Grid::usecond();
  for(int i=0;i<ncall;i++){
    Ddwf.Dhop(src,result,0);
  }
  double t1=Grid::usecond();
  FGrid->Barrier();

  double volume=GJP.Snodes()*GJP.SnodeSites();  for(int mu=0;mu<4;mu++) volume=volume*GJP.NodeSites(mu)*GJP.Nodes(mu);
  double flops=(GJP.Gparity() + 1)*1344*volume*ncall;

  std::cout<<Grid::GridLogMessage << "Called Dw "<<ncall<<" times in "<<t1-t0<<" us"<<std::endl;
  std::cout<<Grid::GridLogMessage << "mflop/s =   "<< flops/(t1-t0)<<std::endl;
  std::cout<<Grid::GridLogMessage << "mflop/s per rank =  "<< flops/(t1-t0)/NP<<std::endl;
  std::cout<<Grid::GridLogMessage << "mflop/s per node =  "<< flops/(t1-t0)/NN<<std::endl;
  //Ddwf.Report(); no longer supported
  return flops/(t1-t0)/NN; //node performance in Mflops
}

//Returns per-node performance in Mflops of single/single and single/half Dirac ops, respectively
template<typename GridPolicies>
std::pair<double,double> gridBenchmarkSinglePrec(Lattice &lat){
  typedef typename GridPolicies::GridFermionField GridFermionField;
  typedef typename GridPolicies::GridFermionFieldF GridFermionFieldF;
  typedef typename GridPolicies::FgridFclass FgridFclass;
  typedef typename GridPolicies::GridDirac GridDirac;
  typedef typename GridPolicies::GridDiracF GridDiracF;
  typedef typename GridPolicies::GridDiracFH GridDiracFH;
  
  typedef typename GridDirac::GaugeField GridGaugeField;
  typedef typename GridDiracF::GaugeField GridGaugeFieldF;
  
  FgridFclass & lgrid = dynamic_cast<FgridFclass &>(lat);
  
  Grid::GridCartesian *FGrid = lgrid.getFGrid();
  Grid::GridCartesian *UGrid = lgrid.getUGrid();
  Grid::GridRedBlackCartesian *UrbGrid = lgrid.getUrbGrid();
  Grid::GridRedBlackCartesian *FrbGrid = lgrid.getFrbGrid();

  GridGaugeField & Umu = *lgrid.getUmu();

  //Setup Dirac operator
  const double mass = 0.1;
  const double mob_b = lgrid.get_mob_b();
  const double mob_c = mob_b - 1.;   //b-c = 1
  const double M5 = GJP.DwfHeight();
  
  typename GridDiracF::ImplParams params;
  lgrid.SetParams(params);

  //Make single prec Grids
  std::vector<int> nodes(4);
  std::vector<int> vol(4);
  for(int i=0;i<4;i++){
    vol[i]= GJP.NodeSites(i)*GJP.Nodes(i);;
    nodes[i]= GJP.Nodes(i);
  }
  Grid::GridCartesian *UGrid_f = Grid::SpaceTimeGrid::makeFourDimGrid(vol,Grid::GridDefaultSimd(Grid::Nd,Grid::vComplexF::Nsimd()),nodes);
  Grid::GridCartesian *FGrid_f = Grid::SpaceTimeGrid::makeFiveDimGrid(GJP.SnodeSites()*GJP.Snodes(),UGrid_f);
  Grid::GridRedBlackCartesian *UrbGrid_f = Grid::SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid_f);
  Grid::GridRedBlackCartesian *FrbGrid_f = Grid::SpaceTimeGrid::makeFiveDimRedBlackGrid(GJP.SnodeSites()*GJP.Snodes(),UGrid_f);

  Grid::RealD NP = UGrid_f->_Nprocessors;
  Grid::RealD NN = UGrid_f->NodeCount();

  //Gauge field and Dirac ops
  GridGaugeFieldF Umu_f(UGrid_f);
  precisionChange(Umu_f,Umu);
  
  GridDiracF Ddwf_F(Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,mass,M5,mob_b,mob_c, params);
  GridDiracFH Ddwf_FH(Umu_f,*FGrid_f,*FrbGrid_f,*UGrid_f,*UrbGrid_f,mass,M5,mob_b,mob_c, params);

  //Source
  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  
  std::cout << Grid::GridLogMessage << "Initialising 4d RNG" << std::endl;
  Grid::GridParallelRNG          RNG4(UGrid_f);  RNG4.SeedFixedIntegers(seeds4);
  std::cout << Grid::GridLogMessage << "Initialising 5d RNG" << std::endl;
  Grid::GridParallelRNG          RNG5(FGrid_f);  RNG5.SeedFixedIntegers(seeds5);
  std::cout << Grid::GridLogMessage << "Initialised RNGs" << std::endl;
  
  GridFermionFieldF src   (FGrid_f); random(RNG5,src);
  Grid::RealD N2 = 1.0/::sqrt(norm2(src));
  src = src*N2;

  GridFermionFieldF result(FGrid_f); result = Grid::Zero();
  GridFermionFieldF    ref(FGrid_f); ref = Grid::Zero();
  GridFermionFieldF    tmp(FGrid_f);
  GridFermionFieldF    err(FGrid_f);

  std::cout << "gridBenchmark: Running Grid Dhop benchmark\n";

  std::cout << Grid::GridLogMessage<< "* Vectorising space-time by "<<Grid::vComplexF::Nsimd()<<std::endl;
#ifdef GRID_OMP
  if ( Grid::WilsonKernelsStatic::Comms == Grid::WilsonKernelsStatic::CommsAndCompute ) std::cout << Grid::GridLogMessage<< "* Using Overlapped Comms/Compute" <<std::endl;
  if ( Grid::WilsonKernelsStatic::Comms == Grid::WilsonKernelsStatic::CommsThenCompute) std::cout << Grid::GridLogMessage<< "* Using sequential comms compute" <<std::endl;
#endif
  if ( Grid::WilsonKernelsStatic::Opt == Grid::WilsonKernelsStatic::OptGeneric   ) std::cout << Grid::GridLogMessage<< "* Using GENERIC Nc WilsonKernels" <<std::endl;
  if ( Grid::WilsonKernelsStatic::Opt == Grid::WilsonKernelsStatic::OptHandUnroll) std::cout << Grid::GridLogMessage<< "* Using Nc=3       WilsonKernels" <<std::endl;
  if ( Grid::WilsonKernelsStatic::Opt == Grid::WilsonKernelsStatic::OptInlineAsm ) std::cout << Grid::GridLogMessage<< "* Using Asm Nc=3   WilsonKernels" <<std::endl;
  std::cout << Grid::GridLogMessage<< "*****************************************************************" <<std::endl;
  int ncall =1000;
  double volume=GJP.Snodes()*GJP.SnodeSites();  for(int mu=0;mu<4;mu++) volume=volume*GJP.NodeSites(mu)*GJP.Nodes(mu);
  double flops=(GJP.Gparity() + 1)*1344*volume*ncall;
  
  Grid::RealD perf_F;
  Grid::RealD perf_FH;
  
  {
    LOGA2A << "Dirac operator type is : " << printType<GridDiracF>() << std::endl;
    FGrid_f->Barrier();
    //Ddwf_F.ZeroCounters(); no longer supporte
    Ddwf_F.Dhop(src,result,0);
    LOGA2A<<Grid::GridLogMessage<<"Called warmup"<<std::endl;
    double t0=Grid::usecond();
    for(int i=0;i<ncall;i++){
      Ddwf_F.Dhop(src,result,0);
    }
    double t1=Grid::usecond();
    FGrid_f->Barrier();

    LOGA2A<<Grid::GridLogMessage << "Called Dw "<<ncall<<" times in "<<t1-t0<<" us"<<std::endl;
    LOGA2A<<Grid::GridLogMessage << "mflop/s =   "<< flops/(t1-t0)<<std::endl;
    LOGA2A<<Grid::GridLogMessage << "mflop/s per rank =  "<< flops/(t1-t0)/NP<<std::endl;
    LOGA2A<<Grid::GridLogMessage << "mflop/s per node =  "<< flops/(t1-t0)/NN<<std::endl;
    //Ddwf_F.Report(); no longer supported
    perf_F = flops/(t1-t0)/NN; 
  }
  {
    LOGA2A << "Dirac operator type is : " << printType<GridDiracFH>() << std::endl;
    FGrid_f->Barrier();
    //Ddwf_FH.ZeroCounters();  no longer supported
    Ddwf_FH.Dhop(src,result,0);
    LOGA2A<<Grid::GridLogMessage<<"Called warmup"<<std::endl;
    double t0=Grid::usecond();
    for(int i=0;i<ncall;i++){
      Ddwf_FH.Dhop(src,result,0);
    }
    double t1=Grid::usecond();
    FGrid_f->Barrier();

    LOGA2A<<Grid::GridLogMessage << "Called Dw "<<ncall<<" times in "<<t1-t0<<" us"<<std::endl;
    LOGA2A<<Grid::GridLogMessage << "mflop/s =   "<< flops/(t1-t0)<<std::endl;
    LOGA2A<<Grid::GridLogMessage << "mflop/s per rank =  "<< flops/(t1-t0)/NP<<std::endl;
    LOGA2A<<Grid::GridLogMessage << "mflop/s per node =  "<< flops/(t1-t0)/NN<<std::endl;
    //Ddwf_FH.Report();  no longer supported
    perf_FH = flops/(t1-t0)/NN; 
  }
  return std::pair<double,double>(perf_F,perf_FH); //node performance in Mflops
}



#endif

CPS_END_NAMESPACE

#endif
