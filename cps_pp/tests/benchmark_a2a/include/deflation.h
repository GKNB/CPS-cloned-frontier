#pragma once

CPS_START_NAMESPACE

#ifdef USE_GRID

//Use nl to control max number of eigenvectors in test
template<typename GridA2Apolicies>
void benchmarkDeflation(typename GridA2Apolicies::FgridGFclass &lattice, const int nl, int argc, char* argv[]){ 
  typedef typename GridA2Apolicies::GridFermionField GridFermionFieldD;
  typedef typename GridA2Apolicies::GridFermionFieldF GridFermionFieldF;
  
  Grid::GridRedBlackCartesian *FrbGridD = lattice.getFrbGrid();
  Grid::GridRedBlackCartesian *FrbGridF = lattice.getFrbGridF();
  
  std::vector<int> seeds5({5,6,7,8});
  Grid::GridParallelRNG RNGD(FrbGridD);  RNGD.SeedFixedIntegers(seeds5);
  Grid::GridParallelRNG RNGF(FrbGridF);  RNGF.SeedFixedIntegers(seeds5);
  Grid::GridSerialRNG SRNG; SRNG.SeedFixedIntegers(seeds5);

  bool do_dd = true, do_ds = true, do_sd=true, do_ss=true;
  std::vector<int> block_sizes;
  int i = 1;
  while(i<argc){
    std::string sarg = argv[i];
    if(sarg == "-deflation_block_sizes"){ //expect Grid-style argument: a.b.c.d  ->  [a,b,c,d]
      Grid::GridCmdOptionIntVector(argv[i+1], block_sizes);
      i+=2;
    }else if(sarg == "-deflation_disable_dd"){ //disable double prec evecs with double prec sources
      do_dd = false; i++;
    }else if(sarg == "-deflation_disable_ds"){ //disable double prec evecs with single prec sources
      do_ds = false; i++;
    }else if(sarg == "-deflation_disable_sd"){ //disable single prec evecs with double prec sources
      do_sd = false; i++;
    }else if(sarg == "-deflation_disable_ss"){ //disable single prec evecs with single prec sources
      do_ss = false; i++;     
    }else{
      i++;
    }
  }

  if(block_sizes.size() == 0){
    //Go up in powers of 2 until reach nl  
    int b = 2;
    while(b <= nl){
      block_sizes.push_back(b);
      b*=2;
    }
    if(block_sizes.size() == 0 || block_sizes.back() < nl) block_sizes.push_back(nl);
  }
 
  //Double precision benchmarks
  if(do_dd || do_ds){
    std::vector<GridFermionFieldD> evecs(nl, FrbGridD);
    std::vector<Grid::RealD> evals(nl);
    for(int i=0;i<nl;i++){
      random(RNGD, evecs[i]);
      random(SRNG, evals[i]);
    }
    EvecInterfaceMixedDoublePrec<GridFermionFieldD, GridFermionFieldF>  eveci(evecs, evals, FrbGridD, FrbGridF);

    if(do_dd){
      std::cout << "Deflating double precision fields in blocks with double precision eigenvectors" << std::endl;
      for(int b: block_sizes){
	std::vector<GridFermionFieldD> srcs(b, evecs[0]); //doesn't matter what the fields are
	std::vector<GridFermionFieldD> defl(b, FrbGridD);
	hostTouch(srcs); hostTouch(defl); hostTouch(evecs);
      
	double time = -dclock();
	eveci.deflatedGuessD(defl, srcs, -1);
	time += dclock();

	std::cout << b << " cold " << time << "s " << b / time << "/s" << std::endl;
      
	time = -dclock();
	eveci.deflatedGuessD(defl, srcs, -1);
	time += dclock();
      
	std::cout << b << " hot " << time << "s " << b / time << "/s" << std::endl;
      }
    }

    GridFermionFieldF src_f(FrbGridF);
    random(RNGF, src_f);

    if(do_ds){
      std::cout << "Deflating single precision fields in blocks with double precision eigenvectors" << std::endl;
      for(int b: block_sizes){
	std::vector<GridFermionFieldF> srcs(b, src_f); //doesn't matter what the fields are
	std::vector<GridFermionFieldF> defl(b, FrbGridF);
	hostTouch(srcs); hostTouch(defl); hostTouch(evecs);
      
	double time = -dclock();
	eveci.deflatedGuessF(defl, srcs, -1);
	time += dclock();

	std::cout << b << " cold " << time << "s " << b / time << "/s" << std::endl;

	time = -dclock();
	eveci.deflatedGuessF(defl, srcs, -1);
	time += dclock();

	std::cout << b << " hot " << time << "s " << b / time << "/s" << std::endl;
      }
    }
  }


  //Single precision benchmarks
  if(do_sd || do_ss){
    std::vector<GridFermionFieldF> evecs(nl, FrbGridF);
    std::vector<Grid::RealD> evals(nl);
    for(int i=0;i<nl;i++){
      random(RNGF, evecs[i]);
      random(SRNG, evals[i]);
    }
    EvecInterfaceSinglePrec<GridFermionFieldD, GridFermionFieldF>  eveci(evecs, evals, FrbGridD, FrbGridF);

    GridFermionFieldD src_d(FrbGridD);
    random(RNGD, src_d);

    if(do_sd){
      std::cout << "Deflating double precision fields in blocks with single precision eigenvectors" << std::endl;
      for(int b: block_sizes){
	std::vector<GridFermionFieldD> srcs(b, src_d);
	std::vector<GridFermionFieldD> defl(b, FrbGridD);
	hostTouch(srcs); hostTouch(defl); hostTouch(evecs);
      
	double time = -dclock();
	eveci.deflatedGuessD(defl, srcs, -1);
	time += dclock();
      
	std::cout << b << " cold " << time << "s " << b / time << "/s" << std::endl;

	time = -dclock();
	eveci.deflatedGuessD(defl, srcs, -1);
	time += dclock();
      
	std::cout << b << " hot " << time << "s " << b / time << "/s" << std::endl;
      }
    }

    if(do_ss){
      std::cout << "Deflating single precision fields in blocks with single precision eigenvectors" << std::endl;
      for(int b: block_sizes){
	std::vector<GridFermionFieldF> srcs(b, evecs[0]);      
	std::vector<GridFermionFieldF> defl(b, FrbGridF);
	hostTouch(srcs); hostTouch(defl); hostTouch(evecs);
      
	double time = -dclock();
	eveci.deflatedGuessF(defl, srcs, -1);
	time += dclock();
	std::cout << b << " cold " << time << "s " << b / time << "/s" << std::endl;

	time = -dclock();
	eveci.deflatedGuessF(defl, srcs, -1);
	time += dclock();
	std::cout << b << " hot " << time << "s " << b / time << "/s" << std::endl;
      }
    }
  }



}



#endif //USE_GRID



CPS_END_NAMESPACE
