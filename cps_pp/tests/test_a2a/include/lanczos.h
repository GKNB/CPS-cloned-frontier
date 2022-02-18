#pragma once

CPS_START_NAMESPACE

#ifdef USE_GRID

template<typename GridA2Apolicies>
void testLanczosIO(typename GridA2Apolicies::FgridGFclass &lattice){
  LancArg lanc_arg;
  lanc_arg.mass = 0.01;
  lanc_arg.stop_rsd = 1e-08;
  lanc_arg.N_true_get = 50;
  GridLanczosWrapper<GridA2Apolicies> lanc;
  lanc.randomizeEvecs(lanc_arg,lattice);

  lanc.writeParallel("lanc");

  {
    GridLanczosWrapper<GridA2Apolicies> lanc2;
    lanc2.readParallel("lanc");

    assert(lanc2.evec_f.size() == 0);
    assert(lanc.evec.size() == lanc2.evec.size());
    assert(lanc.eval.size() == lanc2.eval.size());

    CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d_1;
    CPSfermion5Dcb4Dodd<cps::ComplexD> c_odd_d_2;
  
    for(int i=0;i<lanc.eval.size();i++){
      assert(lanc.eval[i] == lanc2.eval[i]);
      c_odd_d_1.importGridField(lanc.evec[i]);
      c_odd_d_2.importGridField(lanc2.evec[i]);
      
      assert( c_odd_d_1.equals( c_odd_d_2 ) );

      auto view = lanc.evec[i].View(Grid::CpuRead);
      auto view2 = lanc2.evec[i].View(Grid::CpuRead);

      for(int s=0;s<lanc.evec[i].Grid()->oSites();s++)
	assert( GridTensorEquals(view[s] , view2[s]) );
      
      
    }
  }

  lanc.toSingle();
  lanc.writeParallel("lanc");
  
  {
    GridLanczosWrapper<GridA2Apolicies> lanc2;
    lanc2.readParallel("lanc");

    assert(lanc2.evec.size() == 0);
    assert(lanc.evec_f.size() == lanc2.evec_f.size());
    assert(lanc.eval.size() == lanc2.eval.size());

    CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f_1;
    CPSfermion5Dcb4Dodd<cps::ComplexF> c_odd_f_2;
  
    for(int i=0;i<lanc.eval.size();i++){
      assert(lanc.eval[i] == lanc2.eval[i]);
      c_odd_f_1.importGridField(lanc.evec_f[i]);
      c_odd_f_2.importGridField(lanc2.evec_f[i]);
      
      assert( c_odd_f_1.equals( c_odd_f_2 ) );

      auto view = lanc.evec_f[i].View(Grid::CpuRead);
      auto view2 = lanc2.evec_f[i].View(Grid::CpuRead);

      for(int s=0;s<lanc.evec_f[i].Grid()->oSites();s++)
	assert( GridTensorEquals(view[s] , view2[s]) );
    }
  }
}

#endif //USE_GRID


CPS_END_NAMESPACE
