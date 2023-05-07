#pragma once

CPS_START_NAMESPACE

template<typename GridA2Apolicies>
void benchmarkVVgridOffload(const A2AArg &a2a_args, const int ntests, const int nthreads){
  mult_vv_field_offload_timers::get().reset();
  std::cout << "Starting vv benchmark with policies " << printType<GridA2Apolicies>() << std::endl;

  const int nsimd = GridA2Apolicies::ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  A2AvectorWfftw<GridA2Apolicies> Wgrid(a2a_args, simd_dims);
  A2AvectorVfftw<GridA2Apolicies> Vgrid(a2a_args, simd_dims);

  Wgrid.testRandom();
  Vgrid.testRandom();
  

  typedef typename getPropagatorFieldType<GridA2Apolicies>::type PropagatorField;
  typedef mult_vv_field<GridA2Apolicies, A2AvectorVfftw, A2AvectorWfftw, PropagatorField> offload;
  PropagatorField pfield(simd_dims);
      
  Float total_time_field_offload = 0;

  for(int iter=0;iter<ntests;iter++){
    total_time_field_offload -= dclock();    
    mult(pfield, Vgrid, Wgrid, false, true);
    total_time_field_offload += dclock();
  }


  int nf = GJP.Gparity() + 1;

  //Count flops (over all nodes)
  //\sum_i v(il)_{scl,fl}(x) * v(ir)_{scr,fr}(x) for all t, x3d
  ModeContractionIndices<typename offload::leftDilutionType, typename offload::rightDilutionType> i_ind(Vgrid);
  size_t Flops = 0;
  for(int t_glob=0;t_glob<GJP.TnodeSites()*GJP.Tnodes();t_glob++){
    modeIndexSet ilp, irp, jlp, jrp;
    ilp.time = irp.time = t_glob;
    
    for(ilp.flavor=0;ilp.flavor<nf;ilp.flavor++){
      for(ilp.spin_color=0;ilp.spin_color<12;ilp.spin_color++){
	for(irp.flavor=0;irp.flavor<nf;irp.flavor++){
	  for(irp.spin_color=0;irp.spin_color<12;irp.spin_color++){
	    size_t ni = i_ind.getIndexVector(ilp,irp).size();
	    Flops +=  ni * 8; //z = z + (z*z)   z*z=6flops
	  }
	}
      }
    }
  }
  Flops *= GJP.TotalNodes()*GJP.VolNodeSites()/GJP.TnodeSites(); //the above is done for every global 3d site

  double tavg = total_time_field_offload/ntests;
  double Mflops = double(Flops)/tavg/1e6;

  printf("vv: Avg time field offload code %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);

  if(!UniqueID()){
    printf("vv offload timings:\n");
    mult_vv_field_offload_timers::get().print();
  }

}



CPS_END_NAMESPACE
