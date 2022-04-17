#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void benchmarkMFmult(const A2AArg &a2a_args, const int ntests){
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_WV; 
  mf_WV l;
  l.setup(a2a_args,a2a_args,0,0);
  l.testRandom();  

  int nodes = 1; for(int i=0;i<5;i++) nodes *= GJP.Nodes(i);

  if(!UniqueID()) printf("mf_WV sizes %d %d. Using %d threads\n",l.getNrows(),l.getNcols(), omp_get_max_threads());

  mf_WV r;
  r.setup(a2a_args,a2a_args,1,1);
  r.testRandom();  

  const size_t ni = l.getNrows();
  const size_t nk = r.getNcols();

  typedef typename mf_WV::RightDilutionType ConLeftDilutionType;
  typedef typename mf_WV::LeftDilutionType ConRightDilutionType;

  ModeContractionIndices<ConLeftDilutionType,ConRightDilutionType> ind(l.getColParams());
    
  modeIndexSet lmodeparams; lmodeparams.time = l.getColTimeslice();
  modeIndexSet rmodeparams; rmodeparams.time = r.getRowTimeslice();
    
  const size_t nj = ind.getNindices(lmodeparams,rmodeparams);

  //zmul 6 Flops
  //zmadd 8 Flops
  //zvecdot (N) = 6 + (N-1)*8 Flops

  size_t Flops = ni * nk * ( 6 + (nj-1)*8 );
  double time, Mflops, Mflops_per_node;

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c;

  //First call has setup overheads, separate out
  time = -dclock();    
  mult(c, l, r, true); //NODE LOCAL, used in pipi
  time += dclock();

  Mflops = double(Flops)/time/double(1.e6);
  
  if(!UniqueID()) printf("MF mult node local first call (ni=%d nj=%d nk=%d) avg time %f s, %f Mflops\n",ni,nj,nk,time,Mflops);
  
#ifdef MULT_IMPL_CUBLASXT
  std::cout << "cuBLASXT handle setup time (once) " << cuBLAShandles::time() << "s" << std::endl;
  if(!UniqueID()) _mult_impl_base::getTimers().print();
  _mult_impl_base::getTimers().reset();
#endif

  
  time = -dclock();
  for(int i=0;i<ntests;i++){
    mult(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  time /= double(ntests);

  Mflops = double(Flops)/time/double(1.e6);

  if(!UniqueID()) printf("MF mult node local (ni=%d nj=%d nk=%d) calls %d, avg time %f s, %f Mflops\n",ni,nj,nk,ntests,time,Mflops);

#ifdef MULT_IMPL_CUBLASXT
  if(!UniqueID()) _mult_impl_base::getTimers().print();
  _mult_impl_base::getTimers().reset();
#endif

  time = -dclock();
  for(int i=0;i<ntests;i++){
    mult(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  time /= double(ntests); 
  
  Mflops = double(Flops)/time/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult node distributed (ni=%d nj=%d nk=%d) calls %d, avg time %f s, %f Mflops,  %f Mflops/node\n",ni,nj,nk,ntests,time,Mflops, Mflops_per_node);

#ifdef MULT_IMPL_CUBLASXT
  if(!UniqueID()) _mult_impl_base::getTimers().print();
#endif


  //////////////////////////////////////////
#ifdef MULT_IMPL_GSL

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);

  if(!UniqueID()) printf("MF mult_orig node local (ni=%d nj=%d nk=%d) %f Mflops\n",ni,nj,nk,Mflops);

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult_orig node distributed (ni=%d nj=%d nk=%d) %f Mflops,  %f Mflops/node\n",ni,nj,nk,Mflops, Mflops_per_node);


  ////////////////////////////////////////////
  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);

  if(!UniqueID()) printf("MF mult_opt1 node local (ni=%d nj=%d nk=%d) %f Mflops\n",ni,nj,nk,Mflops);

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult_opt1 node distributed (ni=%d nj=%d nk=%d) %f Mflops,  %f Mflops/node\n",ni,nj,nk,Mflops, Mflops_per_node);


  ////////////////////////////////////////////
  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c, l, r, true); //NODE LOCAL, used in pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);

  if(!UniqueID()) printf("MF mult_opt2 node local (ni=%d nj=%d nk=%d) %f Mflops\n",ni,nj,nk,Mflops);

  time = -dclock();
  for(int i=0;i<ntests;i++){
    _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c, l, r, false); //NODE DISTRIBUTED, used in K->pipi
  }
  time += dclock();

  Mflops = double(Flops)/time*double(ntests)/double(1.e6);
  Mflops_per_node = Mflops/nodes;
  
  if(!UniqueID()) printf("MF mult_opt2 node distributed (ni=%d nj=%d nk=%d) %f Mflops,  %f Mflops/node\n",ni,nj,nk,Mflops, Mflops_per_node);
#endif //MULT_IMPL_GSL
}

CPS_END_NAMESPACE
