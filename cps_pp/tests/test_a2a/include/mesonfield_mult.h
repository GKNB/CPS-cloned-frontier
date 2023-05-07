#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void testMFmult(const A2AArg &a2a_args, const double tol){
  std::cout << "Starting testMFmult" << std::endl;  
  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_WV; 
  typedef typename mf_WV::ScalarComplexType ScalarComplexType;

  mf_WV l;
  l.setup(a2a_args,a2a_args,0,0);
  l.testRandom();  

  if(!UniqueID()) printf("mf_WV packed sizes %d %d and nl = %d\n",l.getNrows(),l.getNcols(),a2a_args.nl);

  mf_WV r;
  r.setup(a2a_args,a2a_args,1,1);
  r.testRandom();  

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c_base;
  c_base.setup(a2a_args,a2a_args,0,1);

  A2Aparams a2a_params(a2a_args);
  int nfull = a2a_params.getNv();
  if(!UniqueID()){ printf("Total modes %d\n", nfull); fflush(stdout); }

  {
    CPSautoView(c_base_v,c_base,HostWrite);
    CPSautoView(l_v,l,HostRead);
    CPSautoView(r_v,r,HostRead);
    for(int i=0;i<nfull;i++){
      for(int k=0;k<nfull;k++){
	ScalarComplexType *oe = c_base_v.elem_ptr(i,k);
	if(oe == NULL) continue; //zero by definition
	
	*oe = 0.;
	for(int j=0;j<nfull;j++)
	  *oe += l_v.elem(i,j) * r_v.elem(j,k);
      }
    }
  }

  std::cout << "norm2(l)=" << l.norm2() << " norm2(r)=" << r.norm2() << " norm2(c_base)=" << c_base.norm2() << std::endl; 

  {
    //Another host implementation
    int ni = l.getNrowsFull(), nj = l.getNcolsFull(), nk = r.getNcolsFull(); assert(nj == r.getNrowsFull());
    
    ScalarComplexType *lup = (ScalarComplexType *)memalign_check(128,ni*nj*sizeof(ScalarComplexType));
    ScalarComplexType *rup = (ScalarComplexType *)memalign_check(128,nj*nk*sizeof(ScalarComplexType));
    ScalarComplexType *lrup = (ScalarComplexType *)memalign_check(128,ni*nk*sizeof(ScalarComplexType));
    memset(lrup,0,ni*nk*sizeof(ScalarComplexType));
    l.unpack(lup);

    {
      std::cout << "Sanity check host read views compare to unpack" << std::endl;
      typedef mesonFieldConvertDilution<typename mf_WV::LeftDilutionType> ConvRow;
      typedef mesonFieldConvertDilution<typename mf_WV::RightDilutionType> ConvCol;
      int nl = a2a_params.getNl();
      int nf = a2a_params.getNflavors();
      int nsc = a2a_params.getNspinColor();
      int ntblock = a2a_params.getNtBlocks();
      int tblockrow = a2a_params.tblock(l.getRowTimeslice());
      int tblockcol = a2a_params.tblock(l.getColTimeslice());      
      
      CPSautoView(l_v,l,HostRead);
      for(int i=0;i<nfull;i++){
	auto pi = ConvRow::pack(i,tblockrow,nl,nf,nsc,ntblock);	
	for(int j=0;j<nfull;j++){
	  auto pj = ConvCol::pack(j,tblockcol,nl,nf,nsc,ntblock);	
	  
	  if(pi.second && pj.second){
	    std::cout << i << " " << j << " " << lup[j+nj*i] << " " << l_v.elem(i,j) << " " << l_v(pi.first,pj.first) << std::endl;
	  }else{
	    std::cout << i << " " << j << " " << lup[j+nj*i] << " " << l_v.elem(i,j) << " (!)" << ScalarComplexType(0.0) << std::endl;
	  }	  
	  assert( lup[j + nj*i] == l_v.elem(i,j) );
	}
      }
    }
    
    r.unpack(rup);
    
    for(int i=0;i<ni;i++){
      for(int k=0;k<nk;k++){
	for(int j=0;j<nj;j++){
	  lrup[k+nk*i] += lup[j+nj*i]*rup[k+nk*j];
	}
      }
    }
    A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c_base_2;
    c_base_2.setup(a2a_args,a2a_args,0,1);
    c_base_2.pack(lrup);
    std::cout << "Second host impl norm2(c_base_2)=" << c_base_2.norm2() << std::endl;
    assert(c_base_2.equals(c_base, tol, true));
  }
  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c;
  mult(c, l, r, true); //node local

  std::cout << "norm2(c)=" << c.norm2() << std::endl; 

  bool test = c.equals(c_base, tol, true);
  std::cout << "Node local mult result: " << test << std::endl;
  if(!test) ERR.General("","testMFmult","Node local mult failed!\n");
  
  mult(c, l, r, false); //node distributed

  test = c.equals(c_base, tol, true);

  std::cout << "norm2(c)=" << c.norm2() << std::endl; 
  
  std::cout << "Node distributed mult result: " << test << std::endl;
  if(!test) ERR.General("","testMFmult","Node distributed mult failed!\n");

  if(!UniqueID()) printf("Passed MF mult tests\n");


#ifdef MULT_IMPL_GSL
  //Test other GSL implementations
  
  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, true); //node local

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_orig failed!\n");
  
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, false); //node distributed

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_orig failed!\n");

  if(!UniqueID()) printf("Passed MF mult_orig tests\n");

  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c,l,r, true);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_opt1 failed!\n");

  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c,l,r, false);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_opt1 failed!\n");

  if(!UniqueID()) printf("Passed MF mult_opt1 tests\n");

  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c,l,r, true);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_opt2 failed!\n");

  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c,l,r, false);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_opt2 failed!\n");

  if(!UniqueID()) printf("Passed MF mult_opt2 tests\n");
#endif //MULT_IMPL_GSL
  std::cout << "testMFmult passed" << std::endl;
}



template<typename A2Apolicies>
void testMFmultTblock(A2AArg a2a_args, const double tol){
  std::cout << "Starting testMFmultTblock" << std::endl;
  a2a_args.src_width = 2;

  typedef A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> mf_WV; 
  typedef typename mf_WV::ScalarComplexType ScalarComplexType;

  mf_WV l;
  l.setup(a2a_args,a2a_args,0,0);
  l.testRandom();  

  if(!UniqueID()) printf("mf_WV sizes %d %d\n",l.getNrows(),l.getNcols());

  mf_WV r;
  r.setup(a2a_args,a2a_args,1,1);
  r.testRandom();  

  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c_base;
  c_base.setup(a2a_args,a2a_args,0,1);

  A2Aparams a2a_params(a2a_args);
  int nfull = a2a_params.getNv();
  if(!UniqueID()){ printf("Total modes %d\n", nfull); fflush(stdout); }

  {
    CPSautoView(c_base_v,c_base,HostWrite);
    CPSautoView(l_v,l,HostRead);
    CPSautoView(r_v,r,HostRead);
    for(int i=0;i<nfull;i++){
      for(int k=0;k<nfull;k++){
	ScalarComplexType *oe = c_base_v.elem_ptr(i,k);
	if(oe == NULL) continue; //zero by definition
	
	*oe = 0.;
	for(int j=0;j<nfull;j++)
	  *oe += l_v.elem(i,j) * r_v.elem(j,k);
      }
    }
  }

  std::cout << "norm2(l)=" << l.norm2() << " norm2(r)=" << r.norm2() << " norm2(c_base)=" << c_base.norm2() << std::endl; 
  
  A2AmesonField<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw> c;
  mult(c, l, r, true); //node local

  std::cout << "norm2(c)=" << c.norm2() << std::endl; 

  bool test = c.equals(c_base, tol, true);
  std::cout << "Node local mult result: " << test << std::endl;
  if(!test) ERR.General("","testMFmult","Node local mult failed!\n");
  
  mult(c, l, r, false); //node distributed

  test = c.equals(c_base, tol, true);

  std::cout << "norm2(c)=" << c.norm2() << std::endl; 
  
  std::cout << "Node distributed mult result: " << test << std::endl;
  if(!test) ERR.General("","testMFmult","Node distributed mult failed!\n");

  if(!UniqueID()) printf("Passed MF mult tests\n");


#ifdef MULT_IMPL_GSL
  //Test other GSL implementations
  
  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, true); //node local

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_orig failed!\n");
  
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_orig(c, l, r, false); //node distributed

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_orig failed!\n");

  if(!UniqueID()) printf("Passed MF mult_orig tests\n");

  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c,l,r, true);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_opt1 failed!\n");

  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt1(c,l,r, false);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_opt1 failed!\n");

  if(!UniqueID()) printf("Passed MF mult_opt1 tests\n");

  /////////////////////////////////////
  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c,l,r, true);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node local mult_opt2 failed!\n");

  _mult_impl<A2Apolicies,A2AvectorWfftw,A2AvectorVfftw,A2AvectorWfftw,A2AvectorVfftw>::mult_opt2(c,l,r, false);

  if(!c.equals(c_base, tol, true)) ERR.General("","testMFmult","Node distributed mult_opt2 failed!\n");

  if(!UniqueID()) printf("Passed MF mult_opt2 tests\n");
#endif //MULT_IMPL_GSL

  std::cout << "testMFmultTblock passed" << std::endl;
}


CPS_END_NAMESPACE
