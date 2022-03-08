#pragma once

CPS_START_NAMESPACE

template<typename A2Apolicies>
void testMFmult(const A2AArg &a2a_args, const double tol){
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

  for(int i=0;i<nfull;i++){
    for(int k=0;k<nfull;k++){
      ScalarComplexType *oe = c_base.elem_ptr(i,k);
      if(oe == NULL) continue; //zero by definition

      *oe = 0.;
      for(int j=0;j<nfull;j++)
      	*oe += l.elem(i,j) * r.elem(j,k);
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
}


CPS_END_NAMESPACE
