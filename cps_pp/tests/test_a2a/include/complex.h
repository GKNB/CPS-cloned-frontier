#pragma once

CPS_START_NAMESPACE

void testConvertComplexD(){
  std::cout << "Starting testConvertComplexD" << std::endl;
  std::complex<double> std(3.14, 2.22);
  Grid::ComplexD grid(3.14,2.22);

  std::complex<double> grid_conv = convertComplexD(grid);

  std::cout << "Std (" << std.real() << "," << std.imag() << ")  Grid (" << grid_conv.real() << "," << grid_conv.imag() << ")" << std::endl;
  
  assert( fabs( grid_conv.real() - std.real() ) < 1e-12 && fabs( grid_conv.imag() - std.imag() ) < 1e-12 );
  
  std::cout << "testConvertComplexD passed" << std::endl;
}

#ifdef USE_GRID

template<typename GridPolicies>
void testBasicComplexArray(){
  std::cout << "Starting testBasicComplexArray" << std::endl;
  typedef typename GridPolicies::ComplexType ComplexType;
  typedef Aligned128AllocPolicy AllocPolicy;

  int nodes = GJP.TotalNodes();
  
  constexpr int Nsimd = ComplexType::Nsimd();
  std::cout << "Nsimd=" << Nsimd << std::endl;

  typedef typename ComplexType::scalar_type ScalarComplexType;

  ScalarComplexType one_s(1.,0.);  
  ComplexType one(one_s);
  for(int i=0;i<Nsimd;i++)
    assert( one.getlane(i) == one_s );

  //Check thread sum
  if(1){
    int nthr = 2;
    basicComplexArraySplitAlloc<ComplexType,AllocPolicy> m(1,nthr);
    assert(m.nThreads() == nthr);
    for(int t=0;t<nthr;t++)
      m(0,t) = one;
    
    ScalarComplexType expect_s(nthr, 0.);
    ComplexType expect(expect_s);

    std::cout << "threadSum over " << nthr << " threads" << std::endl;
    
    m.threadSum();
    std::cout << "Lane 0 expect " << expect.getlane(0) << " got " << m[0].getlane(0) << std::endl;
    
    assert(m.nThreads() == 1);

    assert( vTypeEquals(m[0],expect,1e-12,true) );
    std::cout << "Passed testBasicComplexArray thread sum check" << std::endl;
  }

  //Check node sum
  if(1){
    basicComplexArraySplitAlloc<ComplexType,AllocPolicy> m(1,1);
    m[0] = one;
    
    ScalarComplexType expect_s(nodes, 0.);
    ComplexType expect(expect_s);

    std::cout << "nodeSum over " << nodes << " nodes" << std::endl;
    m.nodeSum();
    std::cout << "Lane 0 expect " << expect.getlane(0) << " got " << m[0].getlane(0) << std::endl;
    
    assert( vTypeEquals(m[0],expect,1e-12,true) );
    std::cout << "Passed testBasicComplexArray node sum check" << std::endl;
  }

  //Reproduce perlmutter crash
  {
    int size = 512; //1024 crash //1 pass // 3072 crash;
    basicComplexArraySplitAlloc<ComplexType,AllocPolicy> m(size,1);
    for(int i=0;i<size;i++) m[i] = one;

    //std::cout << "threadSum over 1 threads" << std::endl;
    //m.threadSum();
    
    ScalarComplexType expect_s(nodes, 0.);
    ComplexType expect(expect_s);

    std::cout << "nodeSum over " << nodes << " nodes" << std::endl;
    m.nodeSum();
    /* ComplexType *ptr = &m[0]; */
    /* double* dptr = (double*)ptr; */
    /* size_t dsize = size * 2 * Nsimd; */
    /* MPI_Allreduce(MPI_IN_PLACE, dptr, dsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); */
    std::cout << "Lane 0 expect " << expect.getlane(0) << " got " << m[0].getlane(0) << std::endl;

    for(int i=0;i<size;i++){
      assert( vTypeEquals(m[i],expect,1e-12,true) );
    }
    std::cout << "Passed testBasicComplexArray node sum check 2" << std::endl;
  }


  
  std::cout << "Passed testBasicComplexArray all checks" << std::endl;
}

#endif
  
CPS_END_NAMESPACE
