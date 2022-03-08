#pragma once

CPS_START_NAMESPACE

#ifdef USE_GRID
#ifdef GRID_SYCL

bool equals(const fMatrix<cps::ComplexD> &l, const oneMKLmatrix<cps::ComplexD> &r, double tol = 1e-10){
  if(l.nRows() != r.rows()) return false;
  if(l.nCols() != r.cols()) return false;
  for(int i=0;i<l.nRows();i++){
    for(int j=0;j<l.nCols();j++){
      const cps::ComplexD &aa = l(i,j);
      const cps::ComplexD &bb = r(i,j);	
      if(fabs(aa.real() - bb.real()) > tol)
	return false;
      if(fabs(aa.imag() - bb.imag()) > tol)
	return false;
    }
  }
  return true;
}

void testOneMKLwrapper(){
  std::cout << "Testing oneMKLwrapper" << std::endl; 
   
  std::default_random_engine gen(1234);
  std::normal_distribution<double> dist(5.0,2.0);
  
  fMatrix<cps::ComplexD> A(100,50);
  fMatrix<cps::ComplexD> B(50,100);

  oneMKLmatrix<cps::ComplexD> A2(100,50);
  oneMKLmatrix<cps::ComplexD> B2(50,100);
  
  for(int i=0;i<100;i++){
    for(int j=0;j<50;j++){
      A(i,j) = cps::ComplexD(dist(gen), dist(gen));
      B(j,i) = cps::ComplexD(dist(gen), dist(gen));

      A2(i,j) = A(i,j);
      B2(j,i) = B(j,i);
    }
  }
  assert(!A.equals(B));

  std::cout << "Test ZGEMM" << std::endl;
  fMatrix<cps::ComplexD> C = A*B;

  oneMKLmatrix<cps::ComplexD> C2(100,100);
  mult_offload_oneMKL(C2, A2, B2);

  assert(equals(C,C2,1e-10));

  std::cout << "oneMKLwrapper tests passed" << std::endl;
}

#endif
#endif




CPS_END_NAMESPACE
