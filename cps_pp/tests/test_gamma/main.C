//CK 2021:  Thorough test of gamma matrix conventions and consistency between WilsonMatrix, SpinMatrix, CPSspinMatrix

#include <config.h>
#include <alg/wilson_matrix.h>
#include <iostream>
#include <cassert>
#include <util/gjp.h>
#include <util/command_line.h>
#include <alg/a2a/lattice/spin_color_matrices.h>

#ifdef USE_GRID
#include<Grid.h>
#endif

using namespace std;
USING_NAMESPACE_CPS

void unit_matrix(WilsonMatrix &m){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int a=0;a<3;a++){
  	for(int b=0;b<3;b++){
	  if(i==j && a == b) m(i,a,j,b) = Complex(1.0,0.0);
  	  else m(i,a,j,b) = Complex(0.0,0.0);
  	}
      }
    }
  }
}

void zero(WilsonMatrix &m){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int a=0;a<3;a++){
  	for(int b=0;b<3;b++){
  	  m(i,a,j,b) = Complex(0.0,0.0);
  	}
      }
    }
  }
}


void print_spin(const WilsonMatrix &m){
  printf("\n");
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      double re = m(i,0,j,0).real();
      double im = m(i,0,j,0).imag();
      if(re == 0.0 && im == 1.0) printf("i ");
      else if(re == 0.0 && im == -1.0) printf("-i ");
      else if(re == 1.0 && im == 0.0) printf("1 ");
      else if(re == -1.0 && im == 0.0) printf("-1 ");
      else if(re == 0.0 && im == 0.0) printf("0 ");
      else if(re == 0.0 && im == 2.0) printf("2i ");
      else if(re == 0.0 && im == -2.0) printf("-2i ");
      else if(re == 2.0 && im == 0.0) printf("2 ");
      else if(re == -2.0 && im == 0.0) printf("-2 ");
      else { printf("print can only do +-1, +-2, or +-i, +-2i, got (%f,%f)\n",re,im); exit(-1); }
    }
    printf("\n");
  }
  printf("\n");
}

bool compare(const WilsonMatrix &A, const WilsonMatrix &B, const double tol = 1e-12){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int a=0;a<3;a++){
  	for(int b=0;b<3;b++){
	  if( fabs( A(i,a,j,b).real() - B(i,a,j,b).real() ) > tol ){
	    std::cout << "Fail " << i << " " << j << " " << a << " " << b << " re " << A(i,a,j,b).real() << " " <<  B(i,a,j,b).real() << std::endl;
	    return false;
	  }
	  else if( fabs( A(i,a,j,b).imag() - B(i,a,j,b).imag() ) > tol ){
	    std::cout << "Fail " << i << " " << j << " " << a << " " << b << " im " << A(i,a,j,b).imag() << " " <<  B(i,a,j,b).imag() << std::endl;
	    return false;
	  }
	}
      }
    }
  }
  return true;
}

//These are the conventions we expect
//  Chiral basis
//  gamma(XUP)    gamma(YUP)    gamma(ZUP)    gamma(TUP)    gamma(FIVE)
//  0  0  0  i    0  0  0 -1    0  0  i  0    0  0  1  0    1  0  0  0
//  0  0  i  0    0  0  1  0    0  0  0 -i    0  0  0  1    0  1  0  0
//  0 -i  0  0    0  1  0  0   -i  0  0  0    1  0  0  0    0  0 -1  0
// -i  0  0  0   -1  0  0  0    0  i  0  0    0  1  0  0    0  0  0 -1

void expect_gamma(WilsonMatrix &g, int mu){
  zero(g);
  Complex _i(0,1);
  Complex _mi(0,-1);
  Complex _1(1);
  Complex _m1(-1);  
  
  if(mu == 0){ //X
    for(int a=0;a<3;a++){
      g(0,a,3,a) = _i;
      g(1,a,2,a) = _i;
      g(2,a,1,a) = _mi;
      g(3,a,0,a) = _mi;
    }
  }
  else if(mu == 1){ //Y
    for(int a=0;a<3;a++){
      g(0,a,3,a) = _m1;
      g(1,a,2,a) = _1;
      g(2,a,1,a) = _1;
      g(3,a,0,a) = _m1;
    }
  }
  else if(mu == 2){ //Z
    for(int a=0;a<3;a++){
      g(0,a,2,a) = _i;
      g(1,a,3,a) = _mi;
      g(2,a,0,a) = _mi;
      g(3,a,1,a) = _i;
    }
  }
  else if(mu == 3){ //T
    for(int a=0;a<3;a++){
      g(0,a,2,a) = _1;
      g(1,a,3,a) = _1;
      g(2,a,0,a) = _1;
      g(3,a,1,a) = _1;
    }
  }
  else if(mu == 5){ //g5
    for(int a=0;a<3;a++){
      g(0,a,0,a) = _1;
      g(1,a,1,a) = _1;
      g(2,a,2,a) = _m1;
      g(3,a,3,a) = _m1;
    }
  }
  else{
    assert(0);
  }
}

bool test_eq(const WilsonMatrix &expect, const WilsonMatrix &got){    
  if(compare(expect, got)){
    std::cout << " pass" << std::endl;
    return true;
  }
  else{  
    std::cout << " FAIL" << std::endl; 
    std::cout << "Got ";
    print_spin(got);
    std::cout << "\nExpect ";
    print_spin(expect);
    std::cout << std::endl;
    return false; 
  }
}


bool compare(const WilsonMatrix &A, const SpinMatrix &B, const double tol = 1e-12){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int a=0;a<3;a++){
  	for(int b=0;b<3;b++){
	  if(a==b){
	    if( fabs( A(i,a,j,b).real() - B(i,j).real() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " re " << A(i,a,j,b).real() << " " <<  B(i,j).real() << std::endl;
	      return false;
	    }
	    else if( fabs( A(i,a,j,b).imag() - B(i,j).imag() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " im " << A(i,a,j,b).imag() << " " <<  B(i,j).imag() << std::endl;
	      return false;
	    }
	  }else{
	    if( fabs( A(i,a,j,b).real() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " re " << A(i,a,j,b).real() << " 0"<< std::endl;
	      return false;
	    }
	    else if( fabs( A(i,a,j,b).imag() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " im " << A(i,a,j,b).imag() << " 0"<< std::endl;
	      return false;
	    }
	  }
	}
      }
    }
  }
  return true;
}

bool test_eq(const WilsonMatrix &expect, const SpinMatrix &got){    
  if(compare(expect, got)){
    std::cout << " pass" << std::endl;
    return true;
  }
  else{  
    std::cout << " FAIL" << std::endl; 
    return false; 
  }
}



bool compare(const WilsonMatrix &A, const CPSspinMatrix<Complex> &B, const double tol = 1e-12){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int a=0;a<3;a++){
  	for(int b=0;b<3;b++){
	  if(a==b){
	    if( fabs( A(i,a,j,b).real() - B(i,j).real() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " re " << A(i,a,j,b).real() << " " <<  B(i,j).real() << std::endl;
	      return false;
	    }
	    else if( fabs( A(i,a,j,b).imag() - B(i,j).imag() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " im " << A(i,a,j,b).imag() << " " <<  B(i,j).imag() << std::endl;
	      return false;
	    }
	  }else{
	    if( fabs( A(i,a,j,b).real() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " re " << A(i,a,j,b).real() << " 0"<< std::endl;
	      return false;
	    }
	    else if( fabs( A(i,a,j,b).imag() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " im " << A(i,a,j,b).imag() << " 0"<< std::endl;
	      return false;
	    }
	  }
	}
      }
    }
  }
  return true;
}

bool test_eq(const WilsonMatrix &expect, const CPSspinMatrix<Complex> &got){    
  if(compare(expect, got)){
    std::cout << " pass" << std::endl;
    return true;
  }
  else{  
    std::cout << " FAIL" << std::endl; 
    return false; 
  }
}


#ifdef USE_GRID

bool compare(const WilsonMatrix &A, const Grid::iMatrix<Grid::ComplexD,Grid::Ns> &B, const double tol = 1e-12){
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int a=0;a<3;a++){
  	for(int b=0;b<3;b++){
	  if(a==b){
	    if( fabs( A(i,a,j,b).real() - B(i,j).real() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " re " << A(i,a,j,b).real() << " " <<  B(i,j).real() << std::endl;
	      return false;
	    }
	    else if( fabs( A(i,a,j,b).imag() - B(i,j).imag() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " im " << A(i,a,j,b).imag() << " " <<  B(i,j).imag() << std::endl;
	      return false;
	    }
	  }else{
	    if( fabs( A(i,a,j,b).real() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " re " << A(i,a,j,b).real() << " 0"<< std::endl;
	      return false;
	    }
	    else if( fabs( A(i,a,j,b).imag() ) > tol ){
	      std::cout << "Fail " << i << " " << j << " " << a << " " << b << " im " << A(i,a,j,b).imag() << " 0"<< std::endl;
	      return false;
	    }
	  }
	}
      }
    }
  }
  return true;
}

bool test_eq(const WilsonMatrix &expect, const Grid::iMatrix<Grid::ComplexD,Grid::Ns> &got){    
  if(compare(expect, got)){
    std::cout << " pass" << std::endl;
    return true;
  }
  else{  
    std::cout << " FAIL" << std::endl; 
    return false; 
  }
}

void unit_matrix(Grid::iMatrix<Grid::ComplexD,Grid::Ns> &m){
  for(int i=0;i<Grid::Ns;i++){
    for(int j=0;j<Grid::Ns;j++){
      m(i,j) = i==j ? Grid::ComplexD(1.0) : Grid::ComplexD(0.);
    }
  }
}


#endif




int main(int argc,char *argv[])
{
  Start(&argc,&argv);
  CommandLine::is(argc,argv);
  
  WilsonMatrix unit_wm;
  unit_matrix(unit_wm);

  WilsonMatrix zero_wm;
  zero(zero_wm);

  for(int mu=0;mu<4;mu++){
    WilsonMatrix expect;
    expect_gamma(expect, mu);
    
    WilsonMatrix got = unit_wm;
    got.gl(mu);
    std::cout << "gl(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;

    got = unit_wm;
    got.gr(mu);
    std::cout << "gr(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;
  }
  
  //Test Clifford algebra (Euclidean)
  for(int mu=0;mu<4;mu++){
    for(int nu=0;nu<4;nu++){
      WilsonMatrix gmu;
      expect_gamma(gmu, mu);

      WilsonMatrix gnu;
      expect_gamma(gnu, nu);
    
      WilsonMatrix got = gmu*gnu + gnu*gmu;
      
      WilsonMatrix expect = mu == nu ? unit_wm*2. : zero_wm;

      std::cout << "{g" << mu+1 << ", g" << nu+1 << "} : ";
      if(!test_eq(expect,got)) return 1;
    }
  }


  //Test g5
  {
    WilsonMatrix expect;
    expect_gamma(expect, 5);
  
    WilsonMatrix got = unit_wm;
    got.gl(-5);
    std::cout << "gl(" << -5 << "): ";
    if(!test_eq(expect,got)) return 1;

    got = unit_wm;
    got.gr(-5);
    std::cout << "gr(" << -5 << "): ";
    if(!test_eq(expect,got)) return 1;
  }
  
  //Test g5 is the product of g1g2g3g4
  {
    WilsonMatrix got = unit_wm;
    got.gr(0);
    got.gr(1);
    got.gr(2);
    got.gr(3);

    WilsonMatrix expect;
    expect_gamma(expect, 5);
    
    std::cout << "g5 = g1g2g3g4 : ";
    if(!test_eq(expect,got)) return 1;
  }

  //Test glA, glV, glL, glR and r equivs

  //g(l/r)A   =   gmu g5
  for(int mu=0;mu<4;mu++){
    WilsonMatrix gmu;
    expect_gamma(gmu, mu);

    WilsonMatrix g5;
    expect_gamma(g5, 5);

    WilsonMatrix expect = gmu * g5;
    
    WilsonMatrix got;
    got.glA(unit_wm, mu);
    std::cout << "glA(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;

    got.grA(unit_wm, mu);
    std::cout << "grA(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;
  }

  //g(l/r)V   =   gmu
  for(int mu=0;mu<4;mu++){
    WilsonMatrix gmu;
    expect_gamma(gmu, mu);

    WilsonMatrix expect = gmu;
    
    WilsonMatrix got;
    got.glV(unit_wm, mu);
    std::cout << "glV(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;

    got.grV(unit_wm, mu);
    std::cout << "grV(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;
  }

  //gl(L/R)   =   gmu ( 1-+ g5 )
  for(int mu=0;mu<4;mu++){
    WilsonMatrix gmu;
    expect_gamma(gmu, mu);

    WilsonMatrix g5;
    expect_gamma(g5, 5);

    WilsonMatrix expect = gmu * ( unit_wm - g5 );
    
    WilsonMatrix got = unit_wm;
    got = got.glL(mu);
    std::cout << "glL(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;

    expect = gmu * ( unit_wm + g5 );
    
    got = unit_wm;
    got = got.glR(mu);
    std::cout << "glR(" << mu << "): ";
    if(!test_eq(expect,got)) return 1;
  }

  //left and right multiply by C
  {
    WilsonMatrix g2, g4;
    expect_gamma(g2, 1);
    expect_gamma(g4, 3);
    WilsonMatrix C = g2*g4;
    C*=-1.;

    WilsonMatrix expect = C;
    
    WilsonMatrix got = unit_wm;
    got.ccr(1);
    
    std::cout << "C = ccr(" << 1 << "): ";
    if(!test_eq(expect,got)) return 1;

    got = unit_wm;
    got.ccl(-1);
    
    std::cout << "C = ccl(" << -1 << "): ";
    if(!test_eq(expect,got)) return 1;
  }

  //Test consistency with SpinMatrix
  for(int mu=0;mu<4;mu++){
    WilsonMatrix gmu;
    expect_gamma(gmu, mu);

    WilsonMatrix g5;
    expect_gamma(g5, 5);

    WilsonMatrix expect = gmu;

    SpinMatrix got = SpinMatrix::Gamma(mu);
    std::cout << "SpinMatrix g" << (mu+1) << " ";
    if(!test_eq(expect,got)) return 1;
    
    got = SpinMatrix::GammaMuGamma5(mu);
    expect = gmu*g5;
    
    std::cout << "SpinMatrix g" << (mu+1) << "g5 ";
    if(!test_eq(expect,got)) return 1;
  }


  //Test consistency with CPSspinMatrix
  for(int mu=0;mu<4;mu++){
    WilsonMatrix gmu;
    expect_gamma(gmu, mu);

    WilsonMatrix g5;
    expect_gamma(g5, 5);

    WilsonMatrix expect = gmu;

    CPSspinMatrix<Complex> got;
    got.unit();
    got.gl(mu);
    
    std::cout << "CPSspinMatrix gl(" << mu << ") ";
    if(!test_eq(expect,got)) return 1;
    
    got.unit();
    got.gr(mu);

    std::cout << "CPSspinMatrix gr(" << mu << ") ";
    if(!test_eq(expect,got)) return 1;

    expect = gmu*g5;

    got.unit();
    got.glAx(mu);
    std::cout << "CPSspinMatrix glAx(" << mu << ") ";
    if(!test_eq(expect,got)) return 1;

    got.unit();
    got.grAx(mu);
    std::cout << "CPSspinMatrix grAx(" << mu << ") ";
    if(!test_eq(expect,got)) return 1;
  }


  {
    WilsonMatrix g5;
    expect_gamma(g5, 5);

    WilsonMatrix expect = g5;

    CPSspinMatrix<Complex> got;
    got.unit();
    got.gl(-5);
    
    std::cout << "CPSspinMatrix gl(" << -5 << ") ";
    if(!test_eq(expect,got)) return 1;
    
    got.unit();
    got.gr(-5);

    std::cout << "CPSspinMatrix gr(" << -5 << ") ";
    if(!test_eq(expect,got)) return 1;
  }
  
#ifdef USE_GRID

  //Test consistency with Grid
  for(int mu=0;mu<4;mu++){
    WilsonMatrix gmu;
    expect_gamma(gmu,mu);

    WilsonMatrix expect = gmu;
    
    Grid::iMatrix<Grid::ComplexD,Grid::Ns> unit_grid;
    unit_matrix(unit_grid);

    Grid::iMatrix<Grid::ComplexD,Grid::Ns> got;
    switch(mu){
    case 0:
      rmultGammaX(got, unit_grid);
      break;
    case 1:
      rmultGammaY(got, unit_grid);
      break;
    case 2:
      rmultGammaZ(got, unit_grid);
      break;
    case 3:
      rmultGammaT(got, unit_grid);
      break;
    }
   
    std::cout << "Grid gr(" << mu << ") ";
    if(!test_eq(expect,got)) return 1;
  }

  {
    WilsonMatrix g5;
    expect_gamma(g5, 5);

    WilsonMatrix expect = g5;
    
    Grid::iMatrix<Grid::ComplexD,Grid::Ns> unit_grid;
    unit_matrix(unit_grid);

    Grid::iMatrix<Grid::ComplexD,Grid::Ns> got;
    rmultGamma5(got, unit_grid);
   
    std::cout << "Grid gr(" << -5 << ") ";
    if(!test_eq(expect,got)) return 1;
  }

#endif
        
  return 0;
}


