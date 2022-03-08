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

CPS_END_NAMESPACE
