#ifndef SPIN_FLAVOR_MATRIX_H
#define SPIN_FLAVOR_MATRIX_H

#include<config.h>
#include <alg/alg_base.h>

CPS_START_NAMESPACE

class SpinFlavorMatrix
{
  static const int fsize = 2*4*4*2*2;
  Float u[fsize];	// The matrix


  public:
    SpinFlavorMatrix();
    SpinFlavorMatrix(Float c);
    SpinFlavorMatrix(const Complex& c);
    SpinFlavorMatrix(const SpinFlavorMatrix& m);

    //Set all elements equal to float or complex
    SpinFlavorMatrix& operator=(Float c);
    SpinFlavorMatrix& operator=(const Complex& c);
    SpinFlavorMatrix& operator=(const SpinFlavorMatrix& c);

    Complex& operator()(const int &s1, const int &f1, const int &s2, const int &f2);
    const Complex& operator()(const int &s1, const int &f1, const int &s2, const int &f2) const;
    Complex& operator[](int i) { return ((Complex*)u)[i]; }
    const Complex& operator[](int i) const { return ((Complex*)u)[i]; }

    SpinFlavorMatrix operator*(const Complex &c)const {
        SpinFlavorMatrix ret;
	for(int i=0;i<4*4*2*2;i++)
	  ret[i] = (*this)[i] * c;	
        return ret;
    }

    const SpinFlavorMatrix &operator*=(const Complex &c) {
	for(int i=0;i<4*4*2*2;i++)
	  (*this)[i] *= c;
        return *this;
    }

    SpinFlavorMatrix operator+(const SpinFlavorMatrix &s)const {
      SpinFlavorMatrix ret;
      for(int i=0;i<4*4*2*2;i++)
	ret[i] = (*this)[i] + s[i];	
      return ret;
    }

    const SpinFlavorMatrix &operator+=(const SpinFlavorMatrix &s) {
      for(int i=0;i<4*4*2*2;i++)
	(*this)[i] += s[i];
      return *this;
    }

    SpinFlavorMatrix operator-(const SpinFlavorMatrix &s)const {
      SpinFlavorMatrix ret;
      for(int i=0;i<4*4*2*2;i++)
	ret[i] = (*this)[i] - s[i];	
      return ret;
    }

    const SpinFlavorMatrix &operator-=(const SpinFlavorMatrix &s) {
      for(int i=0;i<4*4*2*2;i++)
	(*this)[i] -= s[i];
      return *this;
    }


    SpinFlavorMatrix operator*(const SpinFlavorMatrix &r)const {
        SpinFlavorMatrix ret(0.0);
	for(int s1=0;s1<4;s1++)
	  for(int f1=0;f1<2;f1++)
	    for(int s2=0;s2<4;s2++)
	      for(int f2=0;f2<2;f2++)
		for(int s3=0;s3<4;s3++)
		  for(int f3=0;f3<2;f3++)
		    ret(s1,f1,s2,f2) += (*this)(s1,f1,s3,f3) * r(s3,f3,s2,f2);
	return ret;
    }

    Complex Trace() const;
};

static inline Rcomplex Trace(const SpinFlavorMatrix &a, const SpinFlavorMatrix &b) {
    Rcomplex ret = 0;
    for(int s1=0;s1<4;s1++)
      for(int f1=0;f1<2;f1++)
	for(int s2=0;s2<4;s2++)
	  for(int f2=0;f2<2;f2++)
	    ret += a(s1,f1,s2,f2) * b(s2,f2,s1,f1);
    return ret;
}







CPS_END_NAMESPACE

#endif
