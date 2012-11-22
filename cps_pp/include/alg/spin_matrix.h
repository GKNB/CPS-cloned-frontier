//------------------------------------------------------------------
//
// Header file for the SpinMatrix class.
//
// This file contains the declarations of the SpinMatrix 
//
//------------------------------------------------------------------


#ifndef INCLUDED_SPINMARTRIX_H
#define INCLUDED_SPINMARTRIX_H

#include <string.h>
#include <util/data_types.h>
CPS_START_NAMESPACE
class SpinMatrix;


enum{SPINS=4};

//------------------------------------------------------------------
// The SpinMatrix class.
//------------------------------------------------------------------
class SpinMatrix
{
    Float u[2*SPINS*SPINS];	// The matrix


  public:
    // CREATORS
    SpinMatrix() {}
    SpinMatrix(Float c);
    SpinMatrix(const Complex& c);

    SpinMatrix& operator=(Float c);
    SpinMatrix& operator=(const Complex& c);

    void ZeroSpinMatrix(void) {
        for(int i=0; i<2*SPINS*SPINS; i++) u[i] = 0;
    }
    void UnitSpinMatrix(void);

    // ACCESSORS
    Complex& operator()(int i, int j) {
        return ((Complex*)u)[i*SPINS+j];
    }
    const Complex& operator()(int i, int j) const {
        return ((Complex*)u)[i*SPINS+j];
    }

    Complex& operator[](int i) {
        return ((Complex*)u)[i];
    }
    const Complex& operator[](int i)const {
        return ((Complex*)u)[i];
    }

    Complex Tr() const;
};

static inline Rcomplex Trace(const SpinMatrix &a, const SpinMatrix &b) {
    Rcomplex ret = 0;
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            ret += a(i, j) * b(j, i);
        }
    }
    return ret;
}

CPS_END_NAMESPACE
#endif

