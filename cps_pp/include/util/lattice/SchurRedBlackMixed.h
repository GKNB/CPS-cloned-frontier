#ifndef GRID_SCHUR_MIXED_H
#define GRID_SCHUR_MIXED_H


namespace Grid {

//  MixedPrecisionConjugateGradientOp
   template<class FieldD,class FieldF, typename std::enable_if< getPrecision<FieldD>::value == 2, int>::type = 0,typename std::enable_if< getPrecision<FieldF>::value == 1, int>::type = 0>
  class MixedPrecisionConjugateGradientOp : public MixedPrecisionConjugateGradient<FieldD,FieldF>, public OperatorFunction<FieldD> {

  public:

      MixedPrecisionConjugateGradientOp(RealD tol, Integer maxinnerit, Integer maxouterit, GridBase* _sp_grid, LinearOperatorBase<FieldF> &_Linop_f, LinearOperatorBase<FieldD> &_Linop_d) :
      MixedPrecisionConjugateGradient<FieldD,FieldF> (tol, maxinnerit, maxouterit, _sp_grid, _Linop_f, _Linop_d) {} ;
      void operator() (LinearOperatorBase<FieldD> &Linop, const FieldD &in, FieldD &out){
              this->MixedPrecisionConjugateGradient<FieldD,FieldF>::operator()(in,out);
      }
  };

//  MixedPrecisionBiCGSTABOp
   template<class FieldD,class FieldF, typename std::enable_if< getPrecision<FieldD>::value == 2, int>::type = 0,typename std::enable_if< getPrecision<FieldF>::value == 1, int>::type = 0>
  class MixedPrecisionBiCGSTABOp : public MixedPrecisionBiCGSTAB<FieldD,FieldF>, public OperatorFunction<FieldD> {

  public:

      MixedPrecisionBiCGSTABOp(RealD tol, Integer maxinnerit, Integer maxouterit, GridBase* _sp_grid, LinearOperatorBase<FieldF> &_Linop_f, LinearOperatorBase<FieldD> &_Linop_d) :
      MixedPrecisionBiCGSTAB<FieldD,FieldF> (tol, maxinnerit, maxouterit, _sp_grid, _Linop_f, _Linop_d) {} ;
      void operator() (LinearOperatorBase<FieldD> &Linop, const FieldD &in, FieldD &out){
              this->MixedPrecisionBiCGSTAB<FieldD,FieldF>::operator()(in,out);
      }
  };

}
#endif
