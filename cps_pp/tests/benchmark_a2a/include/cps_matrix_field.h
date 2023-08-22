#pragma once

CPS_START_NAMESPACE

struct _tr{
  template<typename MatrixType>
  accelerator_inline auto operator()(const MatrixType &matrix) const ->decltype(matrix.Trace()){ return matrix.Trace(); }  
};
struct _trtr{
  template<typename MatrixType>
  accelerator_inline auto operator()(const MatrixType &a, const MatrixType &b) const ->decltype(a.Trace()*b.Trace()){ return a.Trace()*b.Trace(); }  
};

template<typename VectorMatrixType>
struct _trtrV{
  typedef typename VectorMatrixType::scalar_type OutputType;
  accelerator_inline void operator()(OutputType &out, const VectorMatrixType &a, const VectorMatrixType &b, const int lane) const{ 
    typename VectorMatrixType::scalar_type tmp, tmp2; //each thread will have one of these but will only write to a single thread    
    Trace(tmp, a, lane);
    Trace(tmp2, b, lane);
    mult(out, tmp, tmp2, lane);
  }
};



template<typename GridA2Apolicies>
void benchmarkCPSmatrixField(const int ntests){
  std::cout << "Starting CPSmatrixField benchmark\n";

  typedef typename GridA2Apolicies::ComplexType ComplexType;
  const int nsimd = ComplexType::Nsimd();      

  FourDSIMDPolicy<DynamicFlavorPolicy>::ParamType simd_dims;
  FourDSIMDPolicy<DynamicFlavorPolicy>::SIMDdefaultLayout(simd_dims,nsimd,2);

  typedef CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > SCFmatrixField;

  static_assert(isCPSsquareMatrix<typename SCFmatrixField::FieldSiteType>::value == 1);


  SCFmatrixField m1(simd_dims);
  m1.testRandom();
  SCFmatrixField m2(simd_dims);
  m2.testRandom();



  typename std::decay<decltype(Trace(m1))>::type tr_m1(simd_dims);
  
  if(0){
    //Trace
    Float total_time_trace = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1);
    }
    total_time_trace += dclock();
    
    double Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double tavg = total_time_trace/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //unop Trace
    Float total_time_trace = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = unop(m1, _tr());
    }
    total_time_trace += dclock();
    
    double Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double tavg = total_time_trace/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Unop trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }


  if(0){
    //unopV Trace
    Float total_time_trace = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = unop_v(m1, _trV<typename SCFmatrixField::FieldSiteType>());
    }
    total_time_trace += dclock();
    
    double Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double tavg = total_time_trace/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Unop_v trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }


  if(0){
    //Trace * trace
    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1) * Trace(m2);
    }
    total_time += dclock();
    
    double tr_Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double mul_flops = m1.size() * nsimd * 6; //complex mult

    double Flops = 2*tr_Flops + mul_flops;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField)*Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //binop(Trace * trace)
    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = binop(m1, m2, _trtr());
    }
    total_time += dclock();
    
    double tr_Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double mul_flops = m1.size() * nsimd * 6; //complex mult

    double Flops = 2*tr_Flops + mul_flops;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Binop Trace(SCFmatrixField)*Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //binop_v(Trace * trace)
    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = binop_v(m1,m2, _trtrV<typename SCFmatrixField::FieldSiteType>());
    }
    total_time += dclock();
    
    double tr_Flops = m1.size() * 24 * 2 * nsimd; //sum of diagonal elements per site, each sum re/im with SIMD vectorization
    double mul_flops = m1.size() * nsimd * 6; //complex mult

    double Flops = 2*tr_Flops + mul_flops;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Binop_v Trace(SCFmatrixField)*Trace(SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //operator* 
    SCFmatrixField m3(simd_dims);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      m3 = m1 * m2;
    }
    total_time += dclock();
    
    //24*24 matrix multiply
    //Flops = 24* 24 * ( 24 madds )     madd = 8 Flops * nsimd
    double Flops = m1.size() * 24 * 24 * ( 24 * 8 * nsimd );
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("SCFmatrixField*SCFmatrixField %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //operator* binop_v
    SCFmatrixField m3(simd_dims);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      m3 = binop_v(m1,m2, _timesV<typename SCFmatrixField::FieldSiteType>());
    }
    total_time += dclock();
    
    //24*24 matrix multiply
    //Flops = 24* 24 * ( 24 madds )     madd = 8 Flops * nsimd
    double Flops = m1.size() * 24 * 24 * ( 24 * 8 * nsimd );
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Binop_v SCFmatrixField*SCFmatrixField %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

  if(0){
    //gl
    SCFmatrixField m3 = m1;

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      gl(m3, 0);
    }
    total_time += dclock();
    double tavg = total_time/ntests;
    
    printf("gl(SCFmatrixField, 0) %d iters: %g secs\n",ntests,tavg);
  }


  if(0){
    //gl_r
    SCFmatrixField m3 = gl_r(m1,0);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      m3 = gl_r(m1, 0);
    }
    total_time += dclock();
    double tavg = total_time/ntests;
    
    printf("gl_r(SCFmatrixField, 0) %d iters: %g secs\n",ntests,tavg);
  }

  if(0){
    //gr_r
    SCFmatrixField m3 = gr_r(m1,0);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      m3 = gr_r(m1, 0);
    }
    total_time += dclock();
    double tavg = total_time/ntests;
    
    printf("gr_r(SCFmatrixField, 0) %d iters: %g secs\n",ntests,tavg);
  } 

  if(1){
    //gr_r
    SCFmatrixField m3(m1);
    gr_r(m3,m1,0);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      gr_r(m3,m1, 0);
    }
    total_time += dclock();
    double tavg = total_time/ntests;
    
    printf("gr_r(SCFmatrixField, SCFmatrixField, 0) %d iters: %g secs\n",ntests,tavg);
  } 


  
  if(0){
    //Trace(M1*M2)
    tr_m1 = Trace(m1,m2);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1,m2);
    }
    total_time += dclock();
    
    //\sum_{ij} a_{ij}b_{ji}
    double Flops = m1.size() * 24 * 24 * 8 * nsimd;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField*SCFmatrixField) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }


  if(0){    
    //Trace(M1*Ms)  where Ms is not a field
    CPSspinColorFlavorMatrix<ComplexType> Ms;
    {
      CPSautoView(m2_v,m2,HostRead);
      Ms = *m2_v.site_ptr(size_t(0));
    }
    tr_m1 = Trace(m1,Ms);

    Float total_time = -dclock();
    for(int iter=0;iter<ntests;iter++){
      tr_m1 = Trace(m1,Ms);
    }
    total_time += dclock();
    
    //\sum_{ij} a_{ij}b_{ji}
    double Flops = m1.size() * 24 * 24 * 8 * nsimd;
    double tavg = total_time/ntests;
    double Mflops = double(Flops)/tavg/1024./1024.;
    
    printf("Trace(SCFmatrixField*SCFmatrix) %d iters: %g secs   %f Mflops\n",ntests,tavg,Mflops);
  }

}


CPS_END_NAMESPACE
