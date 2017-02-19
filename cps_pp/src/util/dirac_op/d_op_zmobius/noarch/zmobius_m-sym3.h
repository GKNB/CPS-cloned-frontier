//4d precond. mobius Dirac op of symmetric version 2nd kind but include kappa_b mult in dslash_4 :
// 1 - kappa_b M4eo M_5^-1 kappa_b M4oe M_5^{-1} 

void  zmobius_m_sym3 (Vector *out, 
		 Matrix *gauge_field, 
		 Vector *in, 
		 Float mass, 
		 Zmobus *mobius_lib_arg)
{

  //------------------------------------------------------------------
  // Initializations
  //------------------------------------------------------------------
  const size_t f_size = 24 * mobius_lib_arg->vol_4d * mobius_lib_arg->ls / 2;
  const int ls=mobius_lib_arg->ls;
  const int vol_4d_cb = mobius_lib_arg->vol_4d / 2;
  const int ls_stride = 24 * vol_4d_cb;
  const int local_ls = GJP.SnodeSites();
  const int s_nodes = GJP.Snodes();
  const int global_ls = local_ls * s_nodes;
  const int s_node_coor = GJP.SnodeCoor();

  
  Vector  *frm_tmp2 = (Vector *) mobius_lib_arg->frm_tmp2;
  //Vector *temp = (Vector *) smalloc(f_size * sizeof(Float));
  Float norm;

  
  //  out = [ 1 - kappa_b Meo M5inv kappa_b Moe M5inv] in
  // (dslash_5 uses (1+-g5), not P_R,L, i.e. no factor of 1/2 which is here out front)
  //    1. ftmp2 = kappa_b Meo M5inv kappa_b Moe  M5inv in
  //    2. out <-  in
  //    3. out -=  ftmp2


  //--------------------------------------------------------------
  //    1. ftmp2 =  Meo M5inv Moe M5inv in
  //--------------------------------------------------------------

  moveFloat((IFloat*)frm_tmp2, (IFloat*)in, f_size);
  
  //------------------------------------------------------------------
  // Apply M_5^-1 (hopping in 5th dir + diagonal)
  //------------------------------------------------------------------
  zmobius_m5inv(frm_tmp2, mass, 0, mobius_lib_arg,mobius_lib_arg->zmobius_kappa_ratio);
  DEBUG_MOBIUS_DSLASH("mobius_m5inv %e\n", time_elapse());

  Complex *b_coeff = GJP.ZMobius_b();
  Complex *c_coeff = GJP.ZMobius_c();
  const Complex *kappa_b = mobius_lib_arg->zmobius_kappa_b;
  Complex Imag(0,1);
  DEBUG_MOBIUS_DSLASH("before new b,c coeff %e\n", time_elapse());
  for(int s=0;s<ls;++s){
    b_coeff[s] *= kappa_b[s];
    b_coeff[s] *= Imag;
    c_coeff[s] *= kappa_b[s];
    c_coeff[s] *= Imag;
  }
  DEBUG_MOBIUS_DSLASH("after new b,c coeff %e\n", time_elapse());

  // Apply Dslash O <- E
  //------------------------------------------------------------------
  time_elapse();
  zmobius_dslash_4(out, gauge_field, frm_tmp2, 0, 0, mobius_lib_arg, mass);
  DEBUG_MOBIUS_DSLASH("mobius_dslash_4 dag 0 %e\n", time_elapse());

  //------------------------------------------------------------------
  // Apply M_5^-1 (hopping in 5th dir + diagonal)
  //------------------------------------------------------------------
  zmobius_m5inv(out, mass, 0, mobius_lib_arg,mobius_lib_arg->zmobius_kappa_ratio);
  DEBUG_MOBIUS_DSLASH("mobius_m5inv %e\n", time_elapse());
  

  //------------------------------------------------------------------
  // Apply Dslash E <- O
  //------------------------------------------------------------------
  zmobius_dslash_4(frm_tmp2, gauge_field, out, 1, 0, mobius_lib_arg, mass);
  DEBUG_MOBIUS_DSLASH("mobius_dslash_4 dag 0 %e\n", time_elapse());
 

  //------------------------------------------------------------------
  //    2. out <-  in
  //------------------------------------------------------------------
#if 1
  moveFloat((IFloat*)out, (IFloat*)in, f_size);
#else
 !  SENTINEL !  USE_BLAS is not supported YET for complexified mobius
  cblas_dcopy(f_size, (IFloat*)in, 1, (IFloat*)out, 1);
#endif
  DEBUG_MOBIUS_DSLASH("out <- in %e\n", time_elapse());
  
  //------------------------------------------------------------------
  //    3. out +=  ftmp2
  //------------------------------------------------------------------
#if 1
#if 0
  fTimesV1PlusV2((IFloat*)out, minus_kappa_b_sq, (IFloat*)frm_tmp2,
		 (IFloat *)out, f_size);
#else

  vecAddEquVec((IFloat*)out, (IFloat*) frm_tmp2, f_size);

#endif
#else
  ! SENTINEL did not fix below !
  cblas_daxpy(f_size, minus_kappa_b_sq, (IFloat*)frm_tmp2,1, 
            (IFloat *)out,  1);
#endif
  DEBUG_MOBIUS_DSLASH("zmobius out+= ftmp2 %e\n", time_elapse());
  
  // Flops count in this function is two AXPY = 4 flops per vector elements
  //DiracOp::CGflops +=  3*f_size; 

  // Restore b,c coeffs
  for(int s=0;s<ls;++s){
    b_coeff[s] *= -Imag;
    b_coeff[s] /= kappa_b[s];
    c_coeff[s] *= -Imag;
    c_coeff[s] /= kappa_b[s];
  }
  DEBUG_MOBIUS_DSLASH("restore b,c coeffs %e\n", time_elapse());
  
}


