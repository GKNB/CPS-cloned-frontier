//4d precond. mobius Dirac op:
// M_5 - kappa_b M4eo M_5^-1 kappa_b M4oe
void  zmobius_m(Vector *out, 
	       Matrix *gauge_field, 
	       Vector *in, 
	       Float mass, 
	       Dwf *mobius_lib_arg)
{

  //------------------------------------------------------------------
  // Initializations
  //------------------------------------------------------------------
  const int f_size = 24 * mobius_lib_arg->vol_4d * mobius_lib_arg->ls / 2;
  const int ls=mobius_lib_arg->ls;
  const int vol_4d_cb = mobius_lib_arg->vol_4d / 2;
  const int ls_stride = 24 * vol_4d_cb;
  const int local_ls = GJP.SnodeSites();
  const int s_nodes = GJP.Snodes();
  const int global_ls = local_ls * s_nodes;
  const int s_node_coor = GJP.SnodeCoor();

  
#if 0
  const Float kappa_ratio = mobius_lib_arg->mobius_kappa_b/mobius_lib_arg->mobius_kappa_c;
  const Float minus_kappa_b_sq = -mobius_lib_arg->mobius_kappa_b * mobius_lib_arg->mobius_kappa_b;
#endif
  


  Vector  *frm_tmp2 = (Vector *) mobius_lib_arg->frm_tmp2;
  //Vector *temp = (Vector *) smalloc(f_size * sizeof(Float));
  Float norm;

  
  //  out = [ 1 + kappa_b/kappa_c 1/2 dslash_5  - kappa_b Meo M5inv kappa_b Moe] in
  // (dslash_5 uses (1+-g5), not P_R,L, i.e. no factor of 1/2 which is here out front)
  //    1. ftmp2 = - kappa_b Meo M5inv kappa_b Moe in
  //    2. out <-  in
  //    3. out +=  ftmp2
  //    4. out +=  -kappa_b/kappa_c /2 dslash_5 in
  //         (done by the dslash_5 with a5_inv = -kappa_b/kappa_c/2 *GJP.MobiusA5Inv() ) 


  //--------------------------------------------------------------
  //    1. ftmp2 = Meo M5inv Moe in
  //--------------------------------------------------------------
  // Apply Dslash O <- E
  //------------------------------------------------------------------
  time_elapse();
  zmobius_dslash_4(out, gauge_field, in, 0, 0, mobius_lib_arg, mass);
  DEBUG_MOBIUS_DSLASH("mobius_dslash_4 %e\n", time_elapse());

  //------------------------------------------------------------------
  // Apply kappa_b(s)
  //------------------------------------------------------------------
  const Complex *kappa_b = mobius_lib_arg->zmobius_kappa_b;
  for(int s=0;s<local_ls;++s){
    int glb_s = s + local_ls*s_node_coor;
    Complex* cp = (Complex*)( (Float*)out + s * ls_stride);
    vecTimesEquComplex(cp, kappa_b[glb_s], ls_stride);
  }
  DEBUG_MOBIUS_DSLASH("kappa_b(s)\n", time_elapse());

  //------------------------------------------------------------------
  // Apply M_5^-1 (hopping in 5th dir + diagonal)
  //------------------------------------------------------------------
  zmobius_m5inv(out, mass, 0, mobius_lib_arg, mobius_lib_arg->zmobius_kappa_ratio);
  DEBUG_MOBIUS_DSLASH("mobius_m5inv %e\n", time_elapse());
  

  //------------------------------------------------------------------
  // Apply Dslash E <- O
  //------------------------------------------------------------------
  zmobius_dslash_4(frm_tmp2, gauge_field, out, 1, 0, mobius_lib_arg, mass);
  DEBUG_MOBIUS_DSLASH("mobius_dslash_4 %e\n", time_elapse());
  
  //------------------------------------------------------------------
  // Apply - kappa_b(s)  : note the minus in font of kappa_b
  //------------------------------------------------------------------
  for(int s=0;s<local_ls;++s){
    int glb_s = s + local_ls*s_node_coor;
    Complex* cp = (Complex*)( (Float*)frm_tmp2 +s*ls_stride);
    vecTimesEquComplex((Complex*) cp, - kappa_b[glb_s], ls_stride);
  }
  DEBUG_MOBIUS_DSLASH("-kappa_b(s)\n", time_elapse());

  //------------------------------------------------------------------
  //    2. out <-  in
  //------------------------------------------------------------------
#ifndef USE_BLAS
  moveFloat((IFloat*)out, (IFloat*)in, f_size);
#else
 !  SENTINEL !  USE_BLAS is not supported YET for complexified mobius
  cblas_dcopy(f_size, (IFloat*)in, 1, (IFloat*)out, 1);
#endif
  DEBUG_MOBIUS_DSLASH("out <- in %e\n", time_elapse());
  
  //------------------------------------------------------------------
  //    3. out +=  ftmp2
  //------------------------------------------------------------------
#ifndef USE_BLAS
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
  
  //------------------------------------------------------------------
  //    4. out +=  kappa_ratio dslash_5 in
  //------------------------------------------------------------------
  zmobius_kappa_dslash_5_plus_cmplx(out, in, mass, 0, mobius_lib_arg,
    mobius_lib_arg->zmobius_kappa_ratio);
  DEBUG_MOBIUS_DSLASH("zmobius_kappa_dslash_5_plus %e\n", time_elapse());

  // Flops count in this function is two AXPY = 4 flops per vector elements
  //DiracOp::CGflops +=  3*f_size; 

}


