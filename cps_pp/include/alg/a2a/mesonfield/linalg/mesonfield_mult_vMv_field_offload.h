#ifndef _MULT_VMV_FIELD_OFFLOAD_H_
#define _MULT_VMV_FIELD_OFFLOAD_H_

CPS_START_NAMESPACE

template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR,
	 typename ComplexClass>
class _mult_vMv_field_offload_v{};

#ifdef USE_GRID

template<typename mf_Policies, int isGparity>
struct _mult_vMv_field_offload_fields{};

template<typename mf_Policies>
struct _mult_vMv_field_offload_fields<mf_Policies,1>{
  typedef CPSspinColorFlavorMatrix<typename mf_Policies::ComplexType> VectorMatrixType;
  typedef CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy> PropagatorField;
};
template<typename mf_Policies>
struct _mult_vMv_field_offload_fields<mf_Policies,0>{
  typedef CPSspinMatrix<CPScolorMatrix<typename mf_Policies::ComplexType> > VectorMatrixType;
  typedef CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy> PropagatorField;
};

//For A,B,C... \in { A2AvectorW, A2AvectorV, A2AvectorWfftw, A2AvectorVfftw }
//Compute   A (BC) D    where  (BC) is a meson field, A, D are A2A vectors
template<typename mf_Policies, 
	 template <typename> class lA2AfieldL,  template <typename> class lA2AfieldR,
	 template <typename> class rA2AfieldL,  template <typename> class rA2AfieldR>
struct _mult_vMv_field_offload_v<mf_Policies,lA2AfieldL,lA2AfieldR,rA2AfieldL,rA2AfieldR,grid_vector_complex_mark>{
  typedef _mult_vMv_field_offload_fields<mf_Policies, mf_Policies::GPARITY> fdef;
  typedef typename fdef::VectorMatrixType VectorMatrixType;
  typedef typename fdef::PropagatorField PropagatorField;

  typedef lA2AfieldL<mf_Policies> lA2AfieldType;
  typedef rA2AfieldR<mf_Policies> rA2AfieldType;
  typedef A2AmesonField<mf_Policies,lA2AfieldR,rA2AfieldL> MesonFieldType;

  typedef typename lA2AfieldType::DilutionType iLeftDilutionType;
  typedef typename MesonFieldType::LeftDilutionType iRightDilutionType;
  
  typedef typename MesonFieldType::RightDilutionType jLeftDilutionType;    
  typedef typename rA2AfieldType::DilutionType jRightDilutionType;

  typedef typename mf_Policies::ComplexType VectorComplexType;
  typedef typename SIMT<VectorComplexType>::value_type ScalarComplexType;

  static void doit(PropagatorField &into,
		   const lA2AfieldType &l,
		   const MesonFieldType &M,
		   const rA2AfieldType &r,
		   bool conj_l, bool conj_r){
    into.zero();

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);
    
    A2Aparams i_params(l), j_params(r);
    StandardIndexDilution idil(i_params), jdil(j_params);
    
    int ni = idil.getNmodes();
    int nj = jdil.getNmodes();
    
    int nsimd = VectorComplexType::Nsimd();
    
    size_t vol4d = into.size();

    typedef SIMT<VectorComplexType> ACC;

    accelerator_for(x4d, vol4d, nsimd, 
		    {
		      VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d);
		      size_t xop, top;
		      into.fourToThree(xop, top, x4d);

		      for(int sl=0;sl<4;sl++){
			for(int cl=0;cl<3;cl++){
			  for(int fl=0;fl<2;fl++){	  
			    
			    for(int sr=0;sr<4;sr++){
			      for(int cr=0;cr<3;cr++){
				for(int fr=0;fr<2;fr++){
				  VectorComplexType &out = vsite_mat(sl,sr)(cl,cr)(fl,fr);

				  for(int i=0;i<ni;i++){
				    
				    ScalarComplexType lval_tmp = ACC::read(l.elem(i,xop,top,cl+3*sl,fl));
				    ScalarComplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
		  
				    for(int j=0;j<nj;j++){
				      ScalarComplexType rval_tmp = ACC::read(r.elem(j,xop,top,cr+3*sr,fr));
				      ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
				      
				      ScalarComplexType Mval = M.elem(i,j);

				      ScalarComplexType val = ACC::read(out) + lval * Mval * rval;
				      ACC::write(out, val);
				    }
				  }
				}
			      }
			    }
			  }
			}
		      }
		    }
		    );
    



  }








  /* static inline void setupModeIndexSets(ModeIndexSet* into, int nhit, int nf){ */
  /*   int o=0; */
  /*   for(int t=0;t<GJP.TnodeSites();t++)  */
  /*     for(int h=0;h<nhit;h++)  */
  /* 	for(int f=0;f<nf;f++)  */
  /* 	  for(int sc=0;sc<12;sc++){ */
  /* 	    ModeIndexSet &coord = into[o++]; */
  /* 	    coord.spin_color = sc; */
  /* 	    coord.flavors = f; */
  /* 	    coord.hit = h; */
  /* 	    coord.time = GJP.TnodeSites()*GJP.TnodeCoor()+t; //global time */
  /* 	  } */
  /* } */


  /* static void v2(PropagatorField &into, */
  /* 		   const lA2AfieldType &l, */
  /* 		   const MesonFieldType &M, */
  /* 		   const rA2AfieldType &r, */
  /* 		   bool conj_l, bool conj_r){ */
  /*   into.zero(); */

  /*   ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l); */
  /*   ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r); */

  /*   int nodeLt = into.nodeSites(3); */
  /*   assert(nodeLt == GJP.TnodeSites()); //cannot be SIMD packed in t-direction */

  /*   //We need ModeIndexSet instances for each hit, spin_color, flavor, timeslice */
  /*   int nhit_l =  l.getNhits(), nhit_r = r.getNhits(); */
  /*   int nf_l = l.getNflavors(), nf_r = r.getNflavors(); */
  /*   int nidxset_l = nodeLt * 12 * nhit_l * nf_l; //use mapping  color + 3*(spin + 4*(flavor + nf_l*( hit + nhit_l * t )) ) */
  /*   int nidxset_r = nodeLt * 12 * nhit_r * nf_r; */
  /*   ModeIndexSet* lcoord_all = (ModeIndexSet*)managed_alloc_check(nidxset_l * sizeof(ModeIndexSet)); */
  /*   ModeIndexSet* rcoord_all = (ModeIndexSet*)managed_alloc_check(nidxset_r * sizeof(ModeIndexSet)); */
  /*   setupModeIndexSets(lcoord_all, nhit_l, nf_l); */
  /*   setupModeIndexSets(rcoord_all, nhit_r, nf_r); */
  /*   int nl = j_ind */


  /*   A2Aparams i_params(l), j_params(r); */
  /*   StandardIndexDilution idil(i_params), jdil(j_params); */
    
  /*   int ni = idil.getNmodes(); */
  /*   int nj = jdil.getNmodes(); */
    
  /*   int nsimd = VectorComplexType::Nsimd(); */
    
  /*   size_t vol4d = into.size(); */

  /*   typedef SIMT<VectorComplexType> ACC; */

  /*   accelerator_for(x4d, vol4d, nsimd,  */
  /* 		    { */
  /* 		      VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d); */
  /* 		      size_t xop, top; */
  /* 		      into.fourToThree(xop, top, x4d); */

  /* 		      for(int sl=0;sl<4;sl++){ */
  /* 			for(int cl=0;cl<3;cl++){ */
  /* 			  for(int fl=0;fl<2;fl++){	   */
			    
  /* 			    for(int sr=0;sr<4;sr++){ */
  /* 			      for(int cr=0;cr<3;cr++){ */
  /* 				for(int fr=0;fr<2;fr++){ */
  /* 				  VectorComplexType &out = vsite_mat(sl,sr)(cl,cr)(fl,fr); */

  /* 				  for(int i=0;i<ni;i++){ */
				    
  /* 				    ScalarComplexType lval_tmp = ACC::read(l.elem(i,xop,top,cl+3*sl,fl)); */
  /* 				    ScalarComplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp; */
		  
  /* 				    for(int j=0;j<nj;j++){ */
  /* 				      ScalarComplexType rval_tmp = ACC::read(r.elem(j,xop,top,cr+3*sr,fr)); */
  /* 				      ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp; */
				      
  /* 				      ScalarComplexType Mval = M.elem(i,j); */

  /* 				      ScalarComplexType val = ACC::read(out) + lval * Mval * rval; */
  /* 				      ACC::write(out, val); */
  /* 				    } */
  /* 				  } */
  /* 				} */
  /* 			      } */
  /* 			    } */
  /* 			  } */
  /* 			} */
  /* 		      } */
  /* 		    } */
  /* 		    ); */
    



  /* } */







};



#endif //USE_GRID


CPS_END_NAMESPACE

#endif
