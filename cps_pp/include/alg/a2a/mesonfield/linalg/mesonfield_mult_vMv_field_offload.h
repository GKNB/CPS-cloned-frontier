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

  //A slow but simple implementation ignoring the index compression
  static void v1(PropagatorField &into,
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

    int nf = GJP.Gparity() ? 2:1;

    typedef SIMT<VectorComplexType> ACC;

    accelerator_for(x4d, vol4d, nsimd, 
		    {
		      VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d);
		      size_t xop, top;
		      into.fourToThree(xop, top, x4d);

		      for(int sl=0;sl<4;sl++){
			for(int cl=0;cl<3;cl++){
			  for(int fl=0;fl<nf;fl++){	  
			    
			    for(int sr=0;sr<4;sr++){
			      for(int cr=0;cr<3;cr++){
				for(int fr=0;fr<nf;fr++){
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






  //A much faster, but still slow, version that exploits the index compression
  static void v2(PropagatorField &into,
  		   const lA2AfieldType &l,
  		   const MesonFieldType &M,
  		   const rA2AfieldType &r,
  		   bool conj_l, bool conj_r){
    into.zero();

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);

    int nodeLt = into.nodeSites(3);
    assert(nodeLt == GJP.TnodeSites()); //cannot be SIMD packed in t-direction

    int nf_l = l.getNflavors(), nf_r = r.getNflavors();

    int nsimd = VectorComplexType::Nsimd();
    
    size_t vol4d = into.size();

    size_t t_off = GJP.TnodeSites() * GJP.TnodeCoor();

    typedef SIMT<VectorComplexType> ACC;

    accelerator_for(x4d, vol4d, nsimd,
  		    {
  		      VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d);
  		      size_t xop, top;
  		      into.fourToThree(xop, top, x4d);
		      size_t t_glob = top + t_off;

		      modeIndexSet ilp, irp, jlp, jrp;
		      ilp.time = jrp.time = t_glob;
		      irp.time = M.getRowTimeslice();
		      jlp.time = M.getColTimeslice();

		      for(int fl=0;fl<nf_l;fl++){
			ilp.flavor = irp.flavor = fl;
			for(int sl=0;sl<4;sl++){
			  for(int cl=0;cl<3;cl++){
			    ilp.spin_color = irp.spin_color = cl + 3*sl;

			    const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
			    size_t ni = i_ind_pairs.size();

			    for(int fr=0;fr<nf_r;fr++){
			      jlp.flavor = jrp.flavor = fr;
			      for(int sr=0;sr<4;sr++){
				for(int cr=0;cr<3;cr++){
				  jlp.spin_color = jrp.spin_color = cr + 3*sr;
				    
				  const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
				  size_t nj = j_ind_pairs.size();

				  VectorComplexType &out = vsite_mat(sl,sr)(cl,cr)(fl,fr);
				    
				  for(size_t i=0;i<ni;i++){
				    int il = i_ind_pairs[i].first, ir = i_ind_pairs[i].second;
				      
				    ScalarComplexType lval_tmp = ACC::read(l.nativeElem(il,x4d,cl+3*sl,fl));
				    ScalarComplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
		  
				    for(size_t j=0;j<nj;j++){
				      int jl = j_ind_pairs[j].first, jr = j_ind_pairs[j].second;

				      ScalarComplexType rval_tmp = ACC::read(r.nativeElem(jr,x4d,cr+3*sr,fr));
				      ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
					
				      ScalarComplexType Mval = M(ir,jl);
					
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



  //This version uses index compression and index blocking
  static void v3(PropagatorField &into,
  		   const lA2AfieldType &l,
  		   const MesonFieldType &M,
  		   const rA2AfieldType &r,
  		   bool conj_l, bool conj_r){
    into.zero();

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);

    int nodeLt = into.nodeSites(3);
    assert(nodeLt == GJP.TnodeSites()); //cannot be SIMD packed in t-direction

    int nf_l = l.getNflavors(), nf_r = r.getNflavors();

    int nsimd = VectorComplexType::Nsimd();
    
    size_t vol4d = into.size();

    size_t t_off = GJP.TnodeSites() * GJP.TnodeCoor();

    typedef SIMT<VectorComplexType> ACC;

    size_t blocksize = BlockedvMvOffloadArgs::b;

    accelerator_for(x4d, vol4d, nsimd,
  		    {
  		      VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d);
  		      size_t xop, top;
  		      into.fourToThree(xop, top, x4d);
		      size_t t_glob = top + t_off;

		      modeIndexSet ilp, irp, jlp, jrp;
		      ilp.time = jrp.time = t_glob;
		      irp.time = M.getRowTimeslice();
		      jlp.time = M.getColTimeslice();

		      for(int fl=0;fl<nf_l;fl++){
			ilp.flavor = irp.flavor = fl;
			for(int sl=0;sl<4;sl++){
			  for(int cl=0;cl<3;cl++){
			    ilp.spin_color = irp.spin_color = cl + 3*sl;

			    const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
			    size_t ni = i_ind_pairs.size();
			    size_t nblocki = blocksize > ni ? 1 : (ni + blocksize-1) / blocksize; //second statement guarantees partial block to handle remaining indices that don't fit a block

			    for(int fr=0;fr<nf_r;fr++){
			      jlp.flavor = jrp.flavor = fr;
			      for(int sr=0;sr<4;sr++){
				for(int cr=0;cr<3;cr++){
				  jlp.spin_color = jrp.spin_color = cr + 3*sr;
				    
				  const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
				  size_t nj = j_ind_pairs.size();
				  size_t nblockj = blocksize > nj ? 1 : (nj + blocksize-1) / blocksize;

				  VectorComplexType &out = vsite_mat(sl,sr)(cl,cr)(fl,fr);
				    
				  for(int bi=0;bi<nblocki;bi++){
				    size_t istart = bi * blocksize;
				    size_t ilessthan = std::min(istart + blocksize, ni);

				    for(int bj=0;bj<nblockj;bj++){
				      size_t jstart = bj * blocksize;
				      size_t jlessthan = std::min(jstart + blocksize, nj);

				      for(size_t i=istart;i<ilessthan;i++){
					int il = i_ind_pairs[i].first, ir = i_ind_pairs[i].second;
					
					ScalarComplexType lval_tmp = ACC::read(l.nativeElem(il,x4d,cl+3*sl,fl));
					ScalarComplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
					
					for(size_t j=jstart;j<jlessthan;j++){
					  int jl = j_ind_pairs[j].first, jr = j_ind_pairs[j].second;
					    
					  ScalarComplexType rval_tmp = ACC::read(r.nativeElem(jr,x4d,cr+3*sr,fr));
					  ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
					    
					  ScalarComplexType Mval = M(ir,jl);
					    
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
  		      }
  		    }
  		    );
    
  }





  static void v4(PropagatorField &into,
  		   const lA2AfieldType &l,
  		   const MesonFieldType &M,
  		   const rA2AfieldType &r,
  		   bool conj_l, bool conj_r){
    into.zero();

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);

    assert(into.nodeSites(3) == GJP.TnodeSites()); //cannot be SIMD packed in t-direction
    int Lt = GJP.Tnodes() * GJP.TnodeSites();
    int nf = GJP.Gparity() + 1;
    int nsimd = VectorComplexType::Nsimd();
    size_t vol4d = into.size();
    size_t t_off = GJP.TnodeSites() * GJP.TnodeCoor();
    size_t blocksize = BlockedvMvOffloadArgs::b;

    typedef SIMT<VectorComplexType> ACC;

    //Need to compute \sum_i\sum_j v(il)_{scl,fl}(x)  M(ir, jl) * v(jr)_{scr,fr}(x)
    //Break up into  Mr(ir)_{scr,fr}(x) =  \sum_j M(ir, jl) * v(jr)_{scr,fr}(x)
    //and            \sum_i v(il)_{scl,fl}(x) Mr(ir)_{scr,fr}(x)
    //Compute Mr(ir)_{scr,fr}(x) for blocks of ir, but to be efficient compute only for the ir we actually need

    //Tune block size to reduce memory requirement
    VectorComplexType* Mr = (VectorComplexType*)managed_alloc_check( BlockedvMvOffloadArgs::b * 12 * nf * vol4d * sizeof(VectorComplexType)); //block in ir

    //Which ir do we actually need?
    std::vector<bool> ir_need(M.getNrows(), false);
    modeIndexSet ilp, irp;

    irp.time = M.getRowTimeslice();
    for(int tv=0;tv<Lt;tv++){
      ilp.time = tv;
      for(int f=0;f<nf;f++){
	ilp.flavor = irp.flavor = f;
	for(int sc=0;sc<12;sc++){
	  ilp.spin_color = irp.spin_color = sc;
	
	  const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
	  size_t ni = i_ind_pairs.size();
	
	  for(int i=0;i<ni;i++){
	    int ir = i_ind_pairs[i].second;
	    ir_need[ir] = true;
	  }
	}
      }
    }
    
    //Create a map and inverse map to the ir we need
    size_t xx=0;
    std::vector<size_t> ir_to_tmpidx_map(M.getNrows(),-1);
    std::vector<size_t> tmpidx_to_ir_map;
    for(size_t ir=0;ir<M.getNrows();ir++){
      if(ir_need[ir]){
	ir_to_tmpidx_map[ir] = xx++;
	tmpidx_to_ir_map.push_back(ir);
      }
    }
    

    //Block over the subset of ir and perform the M*v multiplication for each site, spin/color and flavor, then do the v*(Mv) product still within the block
    size_t nir_needed = xx;
    size_t nir_blocks = (xx + blocksize-1)/blocksize;
    for(size_t irblock=0; irblock<nir_blocks; irblock++){
      size_t tmpir_start = irblock * blocksize; //start of block in tmpir index
      size_t tmpir_lessthan = std::min( tmpir_start + blocksize,  nir_needed );

      //Step 1: Compute \sum_j  M(ir, jl) * v(jr)_{sc,f}(x)     for ir in block
      accelerator_for(x4d, vol4d, nsimd,
		      {
			size_t xop, top;
			into.fourToThree(xop, top, x4d);
			size_t t_glob = top + t_off;

			modeIndexSet jlp, jrp;
			jlp.time = M.getNcols();
			jrp.time = t_glob;

			for(size_t tmpir = tmpir_start; tmpir < tmpir_lessthan; tmpir++){ //the element of the current block
			  size_t irblock_elem = tmpir - tmpir_start;
			  
			  for(int fr=0;fr<nf;fr++){
			    jlp.flavor = jrp.flavor = fr;
			    for(int scr=0;scr<12;scr++){
			      jlp.spin_color = jrp.spin_color = scr;
			      
			      VectorComplexType & into = Mr[scr + 12*(fr + nf*(x4d + vol4d*irblock_elem) )]; //store Mr in temp memory alloc
			      ACC::write(into, ScalarComplexType(0.));

			      size_t ir = tmpidx_to_ir_map[tmpir]; //the actual ir value
			      
			      //Do the j product-sum
			      const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
			      size_t nj = j_ind_pairs.size();
			      for(size_t j=0;j<nj;j++){
				size_t jl = j_ind_pairs[j].first,  jr = j_ind_pairs[j].second;

				ScalarComplexType rval_tmp = ACC::read(r.nativeElem(jr,x4d,scr,fr));
				ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
					    
				ScalarComplexType Mval = M(ir,jl);
				
				ScalarComplexType val = ACC::read(into) + Mval * rval;
				ACC::write(into, val);
			      }
			    }
			  }
			}
		      });

      //Step 2: Compute \sum_i v(il)_{scl,fl}(x) Mr(ir)_{scr,fr}(x)
      //As we only have ir in the block we can only perform part of the sum over i
      
      //Determine which i we can do
      std::vector<std::vector<std::vector<std::vector<std::pair<size_t,size_t> > > > > i_do; //12 * nf * Lt * (contractions in block). The pair gives (il, ir_block_elem)
      i_do.resize(12);
      for(int sc=0;sc<12;sc++){
	i_do[sc].resize(nf);
	for(int f=0;f<nf;f++)
	  i_do[sc][f].resize(Lt);
      }
      
      irp.time = M.getRowTimeslice();
      for(int tv=0;tv<Lt;tv++){
	ilp.time = tv;
	for(int f=0;f<nf;f++){
	  ilp.flavor = irp.flavor = f;
	  for(int sc=0;sc<12;sc++){
	    ilp.spin_color = irp.spin_color = sc;
	    
	    const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
	    size_t ni = i_ind_pairs.size();
	    
	    for(int i=0;i<ni;i++){
	      int ir = i_ind_pairs[i].second;
	      int tmpir = ir_to_tmpidx_map[ir];
	      assert(tmpir != -1);

	      if(tmpir >= tmpir_start && tmpir < tmpir_lessthan){
		int ir_block_elem = tmpir - tmpir_start; //index within stored data
		int il = i_ind_pairs[i].first;
		i_do[sc][f][tv].push_back(std::pair<size_t,size_t>(il,ir_block_elem));
	      }
	    }
	  }
	}
      }
      
      //Do the i contraction
      accelerator_for(x4d, vol4d, nsimd,
		      {
			VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d);

			size_t xop, top;
			into.fourToThree(xop, top, x4d);
			size_t t_glob = top + t_off;

			for(int fl=0;fl<nf;fl++){
			  for(int sl=0;sl<4;sl++){
			    for(int cl=0;cl<3;cl++){
			      int scl = cl+3*sl;
			      const std::vector<std::pair<size_t,size_t> > &i_do_fsc = i_do[scl][fl][t_glob];
			      
			      for(int fr=0;fr<nf;fr++){
				for(int sr=0;sr<4;sr++){
				  for(int cr=0;cr<3;cr++){
				    int scr = cr+3*sr;

				    VectorComplexType &out = vsite_mat(sl,sr)(cl,cr)(fl,fr);

				    for(int ii=0;ii<i_do_fsc.size();ii++){
				      int il = i_do_fsc[ii].first;
				      int irblock_elem = i_do_fsc[ii].second;
				    
				      const VectorComplexType & Mr_elem_v = Mr[scr + 12*(fr + nf*(x4d + vol4d*irblock_elem) )]; //store Mr in temp memory alloc
				      ScalarComplexType Mr_elem = ACC::read(Mr_elem_v);

				      ScalarComplexType lval_tmp = ACC::read(l.nativeElem(il,x4d,scl,fl));
				      ScalarComplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;

				      ScalarComplexType val = ACC::read(out) + lval * Mr_elem;
				      ACC::write(out, val);
				    }
				  }
				}
			      }
			    }
			  }
			}
		      });
    }//irblock

    managed_free(Mr);
  }//end of func



};



#endif //USE_GRID


CPS_END_NAMESPACE

#endif
