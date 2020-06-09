#ifndef _MULT_VMV_FIELD_OFFLOAD_H_
#define _MULT_VMV_FIELD_OFFLOAD_H_

#include<set>

CPS_START_NAMESPACE

std::ostream & operator<<(std::ostream &os, const std::pair<int,int> &p){
  os << '(' << p.first << "," << p.second << ')';
  return os;
}

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
  static accelerator_inline typename mf_Policies::ComplexType & access(const int s1, const int c1, const int f1,
							   const int s2, const int c2, const int f2,
							   VectorMatrixType &M){
    return M(s1,s2)(c1,c2)(f1,f2);
  }
    
};
template<typename mf_Policies>
struct _mult_vMv_field_offload_fields<mf_Policies,0>{
  typedef CPSspinMatrix<CPScolorMatrix<typename mf_Policies::ComplexType> > VectorMatrixType;
  typedef CPSfield<VectorMatrixType,1, FourDSIMDPolicy<OneFlavorPolicy>, Aligned128AllocPolicy> PropagatorField;
  static accelerator_inline typename mf_Policies::ComplexType & access(const int s1, const int c1, const int f1,
							   const int s2, const int c2, const int f2,
							   VectorMatrixType &M){
    return M(s1,s2)(c1,c2);
  }
};

struct mult_vMv_field_offload_timers{
  struct timers{
    double init1;
    double Mr;
    double init2;
    double v_Mr;
    size_t calls;

    timers(): init1(0), Mr(0), init2(0), v_Mr(0), calls(0){}

    void reset(){
      init1=Mr=init2=v_Mr=0;
      calls = 0;
    }
    void average(){
      init1/=calls;
      Mr/=calls;
      init2/=calls;
      v_Mr/=calls;
    }
    void print(){
      average();
      printf("calls=%zu init1=%g Mr=%g init2=%g v(Mr)=%g\n", calls, init1, Mr, init2, v_Mr);
    }

  };
  static timers & get(){ static timers t; return t; }
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
    mult_vMv_field_offload_timers::timers &time = mult_vMv_field_offload_timers::get();

    ++time.calls;

    time.init1 -= dclock();

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
    size_t Mr_size =  BlockedvMvOffloadArgs::b * 12 * nf * vol4d * sizeof(VectorComplexType);
    VectorComplexType* Mr = (VectorComplexType*)managed_alloc_check(Mr_size); //block in ir. Make this device memory!

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

    //Get the maximum number of j indices that need to be summed over
    size_t j_max = 0;
    for(int t=0;t<GJP.TnodeSites();t++){
      modeIndexSet jlp, jrp;
      jlp.time = M.getNcols();
      jrp.time = t + t_off;
      for(int fr=0;fr<nf;fr++){
	jlp.flavor = jrp.flavor = fr;
	for(int scr=0;scr<12;scr++){
	  jlp.spin_color = jrp.spin_color = scr;
	  j_max = std::max(j_max, j_ind.getIndexVector(jlp,jrp).size());
	}
      }
    }


    time.init1 += dclock();

    //Block over the subset of ir and perform the M*v multiplication for each site, spin/color and flavor, then do the v*(Mv) product still within the block
    size_t nir_needed = xx;
    size_t nir_blocks = (xx + blocksize-1)/blocksize;
    for(size_t irblock=0; irblock<nir_blocks; irblock++){
      size_t tmpir_start = irblock * blocksize; //start of block in tmpir index
      size_t tmpir_lessthan = std::min( tmpir_start + blocksize,  nir_needed );

      time.Mr -= dclock();

      //Step 1: Compute \sum_j  M(ir, jl) * v(jr)_{sc,f}(x)     for ir in block
      accelerator_for(x4d, vol4d, nsimd,
		      {
			size_t xop, top;
			into.fourToThree(xop, top, x4d);
			size_t t_glob = top + t_off;

			modeIndexSet jlp, jrp;
			jlp.time = M.getNcols();
			jrp.time = t_glob;

			//Get the maximum number of j indices that need to be summed over
			size_t njblocks = (j_max + blocksize - 1)/blocksize;
			for(size_t jblock=0;jblock<njblocks;jblock++){
			  size_t j_start = jblock * blocksize;
			  size_t j_lessthan = std::min(j_start + blocksize, j_max);
			  
			  //Loop over ir, scr, fr,  and sum over j
			  for(size_t tmpir = tmpir_start; tmpir < tmpir_lessthan; tmpir++){ //the element of the current block
			    size_t irblock_elem = tmpir - tmpir_start;
			    size_t ir = tmpidx_to_ir_map[tmpir]; //the actual ir value
			    
			    for(int fr=0;fr<nf;fr++){
			      jlp.flavor = jrp.flavor = fr;
			      for(int scr=0;scr<12;scr++){
				jlp.spin_color = jrp.spin_color = scr;
				
				VectorComplexType *into = Mr + scr + 12*(fr + nf*(x4d + vol4d*irblock_elem)); //store Mr in temp memory alloc
				
				//Do the j product-sum
				//foreach(tmpir in irblock, scr, fr)   \sum_{j in jblock} M(ir, jl) * v(jr)_{scr,fr}(x)   reuse v(jr)_{scr,fr}(x)  ~irblock times
				const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
				
				ScalarComplexType val_sum = jblock == 0 ? ScalarComplexType(0) : ACC::read(*into);
				
				for(size_t j=j_start;j<j_lessthan;j++){				  
				  if(j >= j_ind_pairs.size()) continue;
				  
				  size_t jl = j_ind_pairs[j].first,  jr = j_ind_pairs[j].second;
				  
				  ScalarComplexType rval_tmp = ACC::read(r.nativeElem(jr,x4d,scr,fr));
				  ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
				  
				  ScalarComplexType Mval = M(ir,jl);
				  
				  val_sum = val_sum + Mval * rval;
				}
				ACC::write(*into, val_sum);
			      }//scr
			    }//fr
			  }//ir
			}//jblock
		      });

      time.Mr += dclock();
      time.init2 -= dclock();

      //Step 2: Compute \sum_i v(il)_{scl,fl}(x) Mr(ir)_{scr,fr}(x)
      //As we only have ir in the block we can only perform part of the sum over i
      
      //Determine which i we can do
      std::vector<std::vector<std::vector<std::vector<std::pair<size_t,size_t> > > > > i_do; //12 * nf * Lt * (contractions in block). The pair gives (il, ir_block_elem)
      i_do.resize(12);
      for(int sc=0;sc<12;sc++){
	i_do[sc].resize(nf);
	for(int f=0;f<nf;f++)
	  i_do[sc][f].resize(Lt);
	//Maybe reserve memory here to speed up
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

      time.init2 += dclock();
      time.v_Mr -= dclock();
      
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

      time.v_Mr += dclock();

    }//irblock

    managed_free(Mr);

  }//end of func



  template<typename T>
  static accelerator_inline T min_value(const T&a, const T&b){ return a < b ? a : b; }
  

  static void v5(PropagatorField &into,
  		   const lA2AfieldType &l,
  		   const MesonFieldType &M,
  		   const rA2AfieldType &r,
  		   bool conj_l, bool conj_r){
    mult_vMv_field_offload_timers::timers &time = mult_vMv_field_offload_timers::get();

    ++time.calls;

    time.init1 -= dclock();

    into.zero();

    ModeContractionIndices<iLeftDilutionType,iRightDilutionType> i_ind(l);
    ModeContractionIndices<jLeftDilutionType,jRightDilutionType> j_ind(r);

    assert(into.nodeSites(3) == GJP.TnodeSites()); //cannot be SIMD packed in t-direction
    int Lt = GJP.Tnodes() * GJP.TnodeSites();
    int nf = GJP.Gparity() + 1;
    int nsimd = VectorComplexType::Nsimd();
    size_t vol4d = into.size();
    int t_off = GJP.TnodeSites() * GJP.TnodeCoor();
    size_t blocksize = BlockedvMvOffloadArgs::b;
    size_t inner_blocksize = BlockedvMvOffloadArgs::bb;

    typedef SIMT<VectorComplexType> ACC;
    typedef typename ACC::value_type value_type;

    //Need to compute \sum_i\sum_j v(il)_{scl,fl}(x)  M(ir, jl) * v(jr)_{scr,fr}(x)
    
    //Transform into   \sum_i' \sum_j'  v'(i')_{scl, fl}(x) M'(i',j') v'(j')_{scr, fr}(x)  alpha(scl,fl,t,i')   beta(scr,fr,t,j')
    //i' and j' run over the full set of allowed index pairs
    //alpha, beta are boolean masks that zero out pairs that are not valid for a particular sc,f,t
    //v'(i')_{scl, fl}(x) = v(il[i'])_{scl, fl}(x)
    //M'(i',j') = M(ir[i'],jl[j'])


    //Then define
    //va'(i')_{scl, fl}(x) =  alpha(scl,fl,t,i')  v'(i')_{scl, fl}(x)
    //vb'(j')_{scr, fr}(x) =  beta(scr,fr,t,j')  v'(j')_{scr, fr}(x)
    //Such that
    //\sum_i' \sum_j'  va'(i')_{scl, fl}(x) M'(i',j') vb'(j')_{scr, fr}(x)  


    std::set<std::pair<int,int> > il_ir_pairs_s;
    std::set<std::pair<int,int> > jl_jr_pairs_s;

    {
      modeIndexSet ilp, irp, jlp, jrp;
      irp.time = M.getRowTimeslice();
      jlp.time = M.getColTimeslice();
      for(int tv=0;tv<Lt;tv++){
	ilp.time = jrp.time = tv;
	for(int f=0;f<nf;f++){
	  ilp.flavor = irp.flavor = jlp.flavor = jrp.flavor = f;
	  for(int sc=0;sc<12;sc++){
	    ilp.spin_color = irp.spin_color = jlp.spin_color = jrp.spin_color = sc;
	
	    const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
	    for(int i=0;i<i_ind_pairs.size();i++) il_ir_pairs_s.insert(i_ind_pairs[i]);	    

	    const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
	    for(int j=0;j<j_ind_pairs.size();j++) jl_jr_pairs_s.insert(j_ind_pairs[j]);
	  }
	}
      }
    }
    
    ManagedVector< std::pair<int,int> > il_ir_pairs(il_ir_pairs_s.size());
    std::map<std::pair<int,int>, int> il_ir_pairs_index_map;
    int ii=0;
    for(auto it=il_ir_pairs_s.begin(); it != il_ir_pairs_s.end(); it++){
      il_ir_pairs[ii] = *it;
      il_ir_pairs_index_map[*it] = ii;
      ++ii;
    }
    int nil_ir_pairs = il_ir_pairs.size();

    ManagedVector< std::pair<int,int> > jl_jr_pairs(jl_jr_pairs_s.size());
    std::map<std::pair<int,int>, int> jl_jr_pairs_index_map;
    ii=0;
    for(auto it=jl_jr_pairs_s.begin(); it != jl_jr_pairs_s.end(); it++){
      jl_jr_pairs[ii] = *it;
      jl_jr_pairs_index_map[*it] = ii;
      ++ii;
    }
    int njl_jr_pairs = jl_jr_pairs.size();

    //Construct the masks
    ManagedVector<uint8_t> alpha(12*nf*Lt*nil_ir_pairs,0);
    ManagedVector<uint8_t> beta(12*nf*Lt*njl_jr_pairs,0);
    
    {
      modeIndexSet ilp, irp, jlp, jrp;
      irp.time = M.getRowTimeslice();
      jlp.time = M.getColTimeslice();
      for(int tv=0;tv<Lt;tv++){
	ilp.time = jrp.time = tv;
	for(int f=0;f<nf;f++){
	  ilp.flavor = irp.flavor = jlp.flavor = jrp.flavor = f;
	  for(int sc=0;sc<12;sc++){
	    ilp.spin_color = irp.spin_color = jlp.spin_color = jrp.spin_color = sc;

	    const ModeMapType &i_ind_pairs = i_ind.getIndexVector(ilp,irp);
	    for(int i=0;i<i_ind_pairs.size();i++){
	      const std::pair<int, int> &pair = i_ind_pairs[i];
	      int pair_idx = il_ir_pairs_index_map[pair];
	      alpha[sc + 12*(f+ nf*(tv + Lt*pair_idx))] = 1;
	    }
	    const ModeMapType &j_ind_pairs = j_ind.getIndexVector(jlp,jrp);
	    for(int j=0;j<j_ind_pairs.size();j++){
	      const std::pair<int, int> &pair = j_ind_pairs[j];
	      int pair_idx = jl_jr_pairs_index_map[pair];
	      beta[sc + 12*(f+ nf*(tv + Lt*pair_idx))] = 1;
	    }
	  }
	}
      }
    }

    size_t niprime = nil_ir_pairs;
    size_t njprime = njl_jr_pairs;

    size_t field_size = 12 * nf * vol4d;
    size_t vaprime_bytes = field_size * blocksize * sizeof(VectorComplexType);
    size_t vbprime_bytes = field_size * blocksize * sizeof(VectorComplexType);
    size_t Mprime_bytes = blocksize * blocksize * sizeof(VectorComplexType);

    VectorComplexType* vaprime = (VectorComplexType*)managed_alloc_check(vaprime_bytes);
    VectorComplexType* vbprime = (VectorComplexType*)managed_alloc_check(vbprime_bytes);
    VectorComplexType* Mprime = (VectorComplexType*)managed_alloc_check(Mprime_bytes);
    VectorComplexType* Mvbprime = (VectorComplexType*)managed_alloc_check(vaprime_bytes);

    std::cout << "Outer block size " << blocksize << " inner blocksize " << inner_blocksize << std::endl;
    std::cout << "vaprime " << double(vaprime_bytes)/1024./1024. << " MB" << std::endl;
    std::cout << "vbprime " << double(vbprime_bytes)/1024./1024. << " MB" << std::endl;
    std::cout << "Mprime " << double(Mprime_bytes)/1024./1024. << " MB" << std::endl;
    std::cout << "Mvbprime " << double(vaprime_bytes)/1024./1024. << " MB" << std::endl;

    //Do in blocks over i',j' to avoid taking too much space
    size_t niprime_blocks = (niprime + blocksize-1)/blocksize;
    size_t njprime_blocks = (njprime + blocksize-1)/blocksize;

    time.init1 += dclock();
    
    for(size_t iprimeblock =0; iprimeblock < niprime_blocks; iprimeblock++){

      time.init2 -= dclock();
      size_t iprimestart = iprimeblock * blocksize;
      size_t iprimelessthan = std::min(iprimestart + blocksize, niprime);
      size_t niprime_block = iprimelessthan - iprimestart;

      //std::cout << "iprimeblock:" << iprimeblock << " iprimestart:" << iprimestart << " iprimelessthan:" << iprimelessthan << " niprime_block:"<< niprime_block << std::endl;
      //std::cout << "Create va'" << std::endl;

      //Create va'
      {	
	copyControl::shallow() = true;
	accelerator_for(x4d, vol4d, nsimd,
			{
			  size_t xop; int top;
			  into.fourToThree(xop, top, x4d);
			  int t_glob = top + t_off;
			  for(size_t iprime = iprimestart; iprime < iprimelessthan; iprime++){
			    size_t iprimeb = iprime - iprimestart;
			    for(int f=0;f<nf;f++){
			      for(int sc=0;sc<12;sc++){
				VectorComplexType *into = vaprime +  iprimeb + niprime_block*( sc + 12*(f + nf*x4d) ); //contiguous in summed index 
				value_type val = ACC::read(l.nativeElem(il_ir_pairs[iprime].first, x4d, sc, f));
				val = conj_l ? Grid::conjugate(val) : val;
				val = val * double(alpha[sc + 12*(f+ nf*(t_glob + Lt*iprime))]);
				ACC::write(*into, val);
			      }
			    }
			  }
			});
	time.init2 += dclock();
	copyControl::shallow() = false;
      }

      for(size_t jprimeblock =0; jprimeblock < njprime_blocks; jprimeblock++){
	time.init2 -= dclock();
	size_t jprimestart = jprimeblock * blocksize;
	size_t jprimelessthan = std::min(jprimestart + blocksize, njprime);
	size_t njprime_block = jprimelessthan - jprimestart;	

	//std::cout << "jprimeblock:" << jprimeblock << " jprimestart:" << jprimestart << " jprimelessthan:" << jprimelessthan << " njprime_block:"<< njprime_block << std::endl;
	//std::cout << "Create vb'" << std::endl;


	//Create vb'
	{
	  copyControl::shallow() = true;
	  accelerator_for(x4d, vol4d, nsimd,
			  {
			    size_t xop; int top;
			    into.fourToThree(xop, top, x4d);
			    int t_glob = top + t_off;
			    for(size_t jprime = jprimestart; jprime < jprimelessthan; jprime++){
			      size_t jprimeb = jprime - jprimestart;
			      for(int f=0;f<nf;f++){
				for(int sc=0;sc<12;sc++){
				  VectorComplexType *into = vbprime + jprimeb + njprime_block*(sc + 12*(f + nf*x4d) );  //contiguous in summed index
				  value_type val = ACC::read(r.nativeElem(jl_jr_pairs[jprime].second, x4d, sc, f));
				  val = conj_r ? Grid::conjugate(val) : val;
				  val = val * double(beta[sc + 12*(f+ nf*(t_glob + Lt*jprime))]);
				  ACC::write(*into, val);
				}
			      }
			    }
			  });
	  copyControl::shallow() = false;
	}

	//std::cout << "Create Mprime" << std::endl;
	//Create Mprime
	{
	  VectorComplexType *Mptr = Mprime;
	  for(size_t iprime = iprimestart; iprime < iprimelessthan; iprime++){
	    size_t iprimeb = iprime - iprimestart;
	    size_t ir = il_ir_pairs[iprime].second;
	    for(size_t jprime = jprimestart; jprime < jprimelessthan; jprime++){
	      size_t jprimeb = jprime - jprimestart;
	      size_t jl = jl_jr_pairs[jprime].first;
	      Grid::vsplat(*Mptr++, M(ir, jl));
	    }
	  }
	}
    
	time.init2 += dclock();


	//std::cout << "M' * vb'" << std::endl;
	//Mprime * vbprime
	time.Mr -= dclock();
	memset(Mvbprime, 0, vaprime_bytes);
	
	copyControl::shallow() = true;
	{
	  accelerator_for(x4d, vol4d, nsimd,
			  {
			    size_t niprimeb_subblocks = (niprime_block + inner_blocksize - 1)/inner_blocksize;
			    for(int iprimeb_subblock=0; iprimeb_subblock < niprimeb_subblocks; iprimeb_subblock++){
			      size_t iprimeb_start = iprimeb_subblock * inner_blocksize;
			      size_t iprimeb_lessthan = min_value( iprimeb_start + inner_blocksize, niprime_block );
			      
			      size_t njprimeb_subblocks = (njprime_block + inner_blocksize - 1)/inner_blocksize;
			      for(int jprimeb_subblock=0; jprimeb_subblock < njprimeb_subblocks; jprimeb_subblock++){
				size_t jprimeb_start = jprimeb_subblock * inner_blocksize;
				size_t jprimeb_lessthan = min_value( jprimeb_start + inner_blocksize, njprime_block );
				
				for(int iprimeb=iprimeb_start;iprimeb<iprimeb_lessthan; iprimeb++){
				  for(int fr=0;fr<nf;fr++){
				    for(int scr=0;scr<12;scr++){
				      VectorComplexType *into = Mvbprime + iprimeb + niprime_block*(scr + 12*(fr + nf*x4d) );
				      value_type sum = ACC::read(*into);
				      VectorComplexType *rptr = vbprime + jprimeb_start + njprime_block*(scr + 12*(fr + nf*x4d) );
				      VectorComplexType *Mptr = Mprime + jprimeb_start + njprime_block * iprimeb;
				
				      for(int jprimeb=jprimeb_start;jprimeb<jprimeb_lessthan; jprimeb++){
					value_type rval = ACC::read(*rptr++);
					ScalarComplexType Mval = ACC::read(*Mptr++);
					sum = sum + Mval * rval;
				      }
				      ACC::write(*into, sum);
				    }
				  }
				}
			      }
			    }
			  });
	}
	copyControl::shallow() = false;
	
	time.Mr += dclock();

	//std::cout << "va' (M' vb')" << std::endl;
	time.v_Mr -= dclock();
	{
	  copyControl::shallow() = true;
	  accelerator_for(x4d, vol4d, nsimd,
			  {
			    VectorMatrixType &vsite_mat = *into.fsite_ptr(x4d);
			    size_t niprimeb_subblocks = (niprime_block + inner_blocksize - 1)/inner_blocksize;
			    for(int iprimeb_subblock=0; iprimeb_subblock < niprimeb_subblocks; iprimeb_subblock++){
			      size_t iprimeb_start = iprimeb_subblock * inner_blocksize;
			      size_t iprimeb_lessthan = min_value( iprimeb_start + inner_blocksize, niprime_block );

			      for(int fl=0;fl<nf;fl++){
			    	for(int sl=0;sl<4;sl++){
			    	  for(int cl=0;cl<3;cl++){
			    	    int scl = cl+3*sl;
				  
			    	    for(int fr=0;fr<nf;fr++){
			    	      for(int sr=0;sr<4;sr++){
			    		for(int cr=0;cr<3;cr++){
			    		  int scr = cr+3*sr;
				      
			    		  VectorComplexType &out = fdef::access(sl,cl,fl, sr,cr,fr, vsite_mat);
			    		  value_type sum = ACC::read(out);

			    		  VectorComplexType *lptr = vaprime + iprimeb_start + niprime_block*(scl + 12*(fl + nf*x4d) );
			    		  VectorComplexType *Mrptr = Mvbprime + iprimeb_start + niprime_block*(scr + 12*(fr + nf*x4d) );

			    		  for(int iprimeb=iprimeb_start;iprimeb<iprimeb_lessthan; iprimeb++){
			    		    value_type lval = ACC::read(*lptr++); 
			    		    value_type Mrval = ACC::read(*Mrptr++); 
			    		    sum = sum + lval * Mrval;
			    		  }
			    		  ACC::write(out, sum);
			    		}
			    	      }
			    	    }
			    	  }
			    	}
			      }
			    }
			  });
	  copyControl::shallow() = false;
	}

	time.v_Mr += dclock();
      }
    }

    managed_free(vaprime);
    managed_free(vbprime);
    managed_free(Mprime);
    managed_free(Mvbprime);
    
  }//end of func
    



};



#endif //USE_GRID


CPS_END_NAMESPACE

#endif
