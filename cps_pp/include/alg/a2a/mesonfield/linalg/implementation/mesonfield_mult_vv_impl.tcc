#ifndef _MULT_VV_IMPL_H
#define _MULT_VV_IMPL_H

#include "mesonfield_mult_vMv_common.tcc"

template<typename mf_Policies, 
	 template <typename> class lA2Afield,  template <typename> class rA2Afield,
	 typename ComplexClass
	 >
class _mult_lr_impl_v{ };


template<typename mf_Policies, 
	 template <typename> class lA2Afield,  template <typename> class rA2Afield
	 >
class _mult_lr_impl_v<mf_Policies,lA2Afield,rA2Afield,complex_double_or_float_mark>{ 
public:
  typedef typename _mult_vMv_impl_v_getPolicy<mf_Policies::GPARITY>::type OutputPolicy;
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename OutputPolicy::template MatrixType<ScalarComplexType> OutputMatrixType;
  const static int nf = OutputPolicy::nf();

  static void mult(OutputMatrixType &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
		   const int xop, const int top, const bool conj_l, const bool conj_r){
    typedef typename lA2Afield<mf_Policies>::DilutionType iLeftDilutionType;
    typedef typename rA2Afield<mf_Policies>::DilutionType iRightDilutionType;

    out.zero();

    int top_glb = top+GJP.TnodeSites()*GJP.TnodeCoor();

    const StandardIndexDilution &std_idx = l;
    int nv = std_idx.getNmodes();

    //Precompute index mappings
    modeIndexSet ilp, irp;
    ilp.time = top_glb;
    irp.time = top_glb;
    
    //We want to treat this as a matrix mult of a 24 * nv  matrix with an nv * 24 matrix, where nv is the number of fully undiluted indices. First implementation uses regular GSL, but could perhaps be done better with sparse matrices
    int site4dop = l.getMode(0).threeToFour(xop,top);
    const static int nscf = nf*3*4;

    //Pull out the components we need into packed GSL vectors
    typedef gsl_wrapper<typename ScalarComplexType::value_type> gw;

    typename gw::matrix_complex *lgsl = gw::matrix_complex_alloc(nscf, nv);
    typename gw::matrix_complex *rgsl = gw::matrix_complex_alloc(nv,nscf);
    gw::matrix_complex_set_zero(lgsl);
    gw::matrix_complex_set_zero(rgsl);

    assert(sizeof(typename gw::complex) == sizeof(ScalarComplexType) ); 

    for(int s=0;s<4;s++){
      for(int c=0;c<3;c++){
    	ilp.spin_color = irp.spin_color = c + 3*s;
    	for(int f=0;f<nf;f++){
    	  ilp.flavor = irp.flavor = f;
    	  int scf = f+nf*(c+3*s);
	  
	  std::vector<int> lmap;
	  std::vector<bool> lnon_zeroes;
	  l.getIndexMapping(lmap,lnon_zeroes,ilp);

	  for(int i=0;i<nv;i++) //Could probably speed this up using the contiguous block finder and memcpy
	    if(lnon_zeroes[i]){
	      const ScalarComplexType &lval_tmp = l.nativeElem(lmap[i], site4dop, ilp.spin_color, f);
	      gw::matrix_complex_set(lgsl, scf, i, reinterpret_cast<typename gw::complex const&>(lval_tmp));
	    }
	  
	  std::vector<int> rmap;
	  std::vector<bool> rnon_zeroes;
	  r.getIndexMapping(rmap,rnon_zeroes,irp);

	  for(int i=0;i<nv;i++)
	    if(rnon_zeroes[i]){
	      const ScalarComplexType &rval_tmp = r.nativeElem(rmap[i], site4dop, irp.spin_color, f);
	      gw::matrix_complex_set(rgsl, i, scf, reinterpret_cast<typename gw::complex const&>(rval_tmp));
	    }
	}
      }
    }

    if(conj_l)
      for(int i=1;i<nscf*nv*2;i+=2)
    	lgsl->data[i] = -lgsl->data[i];
    if(conj_r)
      for(int i=1;i<nscf*nv*2;i+=2)
    	rgsl->data[i] = -rgsl->data[i];  



    typename gw::matrix_complex *ogsl = gw::matrix_complex_alloc(nscf, nscf);
    typename gw::complex one; GSL_SET_COMPLEX(&one,1.0,0.0);
    typename gw::complex zero; GSL_SET_COMPLEX(&zero,0.0,0.0);

#ifndef MEMTEST_MODE
    gw::blas_gemm(CblasNoTrans, CblasNoTrans, one, lgsl, rgsl, zero, ogsl);
#endif
    
    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<nf;fl++){
	      int scfl = fl+nf*(cl+3*sl);
    	      for(int fr=0;fr<nf;fr++){
		int scfr = fr+nf*(cr+3*sr);
		typename ScalarComplexType::value_type (&ol)[2] = reinterpret_cast<typename ScalarComplexType::value_type(&)[2]>(OutputPolicy::acc(sl,sr,cl,cr,fl,fr,out));
		typename gw::complex* gg = gw::matrix_complex_ptr(ogsl, scfl, scfr);
		ol[0] = gg->dat[0];
		ol[1] = gg->dat[1];
	      }
	    }
	  }
	}
      }
    }

    gw::matrix_complex_free(lgsl);
    gw::matrix_complex_free(rgsl);
    gw::matrix_complex_free(ogsl);
  }

  static void mult_slow(OutputMatrixType &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
			const int xop, const int top, const bool conj_l, const bool conj_r){
    assert( l.paramsEqual(r) );

    int site4dop = xop + GJP.VolNodeSites()/GJP.TnodeSites()*top;

    A2Aparams i_params(l);
    StandardIndexDilution idil(i_params);
    
    int ni = idil.getNmodes();

    out.zero();

    for(int sl=0;sl<4;sl++){
      for(int cl=0;cl<3;cl++){
    	for(int fl=0;fl<nf;fl++){	  

	  for(int sr=0;sr<4;sr++){
	    for(int cr=0;cr<3;cr++){
	      for(int fr=0;fr<nf;fr++){

		for(int i=0;i<ni;i++){

		  const ScalarComplexType &lval_tmp = l.elem(i,xop,top,cl+3*sl,fl);
		  ScalarComplexType lval = conj_l ? std::conj(lval_tmp) : lval_tmp;
		  
		  const ScalarComplexType &rval_tmp = r.elem(i,xop,top,cr+3*sr,fr);
		  ScalarComplexType rval = conj_r ? std::conj(rval_tmp) : rval_tmp;

		  OutputPolicy::acc(sl,sr,cl,cr,fl,fr,out) += lval * rval;
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
};

#ifdef USE_GRID


template<typename mf_Policies, 
	 template <typename> class lA2Afield,  template <typename> class rA2Afield
	 >
class _mult_lr_impl_v<mf_Policies,lA2Afield,rA2Afield,grid_vector_complex_mark>{ 
public:
  typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
  typedef typename mf_Policies::ComplexType SIMDcomplexType;

  typedef typename AlignedVector<SIMDcomplexType>::type AlignedSIMDcomplexVector;

  typedef typename _mult_vMv_impl_v_getPolicy<mf_Policies::GPARITY>::type OutputPolicy;
  typedef typename OutputPolicy::template MatrixType<SIMDcomplexType> OutputMatrixType;
  const static int nf = OutputPolicy::nf();
 

  static void mult(OutputMatrixType &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
		   const int xop, const int top, const bool conj_l, const bool conj_r){
    typedef typename lA2Afield<mf_Policies>::DilutionType iLeftDilutionType;
    typedef typename rA2Afield<mf_Policies>::DilutionType iRightDilutionType;

    out.zero();
    assert(l.getMode(0).SIMDlogicalNodes(3) == 1);

    int top_glb = top+GJP.TnodeSites()*GJP.TnodeCoor();

    //Precompute index mappings
    modeIndexSet ilp, irp;
    ilp.time = top_glb;
    irp.time = top_glb;

    int site4dop = l.getMode(0).threeToFour(xop, top);

    const StandardIndexDilution &std_idx = l;
    int nv = std_idx.getNmodes();
    const static int nscf = nf*3*4;

    std::vector<AlignedSIMDcomplexVector> lcp(nscf, AlignedSIMDcomplexVector(nv) );    
    std::vector<AlignedSIMDcomplexVector> rcp(nv, AlignedSIMDcomplexVector(nscf) );    

    std::vector<std::vector<bool> > lnon_zeroes_all(nscf);
    std::vector<std::vector<bool> > rnon_zeroes_all(nscf);
    
    for(int s=0;s<4;s++){
      for(int c=0;c<3;c++){
    	ilp.spin_color = irp.spin_color = c + 3*s;
    	for(int f=0;f<nf;f++){
    	  ilp.flavor = irp.flavor = f;
    	  int scf = f+nf*(c+3*s);
	  
	  std::vector<int> lmap;
	  std::vector<bool> &lnon_zeroes = lnon_zeroes_all[scf];
	  l.getIndexMapping(lmap,lnon_zeroes,ilp);

	  for(int i=0;i<nv;i++) 
	    if(lnon_zeroes[i]){
#ifndef MEMTEST_MODE
	      const SIMDcomplexType &lval_tmp = l.nativeElem(lmap[i], site4dop, ilp.spin_color, f);
	      lcp[scf][i] = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
#endif
	    }
	  
	  std::vector<int> rmap;
	  std::vector<bool> &rnon_zeroes = rnon_zeroes_all[scf];
	  r.getIndexMapping(rmap,rnon_zeroes,irp);

	  for(int i=0;i<nv;i++)
	    if(rnon_zeroes[i]){
#ifndef MEMTEST_MODE
	      const SIMDcomplexType &rval_tmp = r.nativeElem(rmap[i], site4dop, irp.spin_color, f);
	      rcp[i][scf] = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;
#endif
	    }	  
	}
      }
    }

    for(int sl=0;sl<4;sl++){
      for(int sr=0;sr<4;sr++){
	for(int cl=0;cl<3;cl++){
	  for(int cr=0;cr<3;cr++){
	    for(int fl=0;fl<nf;fl++){
	      for(int fr=0;fr<nf;fr++){
		int scfl = fl+nf*(cl+3*sl);
		int scfr = fr+nf*(cr+3*sr);
#ifndef MEMTEST_MODE
		for(int v=0;v<nv;v++)
		  if(lnon_zeroes_all[scfl][v] && rnon_zeroes_all[scfr][v])
		    OutputPolicy::acc(sl,sr,cl,cr,fl,fr,out) = OutputPolicy::acc(sl,sr,cl,cr,fl,fr,out) + lcp[scfl][v]*rcp[v][scfr];
#endif
	      }
	    }
	  }
	}
      }
    }
  }

  static void mult_slow(OutputMatrixType &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
			const int xop, const int top, const bool conj_l, const bool conj_r){
    assert( l.paramsEqual(r) );

    int site4dop = xop + GJP.VolNodeSites()/GJP.TnodeSites()*top;

    A2Aparams i_params(l);
    StandardIndexDilution idil(i_params);
    
    int ni = idil.getNmodes();

    out.zero();

    for(int sl=0;sl<4;sl++){
      for(int cl=0;cl<3;cl++){
    	for(int fl=0;fl<nf;fl++){	  

	  for(int sr=0;sr<4;sr++){
	    for(int cr=0;cr<3;cr++){
	      for(int fr=0;fr<nf;fr++){

		for(int i=0;i<ni;i++){

		  const SIMDcomplexType &lval_tmp = l.elem(i,xop,top,cl+3*sl,fl);
		  SIMDcomplexType lval = conj_l ? Grid::conjugate(lval_tmp) : lval_tmp;
		  
		  const SIMDcomplexType &rval_tmp = r.elem(i,xop,top,cr+3*sr,fr);
		  ScalarComplexType rval = conj_r ? Grid::conjugate(rval_tmp) : rval_tmp;

		  OutputPolicy::acc(sl,sr,cl,cr,fl,fr,out) = OutputPolicy::acc(sl,sr,cl,cr,fl,fr,out) + lval * rval;
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
};

#endif

// l^i(xop,top) r^i(xop,top)
//argument xop is the *local* 3d site index in canonical ordering, top is the *local* time coordinate
// Node local and unthreaded
template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class rA2Afield  
	 >
void mult(CPSspinColorFlavorMatrix<typename mf_Policies::ComplexType> &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
	  const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_lr_impl_v<mf_Policies,lA2Afield,rA2Afield,typename ComplexClassify<typename mf_Policies::ComplexType>::type >::mult(out,l,r,xop,top,conj_l,conj_r);
}

template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class rA2Afield  
	 >
void mult(CPSspinColorMatrix<typename mf_Policies::ComplexType> &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
	  const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_lr_impl_v<mf_Policies,lA2Afield,rA2Afield,typename ComplexClassify<typename mf_Policies::ComplexType>::type >::mult(out,l,r,xop,top,conj_l,conj_r);
}




template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class rA2Afield  
	 >
void mult_slow(CPSspinColorFlavorMatrix<typename mf_Policies::ComplexType> &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
	       const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_lr_impl_v<mf_Policies,lA2Afield,rA2Afield,typename ComplexClassify<typename mf_Policies::ComplexType>::type >::mult_slow(out,l,r,xop,top,conj_l,conj_r);
}

template<typename mf_Policies, 
	 template <typename> class lA2Afield,  
	 template <typename> class rA2Afield  
	 >
void mult_slow(CPSspinColorMatrix<typename mf_Policies::ComplexType> &out, const lA2Afield<mf_Policies> &l, const rA2Afield<mf_Policies> &r, 
	       const int xop, const int top, const bool conj_l, const bool conj_r){
  _mult_lr_impl_v<mf_Policies,lA2Afield,rA2Afield,typename ComplexClassify<typename mf_Policies::ComplexType>::type >::mult_slow(out,l,r,xop,top,conj_l,conj_r);
}




#endif
