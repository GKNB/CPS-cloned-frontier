//Convenience functions for computing multiple meson fields with an array of sources and/or quark momenta

#ifndef _MESONFIELD_COMPUTE_MANY_H
#define _MESONFIELD_COMPUTE_MANY_H

#include<alg/a2a/mesonfield_computemany_storagetypes.h>
CPS_START_NAMESPACE


template<typename mf_Policies, typename StorageType>
class ComputeMesonFields{  
 public:
#ifdef USE_DESTRUCTIVE_FFT
  typedef const std::vector< A2AvectorW<mf_Policies>*> WspeciesVector;
  typedef const std::vector< A2AvectorV<mf_Policies>*> VspeciesVector;
#else
  typedef const std::vector< A2AvectorW<mf_Policies> const*> WspeciesVector;
  typedef const std::vector< A2AvectorV<mf_Policies> const*> VspeciesVector;
#endif
  
  //W and V are indexed by the quark type index
  static void compute(StorageType &into, WspeciesVector &W, VspeciesVector &V,  Lattice &lattice, const bool node_distribute = false){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::SourcePolicies SourcePolicies;
    typedef typename mf_Policies::FermionFieldType::InputParamType VWfieldInputParams;
    int Lt = GJP.Tnodes()*GJP.TnodeSites();

    assert(W.size() == V.size());
    const int nspecies = W.size();

    const int nbase = GJP.Gparity() ? 2 : 1; //number of base FFTs
    const int nbase_max = 2;
    int p_0[3] = {0,0,0};
    int p_p1[3], p_m1[3];
    GparityBaseMomentum(p_p1,+1);
    GparityBaseMomentum(p_m1,-1);

    ThreeMomentum pbase[nbase];
    if(GJP.Gparity()){ pbase[0] = ThreeMomentum(p_p1); pbase[1] = ThreeMomentum(p_m1); }
    else pbase[0] = ThreeMomentum(p_0);
    
    printMem("ComputeMesonFields::compute Memory prior to compute loop");

    //We need to group the V shifts by base and species
    std::vector< std::vector< std::vector< std::vector< std::vector<int> > > > > base_cidx_map(nbase); //[base_v][qidx_v][base_w][qidx_w]
    for(int bv=0;bv<nbase;bv++){
      base_cidx_map[bv].resize(nspecies);
      for(int sv=0;sv<nspecies;sv++){
	base_cidx_map[bv][sv].resize(nbase);
	for(int bw=0;bw<nbase;bw++){
	  base_cidx_map[bv][sv][bw].resize(nspecies, std::vector<int>(0));
	}
      }
    }
    
    for(int c=0;c<into.nCompute();c++){
      int qidx_w, qidx_v;
      ThreeMomentum p_w, p_v;      
      into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,c);
      ThreeMomentum p_w_base, p_v_base;   int a[3];
      MesonFieldStorageBase::getGPmomParams(a,p_w_base.ptr(), p_w.ptr());
      MesonFieldStorageBase::getGPmomParams(a,p_v_base.ptr(), p_v.ptr());
	
      int base_v = -1;
      int base_w = -1;
      
      for(int b=0;b<nbase;b++){
	if(p_v_base == pbase[b]) base_v = b; 
	if(p_w_base == pbase[b]) base_w = b;
      }

      if(base_v == -1) ERR.General("ComputeMesonFields","compute","Supposed base V momentum %s for species %d of computation %d is not in the set of base momenta\n",p_v_base.str().c_str(),qidx_v,c);
      if(base_w == -1) ERR.General("ComputeMesonFields","compute","Supposed base W momentum %s for species %d of computation %d is not in the set of base momenta\n",p_w_base.str().c_str(),qidx_w,c);

      base_cidx_map[base_v][qidx_v][base_w][qidx_w].push_back(c);
    }

    typedef typename A2AvectorVfftw<mf_Policies>::FieldInputParamType Field4DInputParamTypeV;
    typedef typename A2AvectorWfftw<mf_Policies>::FieldInputParamType Field4DInputParamTypeW;
    
    //Do the computations with an outer loop over the base momentum index and species of the Vfftw
    for(int bv=0;bv<nbase;bv++){
      const ThreeMomentum &pvb = pbase[bv];      
      for(int sv=0;sv<nspecies;sv++){	
	int count = 0;
	for(int bw=0;bw<nbase;bw++) for(int sw=0;sw<nspecies;sw++) count += base_cidx_map[bv][sv][bw][sw].size();
	if(count == 0) continue;

	Field4DInputParamTypeV V_fieldparams = V[sv]->getFieldInputParams();
		
#ifndef DISABLE_FFT_RELN_USAGE

# ifdef USE_DESTRUCTIVE_FFT
	if(!UniqueID()){
	  printf("ComputeMesonFields::compute Allocating V FFT of size %f MB dynamically as V of size %f MB is deallocated\n",
		 A2AvectorVfftw<mf_Policies>::Mbyte_size(V[sv]->getArgs(), V_fieldparams),
		 A2AvectorV<mf_Policies>::Mbyte_size(V[sv]->getArgs(), V_fieldparams));
	  fflush(stdout);
	}
# else
	if(!UniqueID()){ 
	  printf("ComputeMesonFields::compute Allocating a V FFT of size %f MB\n", A2AvectorVfftw<mf_Policies>::Mbyte_size(V[sv]->getArgs(), V_fieldparams)); fflush(stdout);
	}
# endif
	
	A2AvectorVfftw<mf_Policies> fftw_V_base(V[sv]->getArgs(), V_fieldparams);
# ifdef USE_DESTRUCTIVE_FFT
	assert(!fftw_V_base.modeIsAllocated(0));
	fftw_V_base.destructiveGaugeFixTwistFFT(*V[sv], pvb.ptr(),lattice); //allocs Vfft and deallocs V internally	
	assert(!V[sv]->modeIsAllocated(0));
# else
	fftw_V_base.gaugeFixTwistFFT(*V[sv], pvb.ptr(),lattice);
# endif
	
	A2AvectorVfftw<mf_Policies> * Vfftw_base_0 = bv == 0 ? &fftw_V_base : NULL;
	A2AvectorVfftw<mf_Policies> * Vfftw_base_1 = bv == 1 ? &fftw_V_base : NULL;
#endif

	printMem("ComputeMesonFields::compute Memory after V FFT");
	
	for(int bw=0;bw<nbase;bw++){
	  const ThreeMomentum &pwb = pbase[bw];
	  for(int sw=0;sw<nspecies;sw++){
	    if(base_cidx_map[bv][sv][bw][sw].size() == 0) continue;

	    Field4DInputParamTypeW W_fieldparams = W[sw]->getFieldInputParams();
	    
#ifndef DISABLE_FFT_RELN_USAGE

# ifdef USE_DESTRUCTIVE_FFT
	    if(!UniqueID()){
	      printf("ComputeMesonFields::compute Allocating W FFT of size %f MB dynamically as W of size %f MB is deallocated\n",
		     A2AvectorWfftw<mf_Policies>::Mbyte_size(W[sw]->getArgs(), W_fieldparams),
		     A2AvectorW<mf_Policies>::Mbyte_size(W[sw]->getArgs(), W_fieldparams));
	      fflush(stdout);
	    }
# else
	    if(!UniqueID()){ printf("ComputeMesonFields::compute Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[sw]->getArgs(), W_fieldparams)); fflush(stdout); }
# endif
	    
	    A2AvectorWfftw<mf_Policies> fftw_W_base(W[sw]->getArgs(), W_fieldparams);
# ifdef USE_DESTRUCTIVE_FFT
	    fftw_W_base.destructiveGaugeFixTwistFFT(*W[sw], pwb.ptr(),lattice);
# else
	    fftw_W_base.gaugeFixTwistFFT(*W[sw], pwb.ptr(),lattice);
# endif

	    A2AvectorWfftw<mf_Policies> * Wfftw_base_0 = bw == 0 ? &fftw_W_base : NULL;
	    A2AvectorWfftw<mf_Policies> * Wfftw_base_1 = bw == 1 ? &fftw_W_base : NULL;
#endif

	    printMem("ComputeMesonFields::compute Memory after W FFT");

	    //Now loop over computations with this V-base, W-base
	    for(int cc=0; cc < base_cidx_map[bv][sv][bw][sw].size(); cc++){
	      const int cidx = base_cidx_map[bv][sv][bw][sw][cc];
	      
	      typename StorageType::mfComputeInputFormat cdest = into.getMf(cidx);
	      const typename StorageType::InnerProductType &M = into.getInnerProduct(cidx);
	      
	      int qidx_w, qidx_v;
	      ThreeMomentum p_w, p_v;      
	      into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,cidx);
	      
	      assert(qidx_v == sv && qidx_w == sw);
	      
	      if(!UniqueID()){ printf("ComputeMesonFields::compute Computing mesonfield with W species %d and momentum %s and V species %d and momentum %s\n",qidx_w,p_w.str().c_str(),qidx_v,p_v.str().c_str()); fflush(stdout); }

	      //The memory-saving magic of this approach only works if we have FFT relation usage enabled!
#if defined(DISABLE_FFT_RELN_USAGE) || !defined(COMPUTEMANY_INPLACE_SHIFT)
	      if(!UniqueID()){ printf("ComputeMesonFields::compute Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W[qidx_w]->getArgs(), W_fieldparams)); fflush(stdout); }
	      A2AvectorWfftw<mf_Policies> fftw_W(W[qidx_w]->getArgs(), W_fieldparams );

	      if(!UniqueID()){ printf("ComputeMesonFields::compute Allocating a V FFT of size %f MB\n", A2AvectorVfftw<mf_Policies>::Mbyte_size(V[qidx_v]->getArgs(), V_fieldparams)); fflush(stdout); }
	      A2AvectorVfftw<mf_Policies> fftw_V(V[qidx_v]->getArgs(), V_fieldparams );
	      
# ifdef USE_DESTRUCTIVE_FFT
	      fftw_W.allocModes();
	      fftw_V.allocModes();
# endif

# ifdef DISABLE_FFT_RELN_USAGE
#   ifdef USE_DESTRUCTIVE_FFT
#   error "In ComputeMany cannot combine DISABLE_FFT_RELN_USAGE with USE_DESTRUCTIVE_FFT as the W and V vectors have been deallocated by the point of use"
#   else	      
	      fftw_W.gaugeFixTwistFFT(*W[qidx_w], p_w.ptr(),lattice);
	      fftw_V.gaugeFixTwistFFT(*V[qidx_v], p_v.ptr(),lattice);
#   endif
# else
	      fftw_W.getTwistedFFT(p_w.ptr(), Wfftw_base_0, Wfftw_base_1);
	      fftw_V.getTwistedFFT(p_v.ptr(), Vfftw_base_0, Vfftw_base_1);
# endif
	      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(cdest,fftw_W, M, fftw_V);
	      
#else //COMPUTEMANY_INPLACE_SHIFT	      
	      std::vector<int> restore_shift_w, restore_shift_v;
	  
	      if(!UniqueID()){ printf("ComputeMesonFields::compute Shifting base Wfftw in place\n"); fflush(stdout); }
	      std::pair< A2AvectorWfftw<mf_Policies>*, std::vector<int> > inplace_w = A2AvectorWfftw<mf_Policies>::inPlaceTwistedFFT(p_w.ptr(), Wfftw_base_0, Wfftw_base_1);
	      restore_shift_w = inplace_w.second;
	      const A2AvectorWfftw<mf_Policies> &fftw_W = fftw_W_base;

	      if(!UniqueID()){ printf("ComputeMesonFields::compute Shifting base Vfftw in place\n"); fflush(stdout); }
	      std::pair< A2AvectorVfftw<mf_Policies>*, std::vector<int> > inplace_v = A2AvectorVfftw<mf_Policies>::inPlaceTwistedFFT(p_v.ptr(), Vfftw_base_0, Vfftw_base_1);
	      restore_shift_v = inplace_v.second;
	      const A2AvectorVfftw<mf_Policies> &fftw_V = fftw_V_base;

	      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(cdest,fftw_W, M, fftw_V);

# ifndef USE_DESTRUCTIVE_FFT
	      if(cc != base_cidx_map[bv][sv][bw][sw].size() -1){
# endif		
		fftw_W_base.shiftFieldsInPlace(restore_shift_w);
		fftw_V_base.shiftFieldsInPlace(restore_shift_v);
# ifndef USE_DESTRUCTIVE_FFT	       
	      }
# endif
	      
#endif
	      if(node_distribute){
		printMem("ComputeMesonFields::compute Memory before distribute");
		into.nodeDistributeResult(cidx);
		printMem("ComputeMesonFields::compute Memory after distribute");
	      }
	    }//cc

#ifdef USE_DESTRUCTIVE_FFT
	    if(!UniqueID()){
	      printf("ComputeMesonFields::compute Restoring W of size %f MB dynamically as W FFT of size %f MB is deallocated\n",
		     A2AvectorW<mf_Policies>::Mbyte_size(fftw_W_base.getArgs(), W_fieldparams),
		     A2AvectorWfftw<mf_Policies>::Mbyte_size(fftw_W_base.getArgs(), W_fieldparams)
		     );
	      fflush(stdout);
	    }
	    printMem("Prior to W restore");
	    fftw_W_base.destructiveUnapplyGaugeFixTwistFFT(*W[sw], pwb.ptr(),lattice);
	    printMem("After W restore");
#endif
	    
	  }//sw
	}//bw

#ifdef USE_DESTRUCTIVE_FFT
	if(!UniqueID()){
	  printf("ComputeMesonFields::compute Restoring V of size %f MB dynamically as V FFT of size %f MB is deallocated\n",
		 A2AvectorV<mf_Policies>::Mbyte_size(fftw_V_base.getArgs(), V_fieldparams),
		 A2AvectorVfftw<mf_Policies>::Mbyte_size(fftw_V_base.getArgs(), V_fieldparams)
		 );
	  fflush(stdout);
	}
	printMem("Prior to V restore");
	fftw_V_base.destructiveUnapplyGaugeFixTwistFFT(*V[sv], pvb.ptr(),lattice);  //allocs V and deallocs Vfft internally
	printMem("After V restore");
#endif	
      }//sv
    }//bv
    
    printMem("ComputeMesonFields::compute Memory after compute loop");
  }
};



CPS_END_NAMESPACE

#endif
