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
  
  typedef typename mf_Policies::ComplexType ComplexType;
  typedef typename mf_Policies::SourcePolicies SourcePolicies;
  typedef typename mf_Policies::FermionFieldType::InputParamType VWfieldInputParams;
  
  typedef typename A2AvectorVfftw<mf_Policies>::FieldInputParamType Field4DInputParamTypeV;
  typedef typename A2AvectorWfftw<mf_Policies>::FieldInputParamType Field4DInputParamTypeW;

  template<typename FFTvector, typename BaseVector>
  static void gaugeFixTwist(FFTvector &out, BaseVector &in, const ThreeMomentum &p, Lattice &lattice){
#ifdef USE_DESTRUCTIVE_FFT
    out.destructiveGaugeFixTwistFFT(in, p.ptr(),lattice); //allocs out and deallocs in internally	
#else
    out.gaugeFixTwistFFT(in, p.ptr(),lattice);
#endif
  }

  template<typename FFTvector, typename BaseVector>
  static void restore(BaseVector &out, FFTvector &in, const ThreeMomentum &p, Lattice &lattice, const std::string &descr){
#ifdef USE_DESTRUCTIVE_FFT
    if(!UniqueID()){
      printf("ComputeMesonFields::compute Restoring %s of size %f MB dynamically as W FFT of size %f MB is deallocated\n",
	     descr.c_str(),
	     out.Mbyte_size(in.getArgs(),in.getFieldInputParams()),
	     in.Mbyte_size(in.getArgs(),in.getFieldInputParams()));
      fflush(stdout);
    }
    printMem(stringize("Prior to %s restore", descr.c_str()));
    in.destructiveUnapplyGaugeFixTwistFFT(out, p.ptr(),lattice); 
    printMem(stringize("After %s restore", descr.c_str()));
#endif
  }

  template< template<typename> class FFTtype, template<typename> class BaseType>
  static void printAllocMessage(const BaseType<mf_Policies> &base_field, const std::string &descr){
    if(!UniqueID()){
#ifdef USE_DESTRUCTIVE_FFT
      printf("ComputeMesonFields::compute Allocating %s FFT of size %f MB dynamically as %s of size %f MB is deallocated\n",
	descr.c_str(),
	FFTtype<mf_Policies>::Mbyte_size(base_field.getArgs(),base_field.getFieldInputParams()),
	descr.c_str(),
	BaseType<mf_Policies>::Mbyte_size(base_field.getArgs(),base_field.getFieldInputParams()) );
#else
      printf("ComputeMesonFields::compute Allocating a %s FFT of size %f MB\n", descr.c_str(), FFTtype<mf_Policies>::Mbyte_size(base_field.getArgs(), base_field.getFieldInputParams()));
#endif
      fflush(stdout);
    }
  }
  

  //Basic implementation that just runs over contractions and performs them without an eye for optimization
  static void computeSimple(StorageType &into, WspeciesVector &W, VspeciesVector &V,  Lattice &lattice, const bool node_distribute = false){
    for(int c=0;c<into.nCompute();c++){
      int qidx_w, qidx_v;
      ThreeMomentum p_w, p_v;      
      into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,c);
      
      Field4DInputParamTypeV V_fieldparams = V[qidx_v]->getFieldInputParams();
      Field4DInputParamTypeW W_fieldparams = W[qidx_w]->getFieldInputParams();
      
      typename StorageType::mfComputeInputFormat cdest = into.getMf(c);
      const typename StorageType::InnerProductType &M = into.getInnerProduct(c);

      printAllocMessage<A2AvectorWfftw, A2AvectorW>(*W[qidx_w],"W");
      A2AvectorWfftw<mf_Policies> fftw_W(W[qidx_w]->getArgs(), W_fieldparams );
      
      printAllocMessage<A2AvectorVfftw, A2AvectorV>(*V[qidx_v],"V");
      A2AvectorVfftw<mf_Policies> fftw_V(V[qidx_v]->getArgs(), V_fieldparams );

      gaugeFixTwist(fftw_W,*W[qidx_w], p_w.ptr(),lattice);
      gaugeFixTwist(fftw_V,*V[qidx_v], p_v.ptr(),lattice);
      
      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(cdest,fftw_W, M, fftw_V);
      
      into.postContractAction(c);

      restore(*W[qidx_w], fftw_W, p_w, lattice, "W");
      restore(*V[qidx_v], fftw_V, p_v, lattice, "V");

      if(node_distribute){
	printMem("ComputeMesonFields::compute Memory before distribute");
	into.nodeDistributeResult(c);
	printMem("ComputeMesonFields::compute Memory after distribute");
      }
    }
  }


  //We avoid extra FFTs by c-shifting in momentum space a single FFT (no Gparity) or two FFTs (Gparity)
  static void getBaseMomenta(std::vector<ThreeMomentum> &pbase){
    if(GJP.Gparity()){
      int p_p1[3], p_m1[3];
      GparityBaseMomentum(p_p1,+1);
      GparityBaseMomentum(p_m1,-1);
      
      pbase.resize(2);
      pbase[0] = ThreeMomentum(p_p1);
      pbase[1] = ThreeMomentum(p_m1);
    }else{
      pbase.resize(1, ThreeMomentum(0,0,0));
    }
  }

  //Construct a mapping between a base momentum index plus a quark species index for both W and V to the internal index of the StorageType
  //This allows us to loop over the base and species index and perform the appropriate contractions for each
  static void getBaseCidxMap(std::vector< std::vector< std::vector< std::vector< std::vector<int> > > > > &base_cidx_map, 
			     const std::vector<ThreeMomentum> &pbase, const int nspecies, StorageType &into){
    const int nbase = pbase.size();
    base_cidx_map.resize(nbase); //[base_v][qidx_v][base_w][qidx_w]
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
  }


  //Given the base FFTs, get the C-shifted FFTs and compute the meson field
  static void shiftBaseAndComputeMF(typename StorageType::mfComputeInputFormat cdest,
				  const A2AArg &W_args, const Field4DInputParamTypeW &W_fieldparams,
				  const A2AArg &V_args, const Field4DInputParamTypeW &V_fieldparams,
				  const ThreeMomentum &p_w, const ThreeMomentum &p_v,
				  A2AvectorWfftw<mf_Policies> * Wfftw_base[2], A2AvectorVfftw<mf_Policies> * Vfftw_base[2],
				  const typename StorageType::InnerProductType &M){

    if(!UniqueID()){ printf("ComputeMesonFields::compute Allocating a W FFT of size %f MB\n", A2AvectorWfftw<mf_Policies>::Mbyte_size(W_args, W_fieldparams)); fflush(stdout); }
    A2AvectorWfftw<mf_Policies> fftw_W(W_args, W_fieldparams );

    if(!UniqueID()){ printf("ComputeMesonFields::compute Allocating a V FFT of size %f MB\n", A2AvectorVfftw<mf_Policies>::Mbyte_size(V_args, V_fieldparams)); fflush(stdout); }
    A2AvectorVfftw<mf_Policies> fftw_V(V_args, V_fieldparams );
	      
#ifdef USE_DESTRUCTIVE_FFT
    fftw_W.allocModes();
    fftw_V.allocModes();
#endif
    
    fftw_W.getTwistedFFT(p_w.ptr(), Wfftw_base[0], Wfftw_base[1]);
    fftw_V.getTwistedFFT(p_v.ptr(), Vfftw_base[0], Vfftw_base[1]);
    A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(cdest,fftw_W, M, fftw_V);
  }

  //Same as above but in this version we shift the base *in-place* (i.e. without using extra memory) - of course we need to shift it back again afterwards
  static void shiftBaseInPlaceAndComputeMF(typename StorageType::mfComputeInputFormat cdest,
					   const A2AArg &W_args, const Field4DInputParamTypeW &W_fieldparams,
					   const A2AArg &V_args, const Field4DInputParamTypeW &V_fieldparams,
					   const ThreeMomentum &p_w, const ThreeMomentum &p_v,
					   A2AvectorWfftw<mf_Policies> * Wfftw_base[2], A2AvectorVfftw<mf_Policies> * Vfftw_base[2],
					   const typename StorageType::InnerProductType &M, 
					   bool is_last){

    if(!UniqueID()){ printf("ComputeMesonFields::compute Shifting base Wfftw in place\n"); fflush(stdout); }
    std::pair< A2AvectorWfftw<mf_Policies>*, std::vector<int> > inplace_w = A2AvectorWfftw<mf_Policies>::inPlaceTwistedFFT(p_w.ptr(), Wfftw_base[0], Wfftw_base[1]);
    const A2AvectorWfftw<mf_Policies> &fftw_W = *inplace_w.first;

    if(!UniqueID()){ printf("ComputeMesonFields::compute Shifting base Vfftw in place\n"); fflush(stdout); }
    std::pair< A2AvectorVfftw<mf_Policies>*, std::vector<int> > inplace_v = A2AvectorVfftw<mf_Policies>::inPlaceTwistedFFT(p_v.ptr(), Vfftw_base[0], Vfftw_base[1]);
    const A2AvectorVfftw<mf_Policies> &fftw_V = *inplace_v.first;

    A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw>::compute(cdest,fftw_W, M, fftw_V);

    bool do_restore = !is_last; //we can save a shift by not restoring on the last use of this base FFT, as we will be throwing it away anyway
#ifdef USE_DESTRUCTIVE_FFT
    do_restore = true; //we must restore if we are using destructive FFTs as we need to unapply the FFT
#endif
    
    if(do_restore){
      inplace_w.first->shiftFieldsInPlace(inplace_w.second);
      inplace_v.first->shiftFieldsInPlace(inplace_v.second);
    }
  }


  //W and V are indexed by the quark type index
  static void computeOptimized(StorageType &into, WspeciesVector &W, VspeciesVector &V,  Lattice &lattice, const bool node_distribute = false){
    const int Lt = GJP.Tnodes()*GJP.TnodeSites();

    assert(W.size() == V.size());
    const int nspecies = W.size();

    std::vector<ThreeMomentum> pbase;
    getBaseMomenta(pbase);
    const int nbase = pbase.size();
  
    printMem("ComputeMesonFields::compute Memory prior to compute loop");

    //We need to group the V shifts by base and species
    std::vector< std::vector< std::vector< std::vector< std::vector<int> > > > > base_cidx_map;
    getBaseCidxMap(base_cidx_map, pbase, nspecies, into);
    
    //Do the computations with an outer loop over the base momentum index and species of the Vfftw
    for(int bv=0;bv<nbase;bv++){
      const ThreeMomentum &pvb = pbase[bv];      
      for(int sv=0;sv<nspecies;sv++){	
	int count = 0;
	for(int bw=0;bw<nbase;bw++) for(int sw=0;sw<nspecies;sw++) count += base_cidx_map[bv][sv][bw][sw].size();
	if(count == 0) continue;

	Field4DInputParamTypeV V_fieldparams = V[sv]->getFieldInputParams();
	A2AArg V_args = V[sv]->getArgs();

	//Do the FFT of the base V vector
	printAllocMessage<A2AvectorVfftw, A2AvectorV>(*V[sv],"V");
	A2AvectorVfftw<mf_Policies> fftw_V_base(V_args, V_fieldparams);
	gaugeFixTwist(fftw_V_base, *V[sv], pvb, lattice);
	
	A2AvectorVfftw<mf_Policies> * Vfftw_base[2] = {NULL,NULL};
	Vfftw_base[bv] = &fftw_V_base;

	printMem("ComputeMesonFields::compute Memory after V FFT");
	
	for(int bw=0;bw<nbase;bw++){
	  const ThreeMomentum &pwb = pbase[bw];
	  for(int sw=0;sw<nspecies;sw++){
	    if(base_cidx_map[bv][sv][bw][sw].size() == 0) continue;

	    Field4DInputParamTypeW W_fieldparams = W[sw]->getFieldInputParams();
	    A2AArg W_args = W[sw]->getArgs();

	    //Do the FFT of the base W vector
	    printAllocMessage<A2AvectorWfftw, A2AvectorW>(*W[sw],"W");
	    A2AvectorWfftw<mf_Policies> fftw_W_base(W_args, W_fieldparams);
	    gaugeFixTwist(fftw_W_base, *W[sw], pwb, lattice);

	    A2AvectorWfftw<mf_Policies> * Wfftw_base[2] = {NULL,NULL};
	    Wfftw_base[bw] = &fftw_W_base;

	    printMem("ComputeMesonFields::compute Memory after W FFT");

	    //Now loop over computations with this V-base, W-base
	    const int ncon = base_cidx_map[bv][sv][bw][sw].size();
	    for(int cc=0; cc < ncon; cc++){
	      const int cidx = base_cidx_map[bv][sv][bw][sw][cc];
	      
	      typename StorageType::mfComputeInputFormat cdest = into.getMf(cidx);
	      const typename StorageType::InnerProductType &M = into.getInnerProduct(cidx);
	      
	      int qidx_w, qidx_v;
	      ThreeMomentum p_w, p_v;      
	      into.getComputeParameters(qidx_w,qidx_v,p_w,p_v,cidx);
	      
	      assert(qidx_v == sv && qidx_w == sw);
	      
	      if(!UniqueID()){ printf("ComputeMesonFields::compute Computing mesonfield with W species %d and momentum %s and V species %d and momentum %s\n",
				      qidx_w,p_w.str().c_str(),qidx_v,p_v.str().c_str()); fflush(stdout); }

#ifdef COMPUTEMANY_INPLACE_SHIFT
	      shiftBaseInPlaceAndComputeMF(cdest, W_args,W_fieldparams,V_args,V_fieldparams,p_w,p_v,Wfftw_base, Vfftw_base, M, 
					   cc == ncon-1);
#else
	      shiftBaseAndComputeMF(cdest, W_args,W_fieldparams,V_args,V_fieldparams,p_w,p_v,Wfftw_base, Vfftw_base, M);	      
#endif

	      into.postContractAction(cidx);

	      if(node_distribute){
		printMem("ComputeMesonFields::compute Memory before distribute");
		into.nodeDistributeResult(cidx);
		printMem("ComputeMesonFields::compute Memory after distribute");
	      }
	    }//cc

	    restore(*W[sw],fftw_W_base,pwb,lattice,"W");    
	  }//sw
	}//bw

	restore(*V[sv],fftw_V_base,pvb,lattice,"V");    

      }//sv
    }//bv
    
    printMem("ComputeMesonFields::compute Memory after compute loop");
  }

  static void compute(StorageType &into, WspeciesVector &W, VspeciesVector &V,  Lattice &lattice, const bool node_distribute = false){
#ifdef DISABLE_FFT_RELN_USAGE
    computeSimple(into,W,V,lattice,node_distribute);
#else
    computeOptimized(into,W,V,lattice,node_distribute);
#endif
  }
};



CPS_END_NAMESPACE

#endif
