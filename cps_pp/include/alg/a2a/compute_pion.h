#ifndef _COMPUTE_PION_H
#define _COMPUTE_PION_H

#include<memory>
#include<alg/a2a/mf_momcontainer.h>

CPS_START_NAMESPACE

//Policy for RequiredMomenta. This is the set of momenta that Daiqian used.
class StandardPionMomentaPolicy{
public:
  void setupMomenta(const int &ngp){
    RequiredMomentum<StandardPionMomentaPolicy> *tt = static_cast<RequiredMomentum<StandardPionMomentaPolicy>*>(this);
    if(ngp == 0){
      //p_pi = (0,0,0)
      tt->addP("(0,0,0) + (0,0,0)",false);
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      tt->addPandMinusP("(-1,0,0) + (-1,0,0)",false); tt->addPandMinusP("(1,0,0) + (-3,0,0)",true); //alternative momentum   
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,0) + (-1,-1,0)",false); tt->addPandMinusP("(1,1,0) + (-3,-3,0)",true);

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      tt->addPandMinusP("(-1,-1,0) + (3,-1,0)",false); tt->addPandMinusP("(1,1,0) + (1,-3,0)",true);
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,-1)",false); tt->addPandMinusP("(1,1,1) + (-3,-3,-3)",true);

      //p_pi = (2,-2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (3,-1,-1)",false); tt->addPandMinusP("(1,1,1) + (1,-3,-3)",true);

      //p_pi = (-2,2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,3,-1)",false); tt->addPandMinusP("(1,1,1) + (-3,1,-3)",true);

      //p_pi = (-2,-2,2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,3)",false); tt->addPandMinusP("(1,1,1) + (-3,-3,1)",true);
    }else{
      ERR.General("StandardPionMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
};

//This set of momenta does not include the second momentum combination with which we average to reduce the G-parity rotational symmetry breaking
class H4asymmetricMomentaPolicy{
public:
  void setupMomenta(const int &ngp){
    RequiredMomentum<H4asymmetricMomentaPolicy> *tt = static_cast<RequiredMomentum<H4asymmetricMomentaPolicy>*>(this);
    if(ngp == 0){
      //p_pi = (0,0,0)
      tt->addP("(0,0,0) + (0,0,0)",false);
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      tt->addPandMinusP("(-1,0,0) + (-1,0,0)",false);
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,0) + (-1,-1,0)",false);

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      tt->addPandMinusP("(-1,-1,0) + (3,-1,0)",false);
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)  
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,-1)",false);

      //p_pi = (2,-2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (3,-1,-1)",false);

      //p_pi = (-2,2,-2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,3,-1)",false);

      //p_pi = (-2,-2,2)
      tt->addPandMinusP("(-1,-1,-1) + (-1,-1,3)",false);
    }else{
      ERR.General("H4asymmetricMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
};









template<typename mf_Complex>
class ComputePion{
 public:
  //These meson fields are also used by the pi-pi and K->pipi calculations
  template<typename PionMomentumPolicy>
  static void computeMesonFields(std::vector< std::vector<A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw> > > &mf_ll, //output vector for meson fields
				 MesonFieldMomentumContainer<mf_Complex> &mf_ll_con, //convenient storage for pointers to the above, assembled at same time
				 const RequiredMomentum<PionMomentumPolicy> &pion_mom, //object that tells us what quark momenta to use
				 const A2AvectorW<mf_Complex> &W, const A2AvectorV<mf_Complex> &V,
				 const Float &rad, //exponential wavefunction radius
				 Lattice &lattice){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    int nmom = pion_mom.nMom();
    if(pion_mom.nAltMom() > 0 && pion_mom.nAltMom() != nmom)
      ERR.General("ComputePion","computeMesonFields","If alternate momentum combinations are specified there must be one for each pion momentum!\n");

    mf_ll.resize(nmom);

    A2AvectorWfftw<mf_Complex> fftw_W(W.getArgs());
    A2AvectorVfftw<mf_Complex> fftw_V(V.getArgs());

    //For useful info to user, compute required memory size of all light-light meson fields
    {
      double mf_size = A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw>::byte_size(W.getArgs(),V.getArgs()) / (1024.0*1024.0); //in MB
      double all_mf_size = Lt * nmom * mf_size;
      if(!UniqueID()) printf("Memory requirement for light-light meson fields: %f MB (each %f MB)\n",all_mf_size,mf_size);
    }

    //For non-Gparity
    std::auto_ptr<A2AexpSource> expsrc_nogp; 
    std::auto_ptr<SCspinInnerProduct<mf_Complex,A2AexpSource> > mf_struct_nogp;
    if(!GJP.Gparity()){
      expsrc_nogp.reset(new A2AexpSource(rad));
      mf_struct_nogp.reset(new SCspinInnerProduct<mf_Complex,A2AexpSource>(15,*expsrc_nogp));
    }

    for(int pidx=0;pidx<nmom;pidx++){
      if(!UniqueID()) printf("Generating light-light meson fields pidx=%d\n",pidx);
      double time = -dclock();

      mf_ll[pidx].resize(Lt);

      ThreeMomentum p_w = pion_mom.getWmom(pidx);
      ThreeMomentum p_v = pion_mom.getVmom(pidx);

      fftw_W.gaugeFixTwistFFT(W, p_w.ptr(),lattice);
      fftw_V.gaugeFixTwistFFT(V, p_v.ptr(),lattice); 

      //Meson fields with standard momentum configuration
      if(!GJP.Gparity()){
	A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_ll[pidx], fftw_W, *mf_struct_nogp, fftw_V);

	// for(int t=0;t<Lt;t++)
	//   mf_ll[pidx][t].compute(fftw_W, *mf_struct_nogp, fftw_V, t);
      }else{
	A2AflavorProjectedExpSource fpexp(rad, p_v.ptr()); //flavor projection is adjacent to right-hand field
	SCFspinflavorInnerProduct<mf_Complex,A2AflavorProjectedExpSource> mf_struct(sigma3,15,fpexp);

	A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_ll[pidx],fftw_W, mf_struct, fftw_V);

	//for(int t=0;t<Lt;t++)
	//mf_ll[pidx][t].compute(fftw_W, mf_struct, fftw_V, t);
      }

      if(GJP.Gparity() && pion_mom.nAltMom() > 0){
	if(!UniqueID()) printf("Generating second momentum combination G-parity quarks\n",pidx);
      
	//Average with second momentum configuration to reduce rotational symmetry breaking
	ThreeMomentum p_w_alt = pion_mom.getWmom(pidx,true);
	ThreeMomentum p_v_alt = pion_mom.getVmom(pidx,true);

	fftw_W.gaugeFixTwistFFT(W,p_w_alt.ptr(),lattice);
	fftw_V.gaugeFixTwistFFT(V,p_v_alt.ptr(),lattice);
    
	A2AflavorProjectedExpSource fpexp(rad, p_v_alt.ptr()); 
	SCFspinflavorInnerProduct<mf_Complex,A2AflavorProjectedExpSource> mf_struct(sigma3,15,fpexp);

	//A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw> mf_ll_alt;
	// for(int t=0;t<Lt;t++){
	//   mf_ll_alt.compute(fftw_W, mf_struct, fftw_V, t);
	//   mf_ll[pidx][t].average(mf_ll_alt);
	// } 

	std::vector<A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw> > mf_ll_alt(Lt);
	A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw>::compute(mf_ll_alt,fftw_W, mf_struct, fftw_V);
	for(int t=0;t<Lt;t++){
	  mf_ll[pidx][t].average(mf_ll_alt[t]);
	}
      }
#ifdef NODE_DISTRIBUTE_MESONFIELDS
      if(!UniqueID()){ printf("Distributing mf_ll[%d]\n",pidx); fflush(stdout); }
      nodeDistributeMany(1,&mf_ll[pidx]);
      //for(int t=0;t<Lt;t++) mf_ll[pidx][t].nodeDistribute();
#endif

      mf_ll_con.add( pion_mom.getMesonMomentum(pidx), mf_ll[pidx]);
      
      time += dclock();
      print_time("ComputePion","meson field",time);
    }
  }





  //Compute the two-point function using the pre-generated meson fields.
  //The pion is given the momentum associated with index 'pidx' in the RequiredMomentum
  //result is indexed by (tsrc, tsep)  where tsep is the source-sink separation
  template<typename PionMomentumPolicy>
  static void compute(fMatrix<mf_Complex> &into, MesonFieldMomentumContainer<mf_Complex> &mf_ll_con, const RequiredMomentum<PionMomentumPolicy> &pion_mom, const int pidx){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    ThreeMomentum p_pi_src = pion_mom.getMesonMomentum(pidx); //exp(-ipx)
    ThreeMomentum p_pi_snk = -p_pi_src; //exp(+ipx)

    assert(mf_ll_con.contains(p_pi_src));
    assert(mf_ll_con.contains(p_pi_snk));
	   
    //Construct the meson fields
    std::vector<A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_src = mf_ll_con.get(p_pi_src);
    std::vector<A2AmesonField<mf_Complex,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_snk = mf_ll_con.get(p_pi_snk);
#ifdef NODE_DISTRIBUTE_MESONFIELDS
    if(!UniqueID()){ printf("Gathering meson fields\n");  fflush(stdout); }
    nodeGetMany(2,&mf_ll_src,&mf_ll_snk);
    sync();
#endif

    //Compute the two-point function
    // \sum_{xsnk,xsrc} exp(ip xsnk)exp(-ipxsrc) tr( S G(xsnk, tsnk; xsrc, tsrc) S G(xsrc,tsrc; xsnk, tsnk) ) 
    //where S is the vertex spin/color/flavor structure

    //= tr( [[\sum_{xsnk} exp(ip xsnk) w^dag(xsnk,tsnk) S v(xsnk,tsnk)]] [[\sum_{xsrc} exp(-ip xsrc) w^dag(xsrc,tsrc) S v(xsrc,tsrc) ]] ) 
    if(!UniqueID()){ printf("Starting trace\n");  fflush(stdout); }
    trace(into,mf_ll_snk,mf_ll_src);
    into *= mf_Complex(0.5,0);
    rearrangeTsrcTsep(into); //rearrange temporal ordering
    
    sync();
    if(!UniqueID()){ printf("Finished trace\n");  fflush(stdout); }

#ifdef NODE_DISTRIBUTE_MESONFIELDS
    nodeDistributeMany(2,&mf_ll_src,&mf_ll_snk);
#endif
  }

};



CPS_END_NAMESPACE

#endif
