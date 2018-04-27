#ifndef _COMPUTE_PION_H
#define _COMPUTE_PION_H

#include<alg/a2a/required_momenta.h>
#include<alg/a2a/mesonfield_computemany.h>
#include<alg/a2a/inner_product.h>
#include<alg/a2a/mf_momcontainer.h>

CPS_START_NAMESPACE

//Policy for RequiredMomenta. This is the set of momenta that Daiqian used.
class StandardPionMomentaPolicy: public RequiredMomentum{
public:
  StandardPionMomentaPolicy(): RequiredMomentum() {
    this->combineSameTotalMomentum(true); //momentum pairs with same total momentum will be added to same entry and treated as 'alternates' which we average together below

    const int ngp = this->nGparityDirs();
    if(ngp == 0){
      //p_pi = (0,0,0)
      addP("(0,0,0) + (0,0,0)");
      //one unit of momenta in units of 2pi/L
      addPandMinusP("(1,0,0) + (0,0,0)");
      addPandMinusP("(0,1,0) + (0,0,0)");
      addPandMinusP("(0,0,1) + (0,0,0)");
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      addPandMinusP("(-1,0,0) + (-1,0,0)"); addPandMinusP("(1,0,0) + (-3,0,0)"); //alternative momentum   
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      addPandMinusP("(-1,-1,0) + (-1,-1,0)"); addPandMinusP("(1,1,0) + (-3,-3,0)");

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      addPandMinusP("(-1,-1,0) + (3,-1,0)"); addPandMinusP("(1,1,0) + (1,-3,0)");
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)
      addPandMinusP("(-1,-1,-1) + (-1,-1,-1)"); addPandMinusP("(1,1,1) + (-3,-3,-3)");

      //p_pi = (2,-2,-2)
      addPandMinusP("(-1,-1,-1) + (3,-1,-1)"); addPandMinusP("(1,1,1) + (1,-3,-3)");

      //p_pi = (-2,2,-2)
      addPandMinusP("(-1,-1,-1) + (-1,3,-1)"); addPandMinusP("(1,1,1) + (-3,1,-3)");

      //p_pi = (-2,-2,2)
      addPandMinusP("(-1,-1,-1) + (-1,-1,3)"); addPandMinusP("(1,1,1) + (-3,-3,1)");

      assert(nMom() == 8);
      for(int i=0;i<8;i++) assert(nAltMom(i) == 2);
    }else{
      ERR.General("StandardPionMomentaPolicy","constructor","ngp cannot be >3\n");
    }
  };
};


//Same as the above but where we reverse the momentum assignments of the W^dag and V
class ReversePionMomentaPolicy: public StandardPionMomentaPolicy{
public:
  ReversePionMomentaPolicy(): StandardPionMomentaPolicy() {
    this->reverseABmomentumAssignments();
  }
};

//Add Wdag, V momenta and the reverse assignment to make a symmetric combination
class SymmetricPionMomentaPolicy: public StandardPionMomentaPolicy{
public:
  SymmetricPionMomentaPolicy(): StandardPionMomentaPolicy() {
    this->symmetrizeABmomentumAssignments();
  }
};


//This set of momenta does not include the second momentum combination with which we average to reduce the G-parity rotational symmetry breaking
class H4asymmetricMomentaPolicy: public RequiredMomentum{
public:
  void setupMomenta(){
    const int ngp = this->nGparityDirs();
    if(ngp == 0){
      //p_pi = (0,0,0)
      addP("(0,0,0) + (0,0,0)");
    }else if(ngp == 1){
      //p_pi = (-2,0,0)     (units of pi/2L)    
      addPandMinusP("(-1,0,0) + (-1,0,0)");
      //(In case you're wondering why my first momentum has the opposite sign to Daiqian's, its because mine is for W^dagger, not W)
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (-2,-2,0)     (units of pi/2L)  
      addPandMinusP("(-1,-1,0) + (-1,-1,0)");

      //Along off-diagonal direction:      
      //p_pi = (2,-2,0)
      addPandMinusP("(-1,-1,0) + (3,-1,0)");
    }else if(ngp == 3){
      //p_pi = (-2,-2,-2)     (units of pi/2L)  
      addPandMinusP("(-1,-1,-1) + (-1,-1,-1)");

      //p_pi = (2,-2,-2)
      addPandMinusP("(-1,-1,-1) + (3,-1,-1)");

      //p_pi = (-2,2,-2)
      addPandMinusP("(-1,-1,-1) + (-1,3,-1)");

      //p_pi = (-2,-2,2)
      addPandMinusP("(-1,-1,-1) + (-1,-1,3)");
    }else{
      ERR.General("H4asymmetricMomentaPolicy","setupMomenta","ngp cannot be >3\n");
    }
  }
  H4asymmetricMomentaPolicy(): RequiredMomentum() { setupMomenta();};
};









template<typename mf_Policies>
class ComputePion{
/* public: */
/*   typedef typename A2Asource<typename mf_Policies::SourcePolicies::ComplexType, typename mf_Policies::SourcePolicies::MappingPolicy, typename mf_Policies::SourcePolicies::AllocPolicy>::FieldType::InputParamType FieldParamType; */

/* #ifdef USE_DESTRUCTIVE_FFT */
/*   typedef A2AvectorW<mf_Policies> Wtype; */
/*   typedef A2AvectorV<mf_Policies> Vtype; */
/* #else */
/*   typedef const A2AvectorW<mf_Policies> Wtype; */
/*   typedef const A2AvectorV<mf_Policies> Vtype; */
/* #endif */

/*   typedef A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> MesonFieldType; */
/*   typedef std::vector<MesonFieldType> MesonFieldVectorType; */
/*   typedef typename mf_Policies::ComplexType ComplexType; */
/*   typedef typename mf_Policies::SourcePolicies SourcePolicies; */


public:
  
  //Compute the two-point function using the pre-generated meson fields.
  //The pion is given the momentum associated with index 'pidx' in the RequiredMomentum
  //result is indexed by (tsrc, tsep)  where tsep is the source-sink separation
  template<typename PionMomentumPolicy>
  static void compute(fMatrix<typename mf_Policies::ScalarComplexType> &into, MesonFieldMomentumContainer<mf_Policies> &mf_ll_con, const PionMomentumPolicy &pion_mom, const int pidx){
    typedef typename mf_Policies::ComplexType ComplexType;
    typedef typename mf_Policies::ScalarComplexType ScalarComplexType;
    
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    into.resize(Lt,Lt);

    ThreeMomentum p_pi_src = pion_mom.getMesonMomentum(pidx); //exp(-ipx)
    ThreeMomentum p_pi_snk = -p_pi_src; //exp(+ipx)

    assert(mf_ll_con.contains(p_pi_src));
    assert(mf_ll_con.contains(p_pi_snk));
	   
    //Construct the meson fields
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_src = mf_ll_con.get(p_pi_src);
    std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_ll_snk = mf_ll_con.get(p_pi_snk);
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
    into *= ScalarComplexType(0.5,0);
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
