#pragma once

#include "field_vectors.h"

CPS_START_NAMESPACE

template<typename A2Apolicies>
class A2AhighModeSource{
public:
  typedef typename A2Apolicies::GridFermionField GridFermionFieldD;

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  virtual void setHighModeSources(A2AvectorW<A2Apolicies> &into) const = 0;

  //Get the 4D source vector that we will invert upon
  virtual void get4DinverseSource(GridFermionFieldD &src, const int high_mode_idx, const A2AvectorW<A2Apolicies> &W) const{
    CPSfermion4D<typename A2Apolicies::ComplexTypeD,typename A2Apolicies::FermionFieldType::FieldMappingPolicy, 
		 typename A2Apolicies::FermionFieldType::FieldAllocPolicy> v4dfield(W.getFieldInputParams());
    W.getDilutedSource(v4dfield, high_mode_idx);

    //Export to Grid field
    v4dfield.exportGridField(src);
  }
  
  //Perform any post-inverse operations required upon the solutions
  virtual void solutionPostOp(A2AvectorV<A2Apolicies> &into) const{}

  virtual ~A2AhighModeSource(){}
};

//For reproducibility we want to generate the wh field in the same order that Daiqian did originally. Here nhit random numbers are generated for each site/flavor
//We use the same set of random numbers for each spin and dilution as we do not need to rely on stochastic cancellation to separate them
//For legacy reasons we use different random numbers for the two G-parity flavors, although this is not strictly necessary
template<typename A2Apolicies>
class A2AhighModeSourceOriginal: public A2AhighModeSource<A2Apolicies>{
public:

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (original)" << std::endl;
    typedef typename A2AvectorW<A2Apolicies>::ScalarComplexFieldType ScalarComplexFieldType;
    typedef typename ScalarComplexFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarComplexFieldType> tmp(nhits,null_obj);

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarComplexFieldType> views(HostWrite,tmp);
    for(size_t i = 0; i < sites*flavors; ++i) {
      int flav = i / sites;
      size_t st = i % sites;
      
      LRG.AssignGenerator(st,flav);
      for(int j = 0; j < nhits; ++j) {
	FieldSiteType* p = views[j].site_ptr(st,flav);
	RandomComplex<FieldSiteType>::rand(p,rand_type,FOUR_D);
      }
    }
    views.free();

    into.setWh(tmp);
  }    


};

template<typename A2Apolicies>
class A2AhighModeSourceXconj: public A2AhighModeSource<A2Apolicies>{
public:
  typedef typename A2Apolicies::GridFermionField GridFermionFieldD;
  typedef CPSspinMatrix<cps::ComplexD> SpinMat;
private:
  SpinMat Pplus;
  SpinMat Pminus;
  SpinMat mXPplus;
  SpinMat mXPminus;
  SpinMat PplusX;
  SpinMat PminusX;
public:  

  A2AhighModeSourceXconj(){
    SpinMat C; C.unit(); C.gl(1).gl(3); //C=-gY gT = gT gY
    SpinMat X = C; X.gr(-5);
    SpinMat one; one.unit();
    cps::ComplexD _i(0,1);
    Pplus = 0.5*(one + _i*X);
    Pminus = 0.5*(one - _i*X);
    mXPplus = -X*Pplus;
    mXPminus = -X*Pminus;
    PplusX = Pplus * X;
    PminusX = Pminus * X;
  }

  //The high modes sources are diagonal in flavor space but the two diagonal elements are complex conjugate pairs rather than independent
  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (flavor-conjugate)" << std::endl;
    typedef typename A2AvectorW<A2Apolicies>::ScalarComplexFieldType ScalarComplexFieldType;
    typedef typename ScalarComplexFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarComplexFieldType> tmp(nhits,null_obj);

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();

    ViewArray<ScalarComplexFieldType> views(HostWrite,tmp);

    for(size_t st = 0; st < sites; ++st) {
      LRG.AssignGenerator(st,0);
      for(int j = 0; j < nhits; ++j) {
	FieldSiteType* p = views[j].site_ptr(st,0);
	RandomComplex<FieldSiteType>::rand(p,rand_type,FOUR_D);
	*views[j].site_ptr(st,1) = Grid::conjugate(*p);
      }
    }
    views.free();

    into.setWh(tmp);
  }    

  //Get the 4D source vector that we will invert upon
  //We must transform into X-conjugate form, and revert this post-facto for the solutions
  void get4DinverseSource(GridFermionFieldD &src, const int high_mode_idx, const A2AvectorW<A2Apolicies> &W) const override{
    double time = -dclock();
    LOGA2A << "Starting construction of X-conjugate source for high mode " << high_mode_idx << std::endl;
    
    typedef typename A2Apolicies::ComplexTypeD ComplexType;
    typedef CPSfermion4D<typename A2Apolicies::ComplexTypeD,typename A2Apolicies::FermionFieldType::FieldMappingPolicy, 
			 typename A2Apolicies::FermionFieldType::FieldAllocPolicy> CPSfermionField;

    CPSfermionField src_in(W.getFieldInputParams());
    W.getDilutedSource(src_in, high_mode_idx);

    StandardIndexDilution stdidx(W.getArgs());  
    int dhit,dtblock,dspin_color,dflavor;
    stdidx.indexUnmap(high_mode_idx,dhit,dtblock,dspin_color,dflavor);
  
    int dspin = dspin_color / 3; //c+3*s
    int dcolor = dspin_color % 3;

    //Current form is a column of
    //(rho I_12x12 , 0_12x12        )
    //(  0_12x12   ,  rho* I_12x12  )
    //where we are showing the flavor structure
    
    //We need to transform to the equivalent column of
    //(rho P_+       ,   rho P_-     )
    //( -X rho* P_-  ,  -X rho* P_+  )

    //Loop over sites only on timeslice
    assert(src_in.nodeSites(3) == GJP.TnodeSites());
    size_t size_3d = src_in.nsites() / src_in.nodeSites(3); //assumes no simd in time direction

    CPSfermionField src_out(W.getFieldInputParams());
    src_out.zero();

    int tsrc_width = W.getArgs().src_width;
    int tglob_start = dtblock * tsrc_width;
    int tglob_lessthan = (dtblock + 1) * tsrc_width;

    {
      CPSautoView(src_in_v,src_in,HostRead);
      CPSautoView(src_out_v,src_out,HostReadWrite); 
    
      for(int tglob=tglob_start; tglob < tglob_lessthan; tglob++){
	int tlcl = tglob - GJP.TnodeCoor()*GJP.TnodeSites();
	if(tlcl >=0 && tlcl < GJP.TnodeSites()){
#pragma omp parallel for
	  for(size_t x3d=0;x3d<size_3d;x3d++){
	    size_t x4d = src_in.threeToFour(x3d,tlcl);
	    ComplexType* v_f0_p = src_in_v.site_ptr(x4d,0) + dspin_color ;
	    ComplexType* v_f1_p = src_in_v.site_ptr(x4d,1) + dspin_color ;
	    ComplexType* o_f0_p = src_out_v.site_ptr(x4d,0) ;
	    ComplexType* o_f1_p = src_out_v.site_ptr(x4d,1) ;

	    if(dflavor == 0){
	      ComplexType rho = *v_f0_p;
	      for(int s=0;s<4;s++){
		int sc = dcolor + 3*s; //still Kronecker delta in color
		*( o_f0_p + sc ) = rho * Pplus(s,dspin);
		*( o_f1_p + sc ) = conjugate(rho) * mXPminus(s,dspin);
	      }
	    }else{
	      ComplexType rhostar = *v_f1_p;
	      for(int s=0;s<4;s++){
		int sc = dcolor + 3*s;
		*( o_f0_p + sc ) = conjugate(rhostar) * Pminus(s,dspin);
		*( o_f1_p + sc ) = rhostar * mXPplus(s,dspin);
	      }
	    }
	  }
	}
      }
    }
    
    src_out.exportGridField(src);

    {
      LOGA2A << "Check result is X-conjugate" << std::endl;
      static Grid::Gamma X = Grid::Gamma(Grid::Gamma::Algebra::MinusGammaY) * Grid::Gamma(Grid::Gamma::Algebra::GammaT)*Grid::Gamma(Grid::Gamma::Algebra::Gamma5);
      auto tmp = Grid::PeekIndex<GparityFlavourIndex>(src,0);
      tmp = -(X*conjugate(tmp)) - Grid::PeekIndex<GparityFlavourIndex>(src,1);
      double n = norm2(tmp);
      if(n>1e-8) ERR.General("A2AhighModeSourceXconj","get4DinverseSource","Source %d is not X-conjugate: got %g expect 0",high_mode_idx,n);
    }

    a2a_print_time("A2AhighModeSourceXconj","get4DinverseSource",time + dclock());
  }

  void solutionPostOp(A2AvectorV<A2Apolicies> &into) const override{
    double time = -dclock();
    LOGA2A << "Starting reconstruction of G-parity solutions from X-conjugate inverses" << std::endl;
	
    //V = M^-1 (rho P_+       ,   rho P_-     )
    //         ( -X rho* P_-  ,  -X rho* P_+  )
    
    //Need to compute
    //V' = V ( P_+ ,  P_-X ) = ( V_11 V_12 ) ( P_+ ,  P_-X ) = ( V_11 P_+ + V_12 P_-    ,    V_11 P_-X + V_12 P_+X )
    //       ( P_- ,  P_+X )   ( V_21 V_22 ) ( P_- ,  P_+X )   ( V_21 P_+ + V_22 P_-    ,    V_21 P_-X + V_22 P_+X )

    //Full flavor,spin,color indices
    //V'_{11,ss'',cc'} =  V_{11,ss',cc'} P+_{s's''} +  V_{12,ss',cc'} P-_{s's''}
    //V'_{12,ss'',cc'} =  V_{11,ss',cc'} [P_-X]_{s's''} +  V_{12,ss',cc'} [P_+X]_{s's''}

    //V'_{21,ss'',cc'} =  V_{21,ss',cc'} P+_{s's''} +  V_{22,ss',cc'} P-_{s's''}
    //V'_{22,ss'',cc'} =  V_{21,ss',cc'} [P_-X]_{s's''} +  V_{22,ss',cc'} [P_+X]_{s's''}

    //i.e.
    //V'_{f1,ss'',cc'} =  V_{f1,ss',cc'} P+_{s's''} +  V_{f2,ss',cc'} P-_{s's''}
    //V'_{f2,ss'',cc'} =  V_{f1,ss',cc'} [P_-X]_{s's''} +  V_{f2,ss',cc'} [P_+X]_{s's''}

    typedef typename A2AvectorV<A2Apolicies>::FermionFieldType FermionField;

    typename FermionField::View* Vptrs[4][2];
    
    int nl = into.getNl();
    size_t fsize = into.getMode(0).size();
    typedef typename FermionField::FieldSiteType SiteType;

    CPSautoView(into_v,into,HostReadWrite);
    
    StandardIndexDilution stdidx(into.getArgs());
    //Loop over column indices that are not being manipulated
    for(int hit=0;hit<stdidx.getNhits();hit++){
      for(int tcol=0;tcol<stdidx.getNtBlocks();tcol++){
	for(int ccol=0;ccol<3;ccol++){

	  //Store locations of appropriate input fields
	  for(int scol=0;scol<4;scol++)
	    for(int fcol=0;fcol<2;fcol++)
	      Vptrs[scol][fcol] = &into_v.getMode(nl + stdidx.indexMap(hit,tcol,ccol+3*scol,fcol) );
	  	    
#pragma omp parallel for
	  for(size_t i=0;i<fsize;i++){ //units of complex number,  loops over sites and f,s,c row indices
	    SiteType Vs[4][2]; //store temp copies of inputs
	    for(int sp=0;sp<4;sp++)
	      for(int fp=0;fp<2;fp++)
		Vs[sp][fp] = Vptrs[sp][fp]->ptr()[i];
	      
	    for(int spp=0;spp<4;spp++){
	      SiteType &V1s = Vptrs[spp][0]->ptr()[i];
	      SiteType &V2s = Vptrs[spp][1]->ptr()[i];
	      V1s=V2s=0;

	      //V'_{f1,ss'',cc'} =  V_{f1,ss',cc'} P+_{s's''} +  V_{f2,ss',cc'} P-_{s's''}
	      //V'_{f2,ss'',cc'} =  V_{f1,ss',cc'} [P_-X]_{s's''} +  V_{f2,ss',cc'} [P_+X]_{s's''}
	      for(int sp=0;sp<4;sp++){
		V1s += Vs[sp][0] * Pplus(sp,spp) + Vs[sp][1] * Pminus(sp,spp);
		V2s += Vs[sp][0] * PminusX(sp,spp) + Vs[sp][1] * PplusX(sp,spp);
	      }
	    }
	  }
	}//ccol
      }//tcol
    }//hit

    a2a_print_time("A2AhighModeSourceXconj","solutionPostOp",time + dclock());
  }
	  
	  


};


//This version uses the same random numbers for both diagonal flavor elements so the flavor structure is always the unit matrix
template<typename A2Apolicies>
class A2AhighModeSourceFlavorUnit: public A2AhighModeSource<A2Apolicies>{
public:

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (original)" << std::endl;
    typedef typename A2AvectorW<A2Apolicies>::ScalarComplexFieldType ScalarComplexFieldType;
    typedef typename ScalarComplexFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarComplexFieldType> tmp(nhits,null_obj);

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarComplexFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) {
      LRG.AssignGenerator(st,0);
      for(int j = 0; j < nhits; ++j) {
	FieldSiteType* p = views[j].site_ptr(st,0);
	RandomComplex<FieldSiteType>::rand(p,rand_type,FOUR_D);
	*views[j].site_ptr(st,1) = *p;
      }
    }
    views.free();

    into.setWh(tmp);
  }    


};

//This version uses a random number and its complex conjugate as the diagonal entries, matching the X-conjugate source but without the intermediate rotation
template<typename A2Apolicies>
class A2AhighModeSourceFlavorCConj: public A2AhighModeSource<A2Apolicies>{
public:

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (original)" << std::endl;
    typedef typename A2AvectorW<A2Apolicies>::ScalarComplexFieldType ScalarComplexFieldType;
    typedef typename ScalarComplexFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarComplexFieldType> tmp(nhits,null_obj);

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarComplexFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) {
      LRG.AssignGenerator(st,0);
      for(int j = 0; j < nhits; ++j) {
	FieldSiteType* p = views[j].site_ptr(st,0);
	RandomComplex<FieldSiteType>::rand(p,rand_type,FOUR_D);
	*views[j].site_ptr(st,1) = Grid::conjugate(*p);
      }
    }
    views.free();

    into.setWh(tmp);
  }    


};




CPS_END_NAMESPACE
