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

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  virtual void setHighModeSources(A2AvectorWunitary<A2Apolicies> &into) const{ ERR.General("A2AhighModeSource","setHighModeSources","Not implemented for A2AvectorWunitary"); }

  //Get the 4D source vector that we will invert upon
  virtual void get4DinverseSource(GridFermionFieldD &src, const int high_mode_idx, const A2AvectorWunitary<A2Apolicies> &W) const{
    CPSfermion4D<typename A2Apolicies::ComplexTypeD,typename A2Apolicies::FermionFieldType::FieldMappingPolicy, 
		 typename A2Apolicies::FermionFieldType::FieldAllocPolicy> v4dfield(W.getFieldInputParams());
    W.getDilutedSource(v4dfield, high_mode_idx);

    //Export to Grid field
    v4dfield.exportGridField(src);
  }


  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  virtual void setHighModeSources(A2AvectorWtimePacked<A2Apolicies> &into) const{ ERR.General("A2AhighModeSource","setHighModeSources","Not implemented for A2AvectorWtimePacked"); }

  //Get the 4D source vector that we will invert upon
  virtual void get4DinverseSource(GridFermionFieldD &src, const int high_mode_idx, const A2AvectorWtimePacked<A2Apolicies> &W) const{
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

  void setHighModeSources(A2AvectorWunitary<A2Apolicies> &into) const override{ 
    LOGA2A << "Setting high-mode sources (original) for Wunitary" << std::endl;
    typedef typename A2AvectorWunitary<A2Apolicies>::ScalarComplexFieldType ScalarComplexFieldType;
    typedef typename ScalarComplexFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarComplexFieldType> tmp(into.getNhighModes(),null_obj);
    for(int i=0;i<tmp.size();i++) tmp[i].zero();

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarComplexFieldType> views(HostWrite,tmp);
    for(size_t i = 0; i < sites*flavors; ++i) {
      int flav = i / sites;
      size_t st = i % sites;
      
      LRG.AssignGenerator(st,flav);

      //Flavor structure
      //| rnd_1  0   |
      //|  0    rnd2 |

      for(int j = 0; j < nhits; ++j) {
	FieldSiteType* p = views[into.indexMap(j,flav)].site_ptr(st,flav);
	RandomComplex<FieldSiteType>::rand(p,rand_type,FOUR_D);
      }
    }
    views.free();

    into.setWh(tmp);
  }    

  void setHighModeSources(A2AvectorWtimePacked<A2Apolicies> &into) const override{ 
    LOGA2A << "Setting high-mode sources (original) for WtimePacked" << std::endl;
    typedef typename A2AvectorWtimePacked<A2Apolicies>::ScalarFermionFieldType ScalarFermionFieldType;
    typedef typename ScalarFermionFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarFermionFieldType> tmp(into.getNhighModes(),null_obj);
    for(int i=0;i<tmp.size();i++) tmp[i].zero();

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarFermionFieldType> views(HostWrite,tmp);
    for(size_t i = 0; i < sites*flavors; ++i) {
      int flav = i / sites;
      size_t st = i % sites;
      
      LRG.AssignGenerator(st,flav);

      //Flavor/spin/color structure
      //| rnd_1 I_sc       0     |
      //|  0           rnd2 I_sc |
      //where I_sc is a 12x12 unit matrix

      for(int j = 0; j < nhits; ++j) {
	FieldSiteType v; RandomComplex<FieldSiteType>::rand(&v,rand_type,FOUR_D);
	for(int sc=0;sc<12;sc++){
	  FieldSiteType* p = views[into.indexMap(j,sc,flav)].site_ptr(st,flav) + sc; //diagonal spin-color structure
	  *p = v;
	}
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
    LOGA2A << "Setting high-mode sources (Xconj)" << std::endl;
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
    LOGA2A << "Setting high-mode sources (flavor unit)" << std::endl;
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

  void setHighModeSources(A2AvectorWtimePacked<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (flavor unit) for WtimePacked" << std::endl;
    assert(GJP.Gparity());
    typedef typename A2AvectorWtimePacked<A2Apolicies>::ScalarFermionFieldType ScalarFermionFieldType;
    typedef typename ScalarFermionFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarFermionFieldType> tmp(into.getNhighModes(),null_obj);
    for(int i=0;i<tmp.size();i++) tmp[i].zero();

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarFermionFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) { //use different random numbers for each timeslice for convenience
      for(int j = 0; j < nhits; ++j) {
	FieldSiteType u;
	LRG.AssignGenerator(st,0);
	RandomComplex<FieldSiteType>::rand(&u,rand_type,FOUR_D);

	for(int srow=0;srow<4;srow++){
	  for(int scol=0;scol<4;scol++){
	    for(int c=0;c<3;c++){
	      int scrow = c + 3*srow,  sccol = c + 3*scol;     	      
	      for(int f=0;f<flavors;f++){ //diagonal flavor
		*( views[into.indexMap(j,sccol,f)].site_ptr(st,f) + scrow ) = (srow == scol ? u : 0.);
	      }
	    }
	  }
	}
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
  //Flavor structure
  //| rnd_1   0   |
  //|  0    rnd_1*|

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (flavor cconj)" << std::endl;
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


template<typename A2Apolicies>
class A2AhighModeSourceFlavorUnitary: public A2AhighModeSource<A2Apolicies>{
public:
  //Flavor structure
  //| rnd_1   rnd_2 |
  //| -rnd_2* rnd_1*|

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorWunitary<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (flavor unitary) for Wunitary" << std::endl;
    typedef typename A2AvectorWunitary<A2Apolicies>::ScalarComplexFieldType ScalarComplexFieldType;
    typedef typename ScalarComplexFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarComplexFieldType> tmp(into.getNhighModes(),null_obj);

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarComplexFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) {
      for(int j = 0; j < nhits; ++j) {
	FieldSiteType rnd_1, rnd_2;
	LRG.AssignGenerator(st,0);
	RandomComplex<FieldSiteType>::rand(&rnd_1,rand_type,FOUR_D);
	LRG.AssignGenerator(st,1);
	RandomComplex<FieldSiteType>::rand(&rnd_2,rand_type,FOUR_D);

	rnd_1 = rnd_1 / sqrt(2.);
	rnd_2 = rnd_2 / sqrt(2.);
	
	int col_f0 = into.indexMap(j,0), col_f1 = into.indexMap(j,1);
	*views[col_f0].site_ptr(st,0) = rnd_1;	
	*views[col_f0].site_ptr(st,1) = -cconj(rnd_2);
	*views[col_f1].site_ptr(st,0) = rnd_2;	
	*views[col_f1].site_ptr(st,1) = cconj(rnd_1);
      }
    }
    views.free();

    into.setWh(tmp);
  }    

  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{ 
    ERR.General("A2AhighModeSourceFlavorUnitary","setHighModeSources(A2AvectorW)", "Invalid W species");
  }    

};


template<typename A2Apolicies>
class A2AhighModeSourceFlavorRotY: public A2AhighModeSource<A2Apolicies>{
public:
  //Flavor structure
  //exp(i \theta \sigma_2) = \cos \theta + i\sigma_2 \sin \theta
  //|  c    s |
  //| -s   c  |

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorWunitary<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (flavor-Y rotation) for Wunitary" << std::endl;
    typedef typename A2AvectorWunitary<A2Apolicies>::ScalarComplexFieldType ScalarComplexFieldType;
    typedef typename ScalarComplexFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarComplexFieldType> tmp(into.getNhighModes(),null_obj);

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarComplexFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) {
      for(int j = 0; j < nhits; ++j) {
	//To generate angle, use a U(1) random number and obtain the phase angle
	FieldSiteType u;
	LRG.AssignGenerator(st,0);
	RandomComplex<FieldSiteType>::rand(&u,UONE,FOUR_D);
	
	//u  =  cu + isu
	int col_f0 = into.indexMap(j,0), col_f1 = into.indexMap(j,1);
	*views[col_f0].site_ptr(st,0) = u.real();	
	*views[col_f0].site_ptr(st,1) = -u.imag();
	*views[col_f1].site_ptr(st,0) = u.imag();	
	*views[col_f1].site_ptr(st,1) = u.real();
      }
    }
    views.free();

    into.setWh(tmp);
  }    

  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{ 
    ERR.General("A2AhighModeSourceFlavorRotY","setHighModeSources(A2AvectorW)", "Invalid W species");
  }    

};



template<typename A2Apolicies>
class A2AhighModeSourceU1X: public A2AhighModeSource<A2Apolicies>{
private:
  typedef CPSspinMatrix<cps::ComplexD> SpinMat;
  SpinMat X;
public:
  //Flavor structure
  //|  c+sX  0    |
  //|   0   c+sX  |
  //c^2+s^2=1

  A2AhighModeSourceU1X(){
    SpinMat C; C.unit(); C.gl(1).gl(3); //C=-gY gT = gT gY
    X = C; X.gr(-5);
  }

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorWtimePacked<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (U1X) for WtimePacked" << std::endl;
    assert(GJP.Gparity());
    typedef typename A2AvectorWtimePacked<A2Apolicies>::ScalarFermionFieldType ScalarFermionFieldType;
    typedef typename ScalarFermionFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarFermionFieldType> tmp(into.getNhighModes(),null_obj);
    for(int i=0;i<tmp.size();i++) tmp[i].zero();

    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarFermionFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) { //use different random numbers for each timeslice for convenience
      for(int j = 0; j < nhits; ++j) {
	//To generate angle, use a U(1) random number and obtain the phase angle
	FieldSiteType u;
	LRG.AssignGenerator(st,0);
	RandomComplex<FieldSiteType>::rand(&u,UONE,FOUR_D);
	double C = u.real(), S = u.imag();	
	for(int srow=0;srow<4;srow++){
	  for(int scol=0;scol<4;scol++){
	    for(int c=0;c<3;c++){
	      int scrow = c + 3*srow,  sccol = c + 3*scol;     	      
	      for(int f=0;f<flavors;f++){ //diagonal flavor
		*( views[into.indexMap(j,sccol,f)].site_ptr(st,f) + scrow ) = (srow == scol ? C : 0.) + S * X(srow,scol);
	      }
	    }
	  }
	}
      }
    }
    views.free();

    into.setWh(tmp);
  }    

  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{ 
    ERR.General("A2AhighModeSourceU1X","setHighModeSources(A2AvectorW)", "Invalid W species");
  }    

};


template<typename A2Apolicies>
class A2AhighModeSourceU1g0: public A2AhighModeSource<A2Apolicies>{
private:
  typedef CPSspinMatrix<cps::ComplexD> SpinMat;
  SpinMat g0;
public:
  //Flavor structure
  //|  c+isg0  0    |
  //|   0   c+isg0  |
  //c^2+s^2=1
  //g0 = gamma_x

  A2AhighModeSourceU1g0(){
    g0.unit(); g0.gl(0);
  }

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorWtimePacked<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (U1g0) for WtimePacked" << std::endl;
    assert(GJP.Gparity());
    typedef typename A2AvectorWtimePacked<A2Apolicies>::ScalarFermionFieldType ScalarFermionFieldType;
    typedef typename ScalarFermionFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarFermionFieldType> tmp(into.getNhighModes(),null_obj);
    for(int i=0;i<tmp.size();i++) tmp[i].zero();

    FieldSiteType zero(0.);
    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarFermionFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) { //use different random numbers for each timeslice for convenience
      for(int j = 0; j < nhits; ++j) {
	//To generate angle, use a U(1) random number and obtain the phase angle
	FieldSiteType u;
	LRG.AssignGenerator(st,0);
	RandomComplex<FieldSiteType>::rand(&u,UONE,FOUR_D);
	FieldSiteType C = u.real(), iS = FieldSiteType(0,1)*u.imag();
	for(int srow=0;srow<4;srow++){
	  for(int scol=0;scol<4;scol++){
	    for(int c=0;c<3;c++){
	      int scrow = c + 3*srow,  sccol = c + 3*scol;     	      
	      for(int f=0;f<flavors;f++){ //diagonal flavor
		*( views[into.indexMap(j,sccol,f)].site_ptr(st,f) + scrow ) = (srow == scol ? C : zero) + iS * FieldSiteType(g0(srow,scol));
	      }
	    }
	  }
	}
      }
    }
    views.free();

    into.setWh(tmp);
  }    

  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{ 
    ERR.General("A2AhighModeSourceU1g0","setHighModeSources(A2AvectorW)", "Invalid W species");
  }    

};


template<typename A2Apolicies>
class A2AhighModeSourceU1H: public A2AhighModeSource<A2Apolicies>{
private:
  typedef CPSspinMatrix<cps::ComplexD> SpinMat;

  SpinMat Pplus;
  SpinMat Pminus;
  SpinMat mXPplus;
  SpinMat mXPminus;
public:  

  //Flavor structure
  //|  rho P_+         rho P_-      |
  //|   -X rho* P_-    -X rho* P_+  |
  //rho \in U(1), Z2, whatever

  //Alternate P_+, P_- pattern for odd/even parity sites to kill some noise!

  A2AhighModeSourceU1H(){
   SpinMat C; C.unit(); C.gl(1).gl(3); //C=-gY gT = gT gY
    SpinMat X = C; X.gr(-5);
    SpinMat one; one.unit();
    cps::ComplexD _i(0,1);
    Pplus = 0.5*(one + _i*X);
    Pminus = 0.5*(one - _i*X);
    mXPplus = -X*Pplus;
    mXPminus = -X*Pminus;
  }

  //Set the high mode sources. The input vector will be resized to the number of hits prior to this call
  void setHighModeSources(A2AvectorWtimePacked<A2Apolicies> &into) const override{
    LOGA2A << "Setting high-mode sources (U1H) for WtimePacked" << std::endl;
    assert(GJP.Gparity());
    typedef typename A2AvectorWtimePacked<A2Apolicies>::ScalarFermionFieldType ScalarFermionFieldType;
    typedef typename ScalarFermionFieldType::FieldSiteType FieldSiteType;
    NullObject null_obj;
    int nhits = into.getNhits();
    RandomType rand_type = into.getArgs().rand_type;
    std::vector<ScalarFermionFieldType> tmp(into.getNhighModes(),null_obj);
    for(int i=0;i<tmp.size();i++) tmp[i].zero();

    ComplexD zero(0.);
    LRG.SetInterval(1, 0);
    size_t sites = tmp[0].nsites(), flavors = tmp[0].nflavors();
    ViewArray<ScalarFermionFieldType> views(HostWrite,tmp);
    for(size_t st = 0; st < sites; ++st) { //use different random numbers for each timeslice for convenience
      int x[4];
      tmp[0].siteUnmap(st,x);
      int parity = (x[0] + x[1] + x[2]) % 2; //checkerboard pattern in 3D space
      SpinMat const &Pa = parity == 0 ? Pplus : Pminus;
      SpinMat const &Pb = parity == 0 ? Pminus : Pplus;
      SpinMat const &mXPa = parity == 0 ? mXPplus : mXPminus;
      SpinMat const &mXPb = parity == 0 ? mXPminus : mXPplus;

      for(int j = 0; j < nhits; ++j) {
	FieldSiteType rho;
	LRG.AssignGenerator(st,0);
	RandomComplex<FieldSiteType>::rand(&rho,rand_type,FOUR_D);

	FieldSiteType rhostar = Grid::conjugate(rho);

	for(int srow=0;srow<4;srow++){
	  for(int scol=0;scol<4;scol++){
	    for(int c=0;c<3;c++){ //color diagonal
	      int scrow = c + 3*srow,  sccol = c + 3*scol;     	      
	        //|  rho P_a         rho P_b      |
   	        //|   -X rho* P_b    -X rho* P_a  |

	      //0,0
	      *( views[into.indexMap(j,sccol,0)].site_ptr(st,0) + scrow ) = rho * FieldSiteType(Pa(srow,scol));  //rho_Pa_ss; //rho * Pa_ss;  //Pa(srow,scol);
	      //0,1
	      *( views[into.indexMap(j,sccol,1)].site_ptr(st,0) + scrow ) = rho * FieldSiteType(Pb(srow,scol));
	      //1,0
	      *( views[into.indexMap(j,sccol,0)].site_ptr(st,1) + scrow ) = rhostar * FieldSiteType(mXPb(srow,scol));
	      //1,1
	      *( views[into.indexMap(j,sccol,1)].site_ptr(st,1) + scrow ) = rhostar * FieldSiteType(mXPa(srow,scol));
	    }
	  }
	}
      }
    }
    views.free();

    into.setWh(tmp);
  }    

  void setHighModeSources(A2AvectorW<A2Apolicies> &into) const override{ 
    ERR.General("A2AhighModeSourceU1H","setHighModeSources(A2AvectorW)", "Invalid W species");
  }    

};


template<typename Policies>
inline A2AhighModeSource<Policies>* highModeSourceFactory(A2AhighModeSourceType src){
  switch(src){
  case A2AhighModeSourceTypeOrig:
    return new A2AhighModeSourceOriginal<Policies>();
  case A2AhighModeSourceTypeXconj:
    return new A2AhighModeSourceXconj<Policies>();
  case A2AhighModeSourceTypeFlavorUnit:
    return new A2AhighModeSourceFlavorUnit<Policies>();
  case A2AhighModeSourceTypeFlavorCConj:
    return new A2AhighModeSourceFlavorCConj<Policies>();
  case A2AhighModeSourceTypeFlavorUnitary:
    return new A2AhighModeSourceFlavorUnitary<Policies>();
  case A2AhighModeSourceTypeFlavorRotY:
    return new A2AhighModeSourceFlavorRotY<Policies>();
  case A2AhighModeSourceTypeU1X:
    return new A2AhighModeSourceU1X<Policies>();
  case A2AhighModeSourceTypeU1g0:
    return new A2AhighModeSourceU1g0<Policies>();
  case A2AhighModeSourceTypeU1H:
    return new A2AhighModeSourceU1H<Policies>();
  default:
    assert(0);
  }
}

CPS_END_NAMESPACE