//Implementations of methods in a2a.h

//Set this object to be the fast Fourier transform of the input field
//Can optionally supply an object mode_preop that performs a transformation on each mode prior to the FFT
template< typename mf_Policies>
void A2AvectorVfftw<mf_Policies>::fft(const A2AvectorV<mf_Policies> &from, fieldOperation<FermionFieldType>* mode_preop){
  if(!UniqueID()){ printf("Doing V FFT\n"); fflush(stdout); }
  typedef typename FermionFieldType::InputParamType FieldParamType;
  FieldParamType field_setup = from.getMode(0).getDimPolParams();  
  FermionFieldType tmp(field_setup);
  
  Float preop_time = 0;
  Float gather_time = 0;
  Float fft_time = 0;
  Float scatter_time = 0;

  for(int mode=0;mode<nv;mode++){
    FermionFieldType const* init_gather_from = &from.getMode(mode);
    if(mode_preop != NULL){
      Float dtime = dclock();
      (*mode_preop)(from.getMode(mode),tmp);
      init_gather_from = &tmp;
      preop_time += dclock()-dtime;
    }    
    for(int mu=0;mu<3;mu++){
      Float dtime = dclock();

      //Gather
      CPSfermion4DglobalInOneDir<typename mf_Policies::ScalarComplexType> tmp_dbl(mu);
      tmp_dbl.gather( mu==0 ? *init_gather_from : tmp );
      gather_time += dclock()-dtime;

      //FFT
      dtime = dclock();      
      tmp_dbl.fft();
      fft_time += dclock()-dtime;      

      //Scatter
      dtime = dclock();
      tmp_dbl.scatter( mu==2 ? v[mode]: tmp );
      scatter_time += dclock()-dtime;
    }
  }
  if(!UniqueID()){ printf("Finishing V FFT\n"); fflush(stdout); }
  print_time("A2AvectorVfftw::fft","Preop",preop_time);
  print_time("A2AvectorVfftw::fft","gather",gather_time);
  print_time("A2AvectorVfftw::fft","FFT",fft_time);
  print_time("A2AvectorVfftw::fft","scatter",scatter_time);
}

//Set this object to be the fast Fourier transform of the input field
//Can optionally supply an object mode_preop that performs a transformation on each mode prior to the FFT
template< typename mf_Policies>
void A2AvectorWfftw<mf_Policies>::fft(const A2AvectorW<mf_Policies> &from, fieldOperation<FermionFieldType>* mode_preop){
  typedef typename FermionFieldType::InputParamType FieldParamType;
  FieldParamType field_setup = from.getWh(0).getDimPolParams();  
  FermionFieldType tmp(field_setup), tmp2(field_setup);

  //Do wl
  for(int mode=0;mode<nl;mode++){
    FermionFieldType const* init_gather_from = &from.getWl(mode);
    if(mode_preop != NULL){
      (*mode_preop)(from.getWl(mode),tmp);
      init_gather_from = &tmp;
    }
    for(int mu=0;mu<3;mu++){
      CPSfermion4DglobalInOneDir<typename mf_Policies::ScalarComplexType> tmp_dbl(mu);
      tmp_dbl.gather( mu==0 ? *init_gather_from : tmp );
      tmp_dbl.fft();
      tmp_dbl.scatter( mu==2 ? wl[mode]: tmp );
    }
  }
  //Do wh. First we need to uncompact the spin/color index as this is acted upon by the operator
  for(int hit=0;hit<nhits;hit++){
    for(int sc=0;sc<12;sc++){ //spin/color dilution index
      from.getSpinColorDilutedSource(tmp2,hit,sc);
      FermionFieldType* init_gather_from = &tmp2;
      if(mode_preop != NULL){
	(*mode_preop)(tmp2,tmp);
	init_gather_from = &tmp;
      }    
      for(int mu=0;mu<3;mu++){
	CPSfermion4DglobalInOneDir<typename mf_Policies::ScalarComplexType> tmp_dbl(mu);
	tmp_dbl.gather( mu==0 ? *init_gather_from : tmp );
	tmp_dbl.fft();
	tmp_dbl.scatter( mu==2 ? wh[sc+12*hit] : tmp );
      }
    }
  }
}


//Generate the wh field. We store in a compact notation that knows nothing about any dilution we apply when generating V from this
//For reproducibility we want to generate the wh field in the same order that Daiqian did originally. Here nhit random numbers are generated for each site/flavor
template<typename complexFieldType, typename mf_Policies, typename complex_class>
struct _set_wh_random_impl{};

template<typename complexFieldType, typename mf_Policies>
struct _set_wh_random_impl<complexFieldType, mf_Policies, complex_double_or_float_mark>{
  static void doit(std::vector<complexFieldType> &wh, const RandomType &type, const int nhits){
    typedef typename complexFieldType::FieldSiteType FieldSiteType;
    LRG.SetInterval(1, 0);
    int sites = wh[0].nsites(), flavors = wh[0].nflavors();
    
    for(int i = 0; i < sites*flavors; ++i) {
      int flav = i / sites;
      int st = i % sites;
      
      LRG.AssignGenerator(st,flav);
      for(int j = 0; j < nhits; ++j) {
	FieldSiteType* p = wh[j].site_ptr(st,flav);
	RandomComplex<FieldSiteType>::rand(p,type,FOUR_D);
      }
    }
  }
};


template< typename mf_Policies>
void A2AvectorW<mf_Policies>::setWhRandom(const RandomType &type){
  _set_wh_random_impl<ComplexFieldType, mf_Policies, typename ComplexClassify<typename ComplexFieldType::FieldSiteType>::type>::doit(wh,type,nhits);
}

//Get the diluted source with index id.
//We use the same set of random numbers for each spin and dilution as we do not need to rely on stochastic cancellation to separate them
//For legacy reasons we use different random numbers for the two G-parity flavors, although this is not strictly necessary
template< typename mf_Policies>
template<typename TargetFermionFieldType>
void A2AvectorW<mf_Policies>::getDilutedSource(TargetFermionFieldType &into, const int dil_id) const{
  typedef FieldSiteType mf_Complex;
  typedef typename TargetFermionFieldType::FieldSiteType TargetComplex;
  const char* fname = "getDilutedSource(...)";
  int hit, tblock, spin_color, flavor;
  StandardIndexDilution stdidx(getArgs());  
  stdidx.indexUnmap(dil_id,hit,tblock,spin_color,flavor);
    
  VRB.Result(cname.c_str(), fname, "Generating random wall source %d = (%d, %d, %d, %d).\n    ", dil_id, hit, tblock, flavor, spin_color);
  int tblock_origt = tblock * args.src_width;

  into.zero();

  if(tblock_origt / GJP.TnodeSites() != GJP.TnodeCoor()){
    VRB.Result(cname.c_str(), fname, "Not on node\n    ");
    return;
  }

  int tblock_origt_lcl = tblock_origt % GJP.TnodeSites();
    
  int src_size = GJP.VolNodeSites()/GJP.TnodeSites() * args.src_width; //size of source in units of complex numbers
#pragma omp parallel for
  for(int i=0;i<src_size;i++){
    int x[4];
    int rem = i;
    x[0] = rem % GJP.XnodeSites(); rem /= GJP.XnodeSites();
    x[1] = rem % GJP.YnodeSites(); rem /= GJP.YnodeSites();
    x[2] = rem % GJP.ZnodeSites(); rem /= GJP.ZnodeSites();
    x[3] = tblock_origt_lcl + rem;

    TargetComplex *into_site = (TargetComplex*)(into.site_ptr(x,flavor) + spin_color);
    mf_Complex const* from_site = (mf_Complex*)wh[hit].site_ptr(x,flavor); //note same random numbers for each spin/color!
    *into_site = *from_site;
  }
}

//When gauge fixing prior to taking the FFT it is necessary to uncompact the wh field in the spin-color index, as these indices are acted upon by the gauge fixing
//(I suppose technically only the color indices need uncompacting; this might be considered as a future improvement)
template< typename mf_Policies>
void A2AvectorW<mf_Policies>::getSpinColorDilutedSource(FermionFieldType &into, const int hit, const int sc_id) const{
  const char* fname = "getSpinColorDilutedSource(...)";
  
  into.zero();

#pragma omp parallel for
  for(int i=0;i<wh[hit].nfsites();i++){ //same mapping, different site_size
    FieldSiteType &into_site = *(into.fsite_ptr(i) + sc_id);
    const FieldSiteType &from_site = *(wh[hit].fsite_ptr(i));
    into_site = from_site;
  }
}

template<typename mf_Policies>
void randomizeVW(A2AvectorV<mf_Policies> &V, A2AvectorW<mf_Policies> &W){
  typedef typename mf_Policies::FermionFieldType FermionFieldType;
  typedef typename mf_Policies::ComplexFieldType ComplexFieldType;
  
  int nl = V.getNl();
  int nh = V.getNh(); //number of fully diluted high-mode indices
  int nhit = V.getNhits();
  assert(nl == W.getNl());
  assert(nh == W.getNh());
  assert(nhit == W.getNhits());
  

  std::vector<FermionFieldType> wl(nl);
  for(int i=0;i<nl;i++) wl[i].setUniformRandom();
  
  std::vector<FermionFieldType> vl(nl);
  for(int i=0;i<nl;i++) vl[i].setUniformRandom();
  
  std::vector<ComplexFieldType> wh(nhit);
  for(int i=0;i<nhit;i++) wh[i].setUniformRandom();
  
  std::vector<FermionFieldType> vh(nh);
  for(int i=0;i<nh;i++) vh[i].setUniformRandom();
    
  for(int i=0;i<nl;i++){
    V.importVl(vl[i],i);
    W.importWl(wl[i],i);
  }

  for(int i=0;i<nh;i++)
    V.importVh(vh[i],i);
  
  for(int i=0;i<nhit;i++)
    W.importWh(wh[i],i);
}
