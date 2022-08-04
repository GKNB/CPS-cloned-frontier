#ifndef _COMPUTE_KTOPIPI_BASE
#define _COMPUTE_KTOPIPI_BASE

#include<alg/a2a/lattice.h>

CPS_START_NAMESPACE

//Results containers and other helper classes for K->pipi

//Lt * Lt * 8 * ncontract tensor  (option for multiple independent threads)
//Here 8 is the number of combinations of spin-color-flavor matrix pairs (see below for indexing)

//#define KTOPIPI_RESULTSCONTAINER_BIG_ALLOC

template<typename ComplexType, typename AllocPolicy>
struct _resultsContainerBase{
#ifdef KTOPIPI_RESULTSCONTAINER_BIG_ALLOC
  typedef basicComplexArray<ComplexType,AllocPolicy> type; //data region for each thread is part of a single allocation
#else
  typedef basicComplexArraySplitAlloc<ComplexType,AllocPolicy> type; //data region for each thread is allocated independently
#endif
};

template<typename ComplexType, typename AllocPolicy>
class KtoPiPiGparityResultsContainer: public _resultsContainerBase<ComplexType,AllocPolicy>::type{
  typedef typename _resultsContainerBase<ComplexType,AllocPolicy>::type baseClass;

  int Lt;
  int ncontract;
  
  //gcombidx \in {0..7}, cf. below
  inline int map(const int tk, const int t_dis, const int con_idx, const int gcombidx) const{
    return con_idx + ncontract*( gcombidx + 8*( t_dis + Lt*tk ) );
  }
public:
  inline static size_t byte_size(const int ncontract, const int nthread){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    return ncontract * 8 * Lt * Lt * nthread * sizeof(ComplexType);
  }

  void resize(const int _ncontract, const int _nthread=1){
    Lt = GJP.Tnodes()*GJP.TnodeSites();
    ncontract = _ncontract;

    int thread_size = ncontract * 8 * Lt * Lt;
    this->baseClass::resize(thread_size, _nthread);
  }
  int getNcontract() const{ return ncontract; }

  KtoPiPiGparityResultsContainer(): baseClass(){  }
  KtoPiPiGparityResultsContainer(const int _ncontract, const int _nthread = 1): baseClass(){
    resize(_ncontract,_nthread);
  }

  inline ComplexType & operator()(const int tk, const int t_dis, const int con_idx, const int gcombidx, const int thread = 0){
    return this->baseClass::operator()(map(tk,t_dis,con_idx,gcombidx),thread);
  }
  inline const ComplexType & operator()(const int tk, const int t_dis, const int con_idx, const int gcombidx, const int thread = 0) const{
    return this->baseClass::operator()(map(tk,t_dis,con_idx,gcombidx),thread);
  }

  KtoPiPiGparityResultsContainer & operator*=(const Float f){
    for(int t=0;t<this->nThreads();t++)
      for(int i=0;i<this->nElementsPerThread();i++) this->baseClass::operator()(i,t) = this->baseClass::operator()(i,t) * f;
    return *this;
  }
  // //Daiqian's loops are in the following order (outer->inner): tk, tdis, mu, con_idx, gcombidx
  // int off = gcombidx + 8* cidx + (8*n_contract + dq_extrafig)*(tk * Lt + tdis); //Daiqian puts the sbar g5 d diagrams after the usual combination block for type3/type4. For these set dq_extrafig to 2

  //hexfloat option: For reproducibility testing, write the output in hexfloat format rather than truncating the precision
  void write(const std::string &filename, const bool hexfloat = false) const{
    const char* fmt = hexfloat ? "%a %a " : "%.16e %.16e ";
    
    FILE *p;
    if((p = Fopen(filename.c_str(),"w")) == NULL)
      ERR.FileA("KtoPiPiGparityResultsContainer","write",filename.c_str());
    for(int tk=0;tk<Lt;tk++){
      for(int tdis=0;tdis<Lt;tdis++){
	Fprintf(p,"%d %d ", tk, tdis);
	for(int cidx=0; cidx<ncontract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    std::complex<Float> dp = convertComplexD((*this)(tk,tdis,cidx,gcombidx));
	    Fprintf(p,fmt,std::real(dp),std::imag(dp));
	  }
	}
	Fprintf(p,"\n");
      }
    }	
    Fclose(p);
  }
};

//Lt * Lt * 2 tensor  (option for multiple independent threads)
//Here 2 is the number of differen spin-color-flavor matrix insertions (F_0 g5  and  -F_1 g5)
template<typename ComplexType, typename AllocPolicy>
class KtoPiPiGparityMixDiagResultsContainer: public _resultsContainerBase<ComplexType,AllocPolicy>::type{
  typedef typename _resultsContainerBase<ComplexType,AllocPolicy>::type baseClass;
  int Lt;
  
  //fidx \in {0..1}, as above
  inline int map(const int tk, const int t_dis, const int fidx) const{
    return fidx + 2*( t_dis + Lt*tk );
  }
public:
  inline static size_t byte_size(const int nthread){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    return 2 * Lt * Lt * nthread * sizeof(ComplexType);
  }

  void resize(const int _nthread=1){
    Lt = GJP.Tnodes()*GJP.TnodeSites();
    int thread_size = 2 * Lt * Lt;
    this->baseClass::resize(thread_size, _nthread);
  }

  KtoPiPiGparityMixDiagResultsContainer(): baseClass(){
    resize(1);
  }
  KtoPiPiGparityMixDiagResultsContainer(const int _nthread): baseClass(){
    resize(_nthread);
  }

  inline ComplexType & operator()(const int tk, const int t_dis, const int fidx, const int thread = 0){
    return this->baseClass::operator()(map(tk,t_dis,fidx),thread);
  }
  inline const ComplexType & operator()(const int tk, const int t_dis, const int fidx, const int thread = 0) const{
    return this->baseClass::operator()(map(tk,t_dis,fidx),thread);
  }

  KtoPiPiGparityMixDiagResultsContainer<ComplexType,AllocPolicy> & operator*=(const Float &f){
    for(int t=0;t<this->nThreads();t++)
      for(int i=0;i<this->nElementsPerThread();i++) this->baseClass::operator()(i,t) = this->baseClass::operator()(i,t) * f;
    return *this;
  }

  //hexfloat option: For reproducibility testing, write the output in hexfloat format rather than truncating the precision
  void write(const std::string &filename, const bool hexfloat = false) const{
    const char* fmt = hexfloat ? "%a %a " : "%.16e %.16e ";

    FILE *p;
    if((p = Fopen(filename.c_str(),"w")) == NULL)
      ERR.FileA("KtoPiPiGparityResultsContainer","write",filename.c_str());
    for(int tk=0;tk<Lt;tk++){
      for(int tdis=0;tdis<Lt;tdis++){
	Fprintf(p,"%d %d ", tk, tdis);
	for(int fidx=0;fidx<2;fidx++){
	  std::complex<Float> dp = convertComplexD((*this)(tk,tdis,fidx));
	  Fprintf(p,fmt,std::real(dp),std::imag(dp));
	}
	Fprintf(p,"\n");
      }
    }	
    Fclose(p);
  }


};

//Daiqian places both type3 and mix3 as well as type4 and mix4 diagrams into combined files
//hexfloat option: For reproducibility testing, write the output in hexfloat format rather than truncating the precision
template<typename ComplexType, typename AllocPolicy>
inline static void write(const std::string &filename, const KtoPiPiGparityResultsContainer<ComplexType,AllocPolicy> &con, const KtoPiPiGparityMixDiagResultsContainer<ComplexType,AllocPolicy> &mix, const bool hexfloat = false){
  const char* fmt = hexfloat ? "%a %a " : "%.16e %.16e ";
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  int n_contract = con.getNcontract();
  FILE *p;
  if((p = Fopen(filename.c_str(),"w")) == NULL)
    ERR.FileA("KtoPiPiGparityResultsContainer","write",filename.c_str());
  for(int tk=0;tk<Lt;tk++){
    for(int tdis=0;tdis<Lt;tdis++){
      Fprintf(p,"%d %d ", tk, tdis);
      for(int cidx=0; cidx<n_contract; cidx++){
	for(int gcombidx=0;gcombidx<8;gcombidx++){
	  std::complex<Float> dp = convertComplexD(con(tk,tdis,cidx,gcombidx));
	  Fprintf(p,fmt,std::real(dp),std::imag(dp));
	}
      }
      for(int fidx=0;fidx<2;fidx++){
	std::complex<Float> dp = convertComplexD(mix(tk,tdis,fidx));
	Fprintf(p,fmt,std::real(dp),std::imag(dp));
      }
      Fprintf(p,"\n");
    }
  }	
  Fclose(p);
}




class ComputeKtoPiPiGparityBase{
public:
  //Compute i modulo Lt, where i can be negative
  inline static int modLt(int i, const int &Lt){
    while(i<0) i += Lt;
    return i % Lt;
  }

  //Get the matrix "\Gamma" which is one of the following:
  //Gidx are indices with the following mapping
  //0 M_{0,V} = F_0 \gamma_\mu
  //1 M_{0,A} = F_0 \gamma_\mu\gamma^5
  //2 M_{1,V} = -F_1 \gamma_\mu
  //3 M_{1,A} = -F_1 \gamma_\mu\gamma^5

  //First index is the Gidx, second is mu
  template<typename ComplexType>
  static const CPSspinColorFlavorMatrix<ComplexType> & Gamma(const int gidx, const int mu){
    static CPSspinColorFlavorMatrix<ComplexType> _Gamma[4][4];
    static bool setup = false;
    if(!setup){
      for(int mu=0;mu<4;mu++){
	for(int nu=0;nu<4;nu++) _Gamma[nu][mu].unit();
  	_Gamma[0][mu].unit().pr(F0).gr(mu);
  	_Gamma[1][mu].unit().pr(F0).gr(mu).gr(-5);
  	_Gamma[2][mu].unit().pr(F1).gr(mu).timesMinusOne();
  	_Gamma[3][mu].unit().pr(F1).gr(mu).gr(-5).timesMinusOne();
      }
      setup = true;
    }
    return _Gamma[gidx][mu];
  }


  //In practise we need only 8 combinations of gidx:
  // 0V,0A  -> 0,1
  // 0A,0V  -> 1,0
  // 0V,1A  -> 0,3
  // 0A,1V  -> 1,2
  // 1V,0A  -> 2,1
  // 1A,0V  -> 3,0
  // 1V,1A  -> 2,3
  // 1A,1V  -> 3,2

  //This method maps the index i \in {0..7} to the Gamma1 matrix
  template<typename ComplexType>
  inline static const CPSspinColorFlavorMatrix<ComplexType> & Gamma1(const int i, const int mu){
    static int g1[8] = {0,1,0,1,2,3,2,3};
    return Gamma<ComplexType>(g1[i],mu);
  }
  //Same for Gamma2
  template<typename ComplexType>
  inline static const CPSspinColorFlavorMatrix<ComplexType> & Gamma2(const int i, const int mu){
    static int g2[8] = {1,0,3,2,1,0,3,2};
    return Gamma<ComplexType>(g2[i],mu);
  }

  template<typename ComplexType>
  static void multGammaLeft(CPSspinColorFlavorMatrix<ComplexType> &M, const int whichGamma, const int i, const int mu){
    assert(whichGamma == 1 || whichGamma==2);
    static int g1[8] = {0,1,0,1,2,3,2,3};
    static int g2[8] = {1,0,3,2,1,0,3,2};

    int gg = whichGamma == 1 ? g1[i] : g2[i];
    switch(gg){
    case 0:
      M.pl(F0).gl(mu);
      break;
    case 1:
      M.pl(F0).glAx(mu);
      break;
    case 2:
      M.pl(F1).gl(mu).timesMinusOne();
      break;
    case 3:
      M.pl(F1).glAx(mu).timesMinusOne();
      break;
    default:
      ERR.General("ComputeKtoPiPiGparityBase","multGammaLeft","Invalid idx\n");
      break;
    }
  }

  template<typename ComplexType>
  static void multGammaRight(CPSspinColorFlavorMatrix<ComplexType> &M, const int whichGamma, const int i, const int mu){
    assert(whichGamma == 1 || whichGamma==2);
    static int g1[8] = {0,1,0,1,2,3,2,3};
    static int g2[8] = {1,0,3,2,1,0,3,2};

    int gg = whichGamma == 1 ? g1[i] : g2[i];
    switch(gg){
    case 0:
      M.pr(F0).gr(mu);
      break;
    case 1:
      M.pr(F0).grAx(mu);
      break;
    case 2:
      M.pr(F1).gr(mu).timesMinusOne();
      break;
    case 3:
      M.pr(F1).grAx(mu).timesMinusOne();
      break;
    default:
      ERR.General("ComputeKtoPiPiGparityBase","multGammaRight","Invalid idx\n");
      break;
    }
  }


  template<typename ComplexType>
  static CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > multGammaLeft(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &M, const int whichGamma, 
									      const int i, const int mu){
    assert(whichGamma == 1 || whichGamma==2);
    static int g1[8] = {0,1,0,1,2,3,2,3};
    static int g2[8] = {1,0,3,2,1,0,3,2};

    CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > tmp(M.getDimPolParams());
 
    int gg = whichGamma == 1 ? g1[i] : g2[i];
    switch(gg){
    case 0:
      tmp = gl_r(M,mu);
      pl( tmp, F0 );
      break;
    case 1:
      tmp = glAx_r(M, mu);
      pl(tmp , F0 );
      break;
    case 2:
      tmp = gl_r(M,mu);
      timesMinusOne( pl(tmp , F1 ) );      
      break;
    case 3:
      tmp = glAx_r(M,mu);
      timesMinusOne( pl( tmp, F1 ) );
      break;
    default:
      ERR.General("ComputeKtoPiPiGparityBase","multGammaLeft","Invalid idx\n");
      break;
    }
    return tmp;
  }

  template<typename ComplexType>
  static CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > multGammaRight(const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &M, const int whichGamma, 
									       const int i, const int mu){
    assert(whichGamma == 1 || whichGamma==2);
    static int g1[8] = {0,1,0,1,2,3,2,3};
    static int g2[8] = {1,0,3,2,1,0,3,2};

    CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > tmp(M.getDimPolParams());

    int gg = whichGamma == 1 ? g1[i] : g2[i];
    switch(gg){
    case 0:
      tmp = gr_r(M,mu);      
      pr( tmp, F0 );
      break;
    case 1:
      tmp = grAx_r(M,mu);
      pr( tmp, F0 );
      break;
    case 2:
      tmp = gr_r(M,mu);
      timesMinusOne( pr( tmp, F1 ) );
      break;
    case 3:
      tmp = grAx_r(M,mu);
      timesMinusOne( pr( tmp, F1) );
      break;
    default:
      ERR.General("ComputeKtoPiPiGparityBase","multGammaRight","Invalid idx\n");
      break;
    }
    return tmp;
  }


  template<typename ComplexType>
  static void multGammaRight(CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &out,
			     const CPSmatrixField<CPSspinColorFlavorMatrix<ComplexType> > &M, const int whichGamma, 
			     const int i, const int mu){
    assert(whichGamma == 1 || whichGamma==2);
    static int g1[8] = {0,1,0,1,2,3,2,3};
    static int g2[8] = {1,0,3,2,1,0,3,2};

    int gg = whichGamma == 1 ? g1[i] : g2[i];
    switch(gg){
    case 0:
      gr_r(out, M,mu);      
      pr( out, F0 );
      break;
    case 1:
      grAx_r(out, M,mu);
      pr( out, F0 );
      break;
    case 2:
      gr_r(out, M,mu);
      timesMinusOne( pr( out, F1 ) );
      break;
    case 3:
      grAx_r(out, M,mu);
      timesMinusOne( pr( out, F1) );
      break;
    default:
      ERR.General("ComputeKtoPiPiGparityBase","multGammaRight","Invalid idx\n");
      break;
    }
  }


  
  
  //Perform the spatial reduction and add the result into the output container
  template<typename ComplexType, typename AllocPolicy>
  static void add(const int Cidx, KtoPiPiGparityResultsContainer<ComplexType,AllocPolicy> &result, const int t_K, const int gcombidx,  const int Coff, const CPSmatrixField<ComplexType> &field){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    auto sum3d = localNodeSpatialSum(field);
    assert(sum3d.size() == GJP.TnodeSites());
    for(int t_loc=0;t_loc<GJP.TnodeSites();t_loc++){
      int t_glob = t_loc + GJP.TnodeSites()*GJP.TnodeCoor();
      int t_dis = modLt(t_glob - t_K, Lt);
      auto &C = result(t_K,t_dis,Cidx-Coff,gcombidx,0);
      C = C + sum3d[t_loc];
    }
  }

  template<typename ComplexType, typename AllocPolicy>
  static void add(KtoPiPiGparityMixDiagResultsContainer<ComplexType,AllocPolicy> &result, const int t_K, const int fidx,  const CPSmatrixField<ComplexType> &field){
    int Lt = GJP.Tnodes()*GJP.TnodeSites();
    auto sum3d = localNodeSpatialSum(field);
    assert(sum3d.size() == GJP.TnodeSites());
    for(int t_loc=0;t_loc<GJP.TnodeSites();t_loc++){
      int t_glob = t_loc + GJP.TnodeSites()*GJP.TnodeCoor();
      int t_dis = modLt(t_glob - t_K, Lt);
      auto &C = result(t_K,t_dis,fidx,0);
      C = C + sum3d[t_loc];
    }
  }

};

CPS_END_NAMESPACE

#endif

