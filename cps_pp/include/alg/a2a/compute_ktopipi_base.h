#ifndef _COMPUTE_KTOPIPI_BASE
#define _COMPUTE_KTOPIPI_BASE

#include<alg/a2a/fmatrix.h>

CPS_START_NAMESPACE

//Results containers and other helper classes for K->pipi

//Lt * Lt * 8 * ncontract tensor  (option for multiple independent threads)
//Here 8 is the number of combinations of spin-color-flavor matrix pairs (see below for indexing)
class KtoPiPiGparityResultsContainer: public basicComplexArray<Float>{
  int Lt;
  int ncontract;
  
  //gcombidx \in {0..7}, cf. below
  inline int map(const int &tk, const int &t_dis, const int &con_idx, const int &gcombidx, const int &thread) const{
    return con_idx + ncontract*( gcombidx + 8*( t_dis + Lt*( tk + Lt*thread) ) );
  }
public:
  void resize(const int &_ncontract, const int &_nthread){
    Lt = GJP.Tnodes()*GJP.TnodeSites();
    ncontract = _ncontract;

    int thread_size = ncontract * 8 * Lt * Lt;
    this->basicComplexArray<Float>::resize(thread_size, _nthread);
  }
  int getNcontract() const{ return ncontract; }

  KtoPiPiGparityResultsContainer(): basicComplexArray<Float>(){}
  KtoPiPiGparityResultsContainer(const int &_ncontract, const int &_nthread): basicComplexArray<Float>(){
    resize(_ncontract,_nthread);
  }

  inline std::complex<Float> & operator()(const int &tk, const int &t_dis, const int &con_idx, const int &gcombidx, const int &thread = 0){
    return con[map(tk,t_dis,con_idx,gcombidx,thread)];
  }
  inline const std::complex<Float> & operator()(const int &tk, const int &t_dis, const int &con_idx, const int &gcombidx, const int &thread = 0) const{
    return con[map(tk,t_dis,con_idx,gcombidx,thread)];
  }

  KtoPiPiGparityResultsContainer & operator*=(const Float f){
    for(int i=0;i<size;i++) con[i] *= f;
    return *this;
  }
	  // //Daiqian's loops are in the following order (outer->inner): tk, tdis, mu, con_idx, gcombidx
	  // int off = gcombidx + 8* cidx + (8*n_contract + dq_extrafig)*(tk * Lt + tdis); //Daiqian puts the sbar g5 d diagrams after the usual combination block for type3/type4. For these set dq_extrafig to 2

  void write(const std::string &filename) const{
    FILE *p;
    if((p = Fopen(filename.c_str(),"w")) == NULL)
      ERR.FileA("KtoPiPiGparityResultsContainer","write",filename.c_str());
    for(int tk=0;tk<Lt;tk++){
      for(int tdis=0;tdis<Lt;tdis++){
	Fprintf(p,"%d %d ", tk, tdis);
	for(int cidx=0; cidx<ncontract; cidx++){
	  for(int gcombidx=0;gcombidx<8;gcombidx++){
	    const std::complex<Float> &dp = (*this)(tk,tdis,cidx,gcombidx);
	    Fprintf(p,"%.16e %.16e ",std::real(dp),std::imag(dp));
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
class KtoPiPiGparityMixDiagResultsContainer: public basicComplexArray<Float>{
  int Lt;
  
  //fidx \in {0..1}, as above
  inline int map(const int &tk, const int &t_dis, const int &fidx, const int &thread) const{
    return fidx + 2*( t_dis + Lt*( tk + Lt*thread) );
  }
public:
  void resize(const int &_nthread){
    Lt = GJP.Tnodes()*GJP.TnodeSites();
    int thread_size = 2 * Lt * Lt;
    this->basicComplexArray<Float>::resize(thread_size, _nthread);
  }

  KtoPiPiGparityMixDiagResultsContainer(): basicComplexArray<Float>(){}
  KtoPiPiGparityMixDiagResultsContainer(const int &_nthread): basicComplexArray<Float>(){
    resize(_nthread);
  }

  inline std::complex<Float> & operator()(const int &tk, const int &t_dis, const int &fidx, const int &thread = 0){
    return con[map(tk,t_dis,fidx,thread)];
  }
  inline const std::complex<Float> & operator()(const int &tk, const int &t_dis, const int &fidx, const int &thread = 0) const{
    return con[map(tk,t_dis,fidx,thread)];
  }

  KtoPiPiGparityMixDiagResultsContainer & operator*=(const Float &f){
    for(int i=0;i<size;i++) con[i] *= f;
    return *this;
  }

  void write(const std::string &filename) const{
    FILE *p;
    if((p = Fopen(filename.c_str(),"w")) == NULL)
      ERR.FileA("KtoPiPiGparityResultsContainer","write",filename.c_str());
    for(int tk=0;tk<Lt;tk++){
      for(int tdis=0;tdis<Lt;tdis++){
	Fprintf(p,"%d %d ", tk, tdis);
	for(int fidx=0;fidx<2;fidx++){
	  const std::complex<Float> &dp = (*this)(tk,tdis,fidx);
	  Fprintf(p,"%.16e %.16e ",std::real(dp),std::imag(dp));
	}
	Fprintf(p,"\n");
      }
    }	
    Fclose(p);
  }


};

//Daiqian places both type3 and mix3 as well as type4 and mix4 diagrams into combined files
inline static void write(const std::string &filename, const KtoPiPiGparityResultsContainer &con, const KtoPiPiGparityMixDiagResultsContainer &mix){
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
	  const std::complex<Float> &dp = con(tk,tdis,cidx,gcombidx);
	  Fprintf(p,"%.16e %.16e ",std::real(dp),std::imag(dp));
	}
      }
      for(int fidx=0;fidx<2;fidx++){
	const std::complex<Float> &dp = mix(tk,tdis,fidx);
	Fprintf(p,"%.16e %.16e ",std::real(dp),std::imag(dp));
      }
      Fprintf(p,"\n");
    }
  }	
  Fclose(p);
}




class ComputeKtoPiPiGparityBase{
protected:

  //Gidx are indices with the following mapping
  //0 M_{0,V} = F_0 \gamma_\mu
  //1 M_{0,A} = F_0 \gamma_\mu\gamma^5
  //2 M_{1,V} = -F_1 \gamma_\mu
  //3 M_{1,A} = -F_1 \gamma_\mu\gamma^5

  //First index is the Gidx, second is mu  
  static const SpinColorFlavorMatrix & Gamma(const int &gidx, const int &mu){
    static SpinColorFlavorMatrix _Gamma[4][4];
    static bool setup = false;
    if(!setup){
      for(int mu=0;mu<4;mu++){
	_Gamma[0][mu] = _F0 * gmu[mu];
	_Gamma[1][mu] = _F0 * gmu[mu]*g5;
	_Gamma[2][mu] = _F1 * gmu[mu]* -1.0;
	_Gamma[3][mu] = _F1 * gmu[mu]*g5* -1.0;
      }
      setup = true;
    }
    return _Gamma[gidx][mu];
  }

  //In practise we need only 8 combinations of gidx that are needed:
  // 0V,0A  -> 0,1
  // 0A,0V  -> 1,0
  // 0V,1A  -> 0,3
  // 0A,1V  -> 1,2
  // 1V,0A  -> 2,1
  // 1A,0V  -> 3,0
  // 1V,1A  -> 2,3
  // 1A,1V  -> 3,2

  //This method maps the index i \in {0..7} to the Gamma1 matrix
  static const SpinColorFlavorMatrix & Gamma1(const int &i, const int &mu){
    static int g1[8] = {0,1,0,1,2,3,2,3};
    return Gamma(g1[i],mu);
  }
  //Same for Gamma2
  static const SpinColorFlavorMatrix & Gamma2(const int &i, const int &mu){
    static int g2[8] = {1,0,3,2,1,0,3,2};
    return Gamma(g2[i],mu);
  }

  static void multGammaLeft(SpinColorFlavorMatrix &M, const int whichGamma, const int i, const int mu){
    assert(whichGamma == 1 || whichGamma==2);
    static int g1[8] = {0,1,0,1,2,3,2,3};
    static int g2[8] = {1,0,3,2,1,0,3,2};

    int gg = whichGamma == 1 ? g1[i] : g2[i];
    switch(gg){
    case 0:
      M.pl(F0).gl(mu);
      break;
    case 1:
      //M.pl(F0).gl(-5).gl(mu);
      M.pl(F0).glAx(mu);
      break;
    case 2:
      M.pl(F1).gl(mu); M *= -1.0;
      break;
    case 3:
      //M.pl(F1).gl(-5).gl(mu); M *= -1.0;
      M.pl(F1).glAx(mu); M *= -1.0;
      break;
    default:
      ERR.General("ComputeKtoPiPiGparityBase","multGamma1Left","Invalid idx\n");
      break;
    }
  }



  static SpinColorFlavorMatrix _F0;
  static SpinColorFlavorMatrix _F1;
  static SpinColorFlavorMatrix g5;
  static SpinColorFlavorMatrix S2;
  static SpinColorFlavorMatrix unit;
  static SpinColorFlavorMatrix gmu[4];
};

CPS_END_NAMESPACE

#endif

