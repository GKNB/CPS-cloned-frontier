#ifndef _COMPUTE_KTOSIGMA_BASE_TIANLE_H
#define _COMPUTE_KTOSIGMA_BASE_TIANLE_H

#include<alg/a2a/spin_color_matrices.h>
#include<alg/a2a/fmatrix.h>

CPS_START_NAMESPACE


//g2_idx is defined as follows:
//0 -F1, 1-g5
//1 -F1, 1+g5
//2 +F0, 1-g5
//3 +F0, 1+g5

template<typename ComplexType, typename AllocPolicy>
class KtoSigmaGparityResultsContainerTianle: public basicComplexArray<ComplexType,AllocPolicy>
{
  int Lt;
  const static int ndiag = 23;

  //notice here diag_idx is the idx for D, so offset = 1
  inline int map(const int tk, const int tdis, const int diag_idx, const int g12_idx) const
  {
    return (diag_idx - 1) + ndiag * (g12_idx + 8 * (tdis + Lt * tk) );
  }

public:
  void resize()
  {
    Lt = GJP.Tnodes()*GJP.TnodeSites();
    int thread_size = ndiag * 8 * Lt * Lt;
    this->basicComplexArray<ComplexType,AllocPolicy>::resize(thread_size, 1);
  }

  int getNdiag() const {  return ndiag; }

  KtoSigmaGparityResultsContainerTianle(): basicComplexArray<ComplexType,AllocPolicy>(){  resize(); }
  inline ComplexType &operator()(const int tk, const int tdis, const int diag_idx, const int g2_idx) {  return this->con[map(tk,tdis,diag_idx,g2_idx)]; }
  inline ComplexType &operator()(const int tk, const int tdis, const int diag_idx, const int g2_idx) const {  return this->con[map(tk,tdis,diag_idx,g2_idx)]; }
  KtoSigmaGparityResultsContainerTianle &operator*=(const Float f)
  {
    for(int i=0; i<this->size(); i++)
      this->con[i] = this->con[i] * f;
    return *this;
  }
};

template<typename ComplexType, typename AllocPolicy>
class KtoSigmaGparityMixResultsContainerTianle: public basicComplexArray<ComplexType,AllocPolicy>
{
  int Lt;
  const static int ntype = 2;

  //we define them as type12, type3 and type4
  inline int map(const int tk, const int tdis, const int type_idx, const int gidx) const
  {
    return (type_idx - 3) + ntype * (gidx + 2 * (tdis + Lt * tk) );
  }

public:
  void resize()
  {
    Lt = GJP.Tnodes()*GJP.TnodeSites();
    int thread_size = ntype * 2 * Lt * Lt;
    this->basicComplexArray<ComplexType,AllocPolicy>::resize(thread_size, 1);
  }

  int getNtype() const {  return ntype; }
  KtoSigmaGparityMixResultsContainerTianle(): basicComplexArray<ComplexType,AllocPolicy>(){  resize(); }
  inline ComplexType&operator()(const int tk, const int tdis, const int type_idx, const int gidx) {  return this->con[map(tk,tdis,type_idx,gidx)]; }
  inline ComplexType&operator()(const int tk, const int tdis, const int type_idx, const int gidx) const {  return this->con[map(tk,tdis,type_idx,gidx)]; }
  KtoSigmaGparityMixResultsContainerTianle &operator*=(const Float f)
  {
    for(int i=0; i<this->size(); i++)
      this->con[i] = this->con[i] * f;
    return *this;
  }
};

template<typename ComplexType, typename AllocPolicy>
inline static void write(const std::string &filename_pre, const std::string &filename_end, const KtoSigmaGparityResultsContainerTianle<ComplexType, AllocPolicy> &result, const KtoSigmaGparityMixResultsContainerTianle<ComplexType, AllocPolicy> &mix)
{
  const char* fmt = "%.16e %.16e ";
  int Lt = GJP.Tnodes()*GJP.TnodeSites();

  const int ntype12 = 5;
  int type12[ntype12] = {1, 6, 8, 11, 19};
  std::string filename_12 = filename_pre + "12" + filename_end;
  FILE *p;
  if((p = Fopen(filename_12.c_str(),"w")) == NULL)
    ERR.FileA("KtoSigmaGparityResultsContainerTianle","write_type12",filename_12.c_str());
  for(int tk=0; tk<Lt; tk++)
  {
    for(int tdis=0; tdis<Lt; tdis++)
    {
      Fprintf(p,"%d %d ", tk, tdis);
      for(int diag_idx=0; diag_idx<ntype12; diag_idx++)
      {
        for(int g12_idx=0;g12_idx<8;g12_idx++)
        {
          std::complex<Float> dp = convertComplexD(result(tk,tdis,type12[diag_idx],g12_idx));
          Fprintf(p,fmt,std::real(dp),std::imag(dp));
        }
      }
      Fprintf(p,"\n");
    }
  }
  Fclose(p);


  const int ntype3 = 9;
  int type3[ntype3] = {2, 3, 7, 10, 14, 16, 18, 21, 23};
  std::string filename_3 = filename_pre + "3" + filename_end;
  if((p = Fopen(filename_3.c_str(),"w")) == NULL)
    ERR.FileA("KtoSigmaGparityResultsContainerTianle","write_type3",filename_3.c_str());
  for(int tk=0; tk<Lt; tk++)
  {
    for(int tdis=0; tdis<Lt; tdis++)
    {
      Fprintf(p,"%d %d ", tk, tdis);
      for(int diag_idx=0; diag_idx<ntype3; diag_idx++)
      {
        for(int g12_idx=0;g12_idx<8;g12_idx++)
        {
          std::complex<Float> dp = convertComplexD(result(tk,tdis,type3[diag_idx],g12_idx));
          Fprintf(p,fmt,std::real(dp),std::imag(dp));
        }
      }
      for(int gidx=0;gidx<2;gidx++)
      {
        std::complex<Float> dp = convertComplexD(mix(tk,tdis,3,gidx));
        Fprintf(p,fmt,std::real(dp),std::imag(dp));
      }
      Fprintf(p,"\n");
    }
  }
  Fclose(p);

  const int ntype4 = 9;
  int type4[ntype4] = {4, 5, 9, 12, 13, 15, 17, 20, 22};
  std::string filename_4 = filename_pre + "4" + filename_end;
  if((p = Fopen(filename_4.c_str(),"w")) == NULL)
    ERR.FileA("KtoSigmaGparityResultsContainerTianle","write_type4",filename_4.c_str());
  for(int tk=0; tk<Lt; tk++)
  {
    for(int tdis=0; tdis<Lt; tdis++)
    {
      Fprintf(p,"%d %d ", tk, tdis);
      for(int diag_idx=0; diag_idx<ntype4; diag_idx++)
      {
        for(int g12_idx=0;g12_idx<8;g12_idx++)
        {
          std::complex<Float> dp = convertComplexD(result(tk,tdis,type4[diag_idx],g12_idx));
          Fprintf(p,fmt,std::real(dp),std::imag(dp));
        }
      }
      for(int gidx=0;gidx<2;gidx++)
      {
        std::complex<Float> dp = convertComplexD(mix(tk,tdis,4,gidx));
        Fprintf(p,fmt,std::real(dp),std::imag(dp));
      }
      Fprintf(p,"\n");
    }
  }
  Fclose(p);
}

CPS_END_NAMESPACE

#endif
