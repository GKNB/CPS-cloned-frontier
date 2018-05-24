#ifndef _COMPUTE_KTOSIGMA_TIANLE_H
#define _COMPUTE_KTOSIGMA_TIANLE_H

#include<algorithm>
#include<map>
#include<utility>

#include "compute_ktosigma_base_tw.h"

#include<alg/a2a/mf_momcontainer.h>
#include<alg/a2a/mesonfield_mult_vMv_split.h>
#include<alg/a2a/mesonfield_mult_vMv_split_grid.h>
#include<alg/a2a/required_momenta.h>
#include<alg/a2a/inner_product.h>
#include<alg/a2a/mf_productstore.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
class ComputeKtoSigmaGparityTianle
{
  inline static int modLt(int i, const int &Lt)
  {
    while(i<0) i += Lt;
    return i % Lt;
  }
  static bool topNeedMultType1(const int &top_glb, const std::vector<int> &tsep, const int &tsig, const int &Lt)
  {
    int tsep_max = *(std::max_element(tsep.cbegin(), tsep.cend()));
    int tk = modLt(tsig - tsep_max, Lt);
    int tdis = modLt(top_glb - tk, Lt);
    return (tdis !=0 && tdis < tsep_max);
  }

public:
  typedef typename mf_Policies::ComplexType ComplexType;
  typedef CPSspinColorFlavorMatrix<ComplexType> SCFmat;
  typedef KtoSigmaGparityResultsContainerTianle<typename mf_Policies::ComplexType, typename mf_Policies::AllocPolicy> ResultsContainerType;
  typedef KtoSigmaGparityMixResultsContainerTianle<typename mf_Policies::ComplexType, typename mf_Policies::AllocPolicy> MixResultsContainerType;

  //g2_idx is defined as follows:
  //0 +F0, 1    0V
  //1 +F0, g5   0A
  //2 -F1, 1    1V
  //3 -F1, g5   1A

  static void multGammaLeft(SCFmat &M, const int idx, const int mu)
  {
    switch(idx)
    {
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
        assert(0 && "Error:multGammaLeft failed: idx not in range\n");
    }
  }

  //idx is defined as follows
  // 0V,0A  -> 0,1
  // 0A,0V  -> 1,0
  // 0V,1A  -> 0,3
  // 0A,1V  -> 1,2
  // 1V,0A  -> 2,1
  // 1A,0V  -> 3,0
  // 1V,1A  -> 2,3
  // 1A,1V  -> 3,2

  static void multGammaLeft(SCFmat &M, const int g_idx, const int idx, const int mu)
  {
    assert(idx >= 0 && idx <= 7);
    static int g1[8] = {0,1,0,1,2,3,2,3};
    static int g2[8] = {1,0,3,2,1,0,3,2};
    if(g_idx == 1)
    {
      multGammaLeft(M,g1[idx],mu);
    }
    else if(g_idx == 2)
    {
      multGammaLeft(M,g2[idx],mu);
    }
    else
      assert(0 && "Error: g_idx can only be 1 or 2\n");
  }

  static void compute(std::vector<ResultsContainerType> &result, std::vector<MixResultsContainerType> &mix, 
                      const std::vector<int> &tsep, const int &tstep,
                      const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon,
                      const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma,
                      const A2AvectorV<mf_Policies> &vL, const A2AvectorV<mf_Policies> &vH, 
                      const A2AvectorW<mf_Policies> &wL, const A2AvectorW<mf_Policies> &wH);

  static void mf_product(std::map<std::pair<int,int>, A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> > &mf_prod_con, //key = <tsig,tsep>
                         const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon,
                         const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma,
                         const std::vector<int> &tsep, const int tstep, const int Lt);



};

template<typename mf_Policies>
void ComputeKtoSigmaGparityTianle<mf_Policies>::mf_product(std::map<std::pair<int,int>, A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> > &mf_prod_con, //key = <tsig,tsep>
                                                           const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon,
                                                           const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma,
                                                           const std::vector<int> &tsep, const int tstep, const int Lt)
{
  for(int _tsig = 1; _tsig <= Lt; _tsig += tstep)
  {
    int tsig = modLt(_tsig, Lt);
    for(int i_sep = 0; i_sep < tsep.size(); i_sep++)
    {
      int t_ks = tsep[i_sep];
      int tk = modLt(tsig - t_ks, Lt);
      std::pair<int,int> key_prod(tsig,t_ks);
      A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> *tmp = &mf_prod_con[key_prod];
      mult(*tmp, mf_sigma[tsig], mf_kaon[tk]);
    }
  }
}

template<typename mf_Policies>
void ComputeKtoSigmaGparityTianle<mf_Policies>::compute(std::vector<ResultsContainerType> &result, std::vector<MixResultsContainerType> &mix, 
                                                        const std::vector<int> &tsep, const int &tstep,
                                                        const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> > &mf_kaon,
                                                        const std::vector<A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorVfftw> > &mf_sigma,
                                                        const A2AvectorV<mf_Policies> &vL, const A2AvectorV<mf_Policies> &vH, 
                                                        const A2AvectorW<mf_Policies> &wL, const A2AvectorW<mf_Policies> &wH)
{
  //we can add timer here
  result.resize(tsep.size());
  mix.resize(tsep.size());
  SCFmat mix3_Gamma[2];
  mix3_Gamma[0].unit().pr(F0).gr(-5);
  mix3_Gamma[1].unit().pr(F1).gr(-5).timesMinusOne();

  const int Lt = GJP.Tnodes()*GJP.TnodeSites();
  assert(Lt % tstep == 0);
  const int size_3d = vL.getMode(0).nodeSites(0) * vL.getMode(0).nodeSites(1) * vL.getMode(0).nodeSites(2);
  std::map<std::pair<int,int>,int> t_K_map; //key = <tsig,tsepidx>, value = tk
  for(int _tsig = 1; _tsig <= Lt; _tsig += tstep)
  {
    int tsig = modLt(_tsig,Lt);
    for(int tsepidx = 0; tsepidx < tsep.size(); tsepidx++)
      t_K_map[std::pair<int,int>(tsig,tsepidx)] = modLt(tsig - tsep[tsepidx],Lt);
  }
  std::map<std::pair<int,int>, A2AmesonField<mf_Policies,A2AvectorWfftw,A2AvectorWfftw> > mf_prod_con;
  mf_product(mf_prod_con, mf_kaon, mf_sigma, tsep, tstep, Lt);

  for(int top_loc = 0; top_loc < GJP.TnodeSites(); top_loc++)
  {
    const int top_glb = top_loc + GJP.TnodeCoor()*GJP.TnodeSites();
    for(int xop3d_loc = 0; xop3d_loc < size_3d; xop3d_loc++)
    {
      std::map<int,SCFmat> part1_storage_vMv; //key=tk,value=part1 which is generated by vMv, can be used in type1,3
      std::map<std::pair<int,int>,SCFmat> part1_storage_vMMv; //key=<tsig,tk>,value=part1 which is generated by vM1M2v, can be used in type2
      std::map<int,SCFmat> part2_storage_vMv; //key=tsig,value=part2 which is generated by vMv, can be used in type1
      SCFmat part2_L; //generated by vv_Low, can be used in type2,3
      SCFmat part2_H; //generated by vv_High, can be used in type2,3
      for(int _tsig = 1; _tsig <= Lt; _tsig += tstep)
      {
        int tsig = modLt(_tsig,Lt);
        for(int tsep_idx = 0; tsep_idx < tsep.size(); tsep_idx++)
        {
          int tk = modLt(tsig - tsep[tsep_idx],Lt);
          int tdis = modLt(top_glb - tk, Lt);
          if(tdis !=0 && tdis < tsep[tsep_idx] && part1_storage_vMv.find(tk) == part1_storage_vMv.end())
          {
            SCFmat *part1 = &part1_storage_vMv[tk];
            mult(*part1, vL, mf_kaon[tk], vH, xop3d_loc, top_loc, false, true);
            part1->gr(-5);
          }
          std::pair<int,int> p2_key(tsig,tk);
          if(tdis !=0 && tdis < tsep[tsep_idx] && part1_storage_vMMv.find(p2_key) == part1_storage_vMMv.end())
          {
            SCFmat *part1 = &part1_storage_vMMv[p2_key];
            mult(*part1, vL, mf_prod_con.at(std::pair<int,int>(tsig,tsep[tsep_idx])), vH, xop3d_loc, top_loc, false, true);
            part1->gr(-5);
          }
        }
        if(/*topNeedMultType1(top_glb,tsep,tsig,Lt) && */part2_storage_vMv.find(tsig) == part2_storage_vMv.end())
        {
          SCFmat *part2 = &part2_storage_vMv[tsig];
          mult(*part2, vL, mf_sigma[tsig], wL, xop3d_loc, top_loc, false, true);
        }
      }
      mult(part2_L, vL, wL, xop3d_loc, top_loc, false, true);
      mult(part2_H, vH, wH, xop3d_loc, top_loc, false, true);

      for(int _tsig = 1; _tsig <= Lt; _tsig += tstep)
      {
        int tsig = modLt(_tsig,Lt);
        for(int tsep_idx = 0; tsep_idx < tsep.size(); tsep_idx++)
        {
          int tk = modLt(tsig - tsep[tsep_idx],Lt);
          int tdis = modLt(top_glb - tk, Lt);
          if(tdis >= tsep[tsep_idx] || tdis == 0) continue;
          for(int g12_idx=0; g12_idx<8; g12_idx++)
          {
            for(int mu=0; mu<4; mu++)
            {
              //initialize
              SCFmat G1_pt1_vMv = part1_storage_vMv.at(tk);
              SCFmat G1_pt1_vMMv = part1_storage_vMMv.at(std::pair<int,int>(tsig,tk));
              SCFmat G1_pt2_L = part2_L;
              SCFmat G1_pt2_H = part2_H;
              SCFmat G1_pt2_vMv = part2_storage_vMv.at(tsig);

              SCFmat G2_pt1_vMv = part1_storage_vMv.at(tk);
              SCFmat G2_pt1_vMMv = part1_storage_vMMv.at(std::pair<int,int>(tsig,tk));
              SCFmat G2_pt2_L = part2_L;
              SCFmat G2_pt2_H = part2_H;
              SCFmat G2_pt2_vMv = part2_storage_vMv.at(tsig);

              //make the name consistent with content
              multGammaLeft(G1_pt1_vMv,1,g12_idx,mu);
              multGammaLeft(G1_pt1_vMMv,1,g12_idx,mu);
              multGammaLeft(G1_pt2_L,1,g12_idx,mu);
              multGammaLeft(G1_pt2_H,1,g12_idx,mu);
              multGammaLeft(G1_pt2_vMv,1,g12_idx,mu);

              multGammaLeft(G2_pt1_vMv,2,g12_idx,mu);
              multGammaLeft(G2_pt1_vMMv,2,g12_idx,mu);
              multGammaLeft(G2_pt2_L,2,g12_idx,mu);
              multGammaLeft(G2_pt2_H,2,g12_idx,mu);
              multGammaLeft(G2_pt2_vMv,2,g12_idx,mu);

              auto G1_pt2_vMv_Tc = G1_pt2_vMv;  G1_pt2_vMv_Tc.TransposeColor();
              auto G2_pt2_vMv_Tc = G2_pt2_vMv;  G2_pt2_vMv_Tc.TransposeColor();
              auto G1_pt2_L_Tc = G1_pt2_L;      G1_pt2_L_Tc.TransposeColor();
              auto G2_pt2_L_Tc = G2_pt2_L;      G2_pt2_L_Tc.TransposeColor();

              result[tsep_idx](tk,tdis,1,g12_idx) +=  Trace(G2_pt1_vMv, G1_pt2_vMv);
              result[tsep_idx](tk,tdis,6,g12_idx) +=  G1_pt1_vMv.Trace() * G2_pt2_vMv.Trace();
              result[tsep_idx](tk,tdis,8,g12_idx) +=  Trace(G2_pt1_vMv, G1_pt2_vMv_Tc);
              result[tsep_idx](tk,tdis,11,g12_idx) += Trace(G1_pt1_vMv.SpinFlavorTrace(), G2_pt2_vMv_Tc.SpinFlavorTrace());
              result[tsep_idx](tk,tdis,19,g12_idx) += Trace(G2_pt1_vMv.ColorTrace(), G1_pt2_vMv.ColorTrace());

              result[tsep_idx](tk,tdis,2,g12_idx) +=  G1_pt1_vMMv.Trace() * G2_pt2_L.Trace();
              result[tsep_idx](tk,tdis,3,g12_idx) +=  Trace(G2_pt1_vMMv, G1_pt2_L);
              result[tsep_idx](tk,tdis,7,g12_idx) +=  Trace(G2_pt1_vMMv, G1_pt2_L_Tc);
              result[tsep_idx](tk,tdis,10,g12_idx) += Trace(G1_pt1_vMMv.SpinFlavorTrace(), G2_pt2_L_Tc.SpinFlavorTrace());
              result[tsep_idx](tk,tdis,14,g12_idx) += G1_pt1_vMMv.Trace() * G2_pt2_H.Trace();
              result[tsep_idx](tk,tdis,16,g12_idx) += Trace(G1_pt1_vMMv, G2_pt2_H);
              result[tsep_idx](tk,tdis,18,g12_idx) += Trace(G2_pt1_vMMv.ColorTrace(), G1_pt2_L.ColorTrace());
              result[tsep_idx](tk,tdis,21,g12_idx) += Trace(G1_pt1_vMMv.SpinFlavorTrace(), G2_pt2_H.SpinFlavorTrace());
              result[tsep_idx](tk,tdis,23,g12_idx) += Trace(G1_pt1_vMMv.ColorTrace(), G2_pt2_H.ColorTrace());


              result[tsep_idx](tk,tdis,4,g12_idx) +=  Trace(G2_pt1_vMv, G1_pt2_L);
              result[tsep_idx](tk,tdis,5,g12_idx) +=  G1_pt1_vMv.Trace() * G2_pt2_L.Trace();
              result[tsep_idx](tk,tdis,9,g12_idx) +=  Trace(G2_pt1_vMv, G1_pt2_L_Tc);
              result[tsep_idx](tk,tdis,12,g12_idx) += Trace(G1_pt1_vMv.SpinFlavorTrace(), G2_pt2_L_Tc.SpinFlavorTrace());
              result[tsep_idx](tk,tdis,13,g12_idx) += G1_pt1_vMv.Trace() * G2_pt2_H.Trace();
              result[tsep_idx](tk,tdis,15,g12_idx) += Trace(G1_pt1_vMv, G2_pt2_H);
              result[tsep_idx](tk,tdis,17,g12_idx) += Trace(G2_pt1_vMv.ColorTrace(), G1_pt2_L.ColorTrace());
              result[tsep_idx](tk,tdis,20,g12_idx) += Trace(G1_pt1_vMv.SpinFlavorTrace(), G2_pt2_H.SpinFlavorTrace());
              result[tsep_idx](tk,tdis,22,g12_idx) += Trace(G1_pt1_vMv.ColorTrace(), G2_pt2_H.ColorTrace());

            }
          }
          for(int mix_gidx=0; mix_gidx<2; mix_gidx++)
          {
            mix[tsep_idx](tk,tdis,3,mix_gidx) += Trace(part1_storage_vMMv.at(std::pair<int,int>(tsig,tk)), mix3_Gamma[mix_gidx]);
            mix[tsep_idx](tk,tdis,4,mix_gidx) += Trace(part1_storage_vMv.at(tk), mix3_Gamma[mix_gidx]);
          }
        }
      }
    }  
  }
  for(int i=0; i<tsep.size(); i++)
  {
    result[i].threadSum();
    result[i].nodeSum();
    mix[i].threadSum();
    mix[i].nodeSum();
  }
}

CPS_END_NAMESPACE
#endif
