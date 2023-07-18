#ifndef A2A_GRID_MADWF_H
#define A2A_GRID_MADWF_H

#include<config.h>

#if defined(USE_GRID) && defined(USE_GRID_A2A)
#include<util/lattice/fgrid.h>
#include <alg/ktopipi_jobparams.h>

CPS_START_NAMESPACE

struct ZmobParams{
  double b_plus_c_inner;
  int Ls_inner;
  double b_plus_c_outer;
  int Ls_outer;
  double lambda_max;
  bool complex_coeffs; //use Zmobius or regular Mobius
  
  ZmobParams(double b_plus_c_inner, int Ls_inner, double b_plus_c_outer, int Ls_outer, double lambda_max, bool complex_coeffs): b_plus_c_inner(b_plus_c_inner), Ls_inner(Ls_inner),
																b_plus_c_outer(b_plus_c_outer), Ls_outer(Ls_outer), 
																lambda_max(lambda_max), complex_coeffs(complex_coeffs){}
  
  bool operator<(const ZmobParams &r) const{
    if(b_plus_c_inner == r.b_plus_c_inner){
      if(Ls_inner == r.Ls_inner){
	if(b_plus_c_outer == r.b_plus_c_outer){
	  if(Ls_outer == r.Ls_outer){
	    if(lambda_max == r.lambda_max){
	      return complex_coeffs < r.complex_coeffs;
	    }else return lambda_max < r.lambda_max;
	  }else return Ls_outer < r.Ls_outer;
	}else return b_plus_c_outer < r.b_plus_c_outer;
      }else return Ls_inner<r.Ls_inner;
    }else return b_plus_c_inner < r.b_plus_c_inner;
  }
};

//Compute the (Z)Mobius gamma vector with caching. Here complex_coeffs == true implies ZMobius, false implies regular Mobius
inline std::vector<Grid::ComplexD> computeZmobiusGammaWithCache(double b_plus_c_inner, int Ls_inner, double b_plus_c_outer, int Ls_outer, double lambda_max, bool complex_coeffs){
  ZmobParams pstruct(b_plus_c_inner, Ls_inner, b_plus_c_outer, Ls_outer, lambda_max, complex_coeffs);
  static std::map<ZmobParams, std::vector<Grid::ComplexD> > cache;
  auto it = cache.find(pstruct);
  if(it == cache.end()){    
    std::vector<Grid::ComplexD> gamma_inner;
    
    LOGA2A << "MADWF Compute parameters with inner Ls = " << Ls_inner << std::endl;
    if(complex_coeffs){
      Grid::Approx::computeZmobiusGamma(gamma_inner, b_plus_c_inner, Ls_inner, b_plus_c_outer, Ls_outer, lambda_max);
    }else{
      Grid::Approx::zolotarev_data *zdata = Grid::Approx::higham(1.0,Ls_inner);
      gamma_inner.resize(Ls_inner);
      for(int s=0;s<Ls_inner;s++) gamma_inner[s] = zdata->gamma[s];
      Grid::Approx::zolotarev_free(zdata);
    }
    LOGA2A << "gamma:\n";
    for(int s=0;s<Ls_inner;s++) LOGA2A << s << " " << gamma_inner[s] << std::endl;
    
    cache[pstruct] = gamma_inner;
    return gamma_inner;
  }else{
    LOGA2A << "gamma (from cache):\n";
    for(int s=0;s<Ls_inner;s++) LOGA2A << s << " " << it->second[s] << std::endl;
    return it->second;
  }
}

//Get the (Z)Mobius parameters using the parameters in cg_controls, either through direct computation or from the struct directly
inline std::vector<Grid::ComplexD> getZMobiusGamma(const double b_plus_c_outer, const int Ls_outer,
					    const MADWFparams &madwf_p){
  const ZMobiusParams &zmp = madwf_p.ZMobius_params;

  std::vector<Grid::ComplexD> gamma_inner;

  //Get the parameters from the input struct
  if(zmp.gamma_src == A2A_ZMobiusGammaSourceInput){
    assert(zmp.gamma_real.gamma_real_len == madwf_p.Ls_inner);
    assert(zmp.gamma_imag.gamma_imag_len == madwf_p.Ls_inner);

    gamma_inner.resize(madwf_p.Ls_inner);
    for(int s=0;s<madwf_p.Ls_inner;s++)
      gamma_inner[s] = Grid::ComplexD( zmp.gamma_real.gamma_real_val[s], zmp.gamma_imag.gamma_imag_val[s] );
  }else{
    //Compute the parameters directly
    gamma_inner = computeZmobiusGammaWithCache(madwf_p.b_plus_c_inner, 
					       madwf_p.Ls_inner, 
					       b_plus_c_outer, Ls_outer,
					       zmp.compute_lambda_max, madwf_p.use_ZMobius);
  }
  return gamma_inner;
}

  



template<typename FermionFieldType>
struct CGincreaseTol : public Grid::MADWFinnerIterCallbackBase{
  Grid::ConjugateGradient<FermionFieldType> &cg_inner;  
  Grid::RealD outer_resid;

  CGincreaseTol(Grid::ConjugateGradient<FermionFieldType> &cg_inner,
	       Grid::RealD outer_resid): cg_inner(cg_inner), outer_resid(outer_resid){}
  
  void operator()(const Grid::RealD current_resid){
    LOGA2A << "CGincreaseTol with current residual " << current_resid << " changing inner tolerance " << cg_inner.Tolerance << " -> ";
    while(cg_inner.Tolerance < current_resid) cg_inner.Tolerance *= 2;    
    //cg_inner.Tolerance = outer_resid/current_resid;
    LOGA2ANT << cg_inner.Tolerance << std::endl;
  }
};

CPS_END_NAMESPACE

#endif

#endif
