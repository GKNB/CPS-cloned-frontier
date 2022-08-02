#ifndef _A2A_PARAMS_H
#define _A2A_PARAMS_H

#include<alg/a2a_arg.h>
#include<string>
#include<sstream>

CPS_START_NAMESPACE

class A2Aparams{
protected:
  A2AArg args;
  int nl; //number of low modes
  int nhits; //number of stochastic hits per site for high modes
  int nflavors; //number of flavors
  int nspincolor; //number of spin*color
  int ntblocks; //number of blocked timeslices
  int ndilute; //number of high mode stochastic dilutions (e.g. if spin/color/timeslice diluting then  ndilute = 12*Lt 
  int nh; //total number of high mode fields = ndilute * nhits
  int nv; //total number of v fields =  nl + ndilute * nhits
  int Lt; //global lattice time extent

public:
  inline int getNl() const{ return nl; }
  inline int getNhits() const{ return nhits; }
  inline int getNflavors() const{ return nflavors; }
  inline int getNspinColor() const{ return nspincolor; }
  inline int getNtBlocks() const{ return ntblocks; }
  inline int getNdilute() const{ return ndilute; }
  inline int getNh() const{ return nh; }
  inline int getNv() const{ return nv; }
  inline int getLt() const{ return Lt; }

  A2Aparams(): nl(0),nhits(0),nflavors(0),nspincolor(0),ntblocks(0),ndilute(0),nh(0),nv(0),Lt(0){}
  A2Aparams(const A2AArg &_args);

  //FIXME: I change it a little bit to debug
  inline bool paramsEqual(const A2Aparams &r) const{
	  bool res = (nl == r.nl  &&  nhits == r.nhits  &&  nflavors == r.nflavors  &&  ntblocks == r.ntblocks  &&  
	    ndilute == r.ndilute  &&  nh == r.nh  &&  nv == r.nv && Lt == r.Lt);
	  if(!res)
	  {
		  std::cout << "nl = " << nl << "\t r.nl = " << r.nl
                      << "\nnhits = " << nhits << "\t r.nhits = " <<  r.nhits
                      << "\nnflavors = " <<  nflavors << "\t r.nflavors = " <<  r.nflavors
                      << "\nntblocks = " <<  ntblocks << "\t r.ntblocks = " <<  r.ntblocks
                      << "\nndilute = " <<  ndilute << "\t r.ndilute = " << r.ndilute
                      << "\nnh = " <<  nh << "\t r.nh = " << r.nh
                      << "\nnv = " <<  nv << "\t r.nv = " << r.nv
                      << "\nLt = " << Lt << "\t r.Lv = " << r.Lt;
	  }
	  return res;
  }

  inline const A2AArg &getArgs() const{ return args; }

  //Get the time block for a given lattice time t
  inline int tblock(const int t) const{ return t / args.src_width; }

  inline std::string print() const{
    std::ostringstream os; os << "nl=" << nl << " nhits=" << nhits << " nflavors=" << nflavors 
			      << " nspincolor=" << nspincolor << " ntblocks=" << ntblocks << " ndilute=" << ndilute << " nh=" << nh << " nv=" << nv << " Lt=" << Lt;
    return os.str();
  }
};

CPS_END_NAMESPACE

#endif
