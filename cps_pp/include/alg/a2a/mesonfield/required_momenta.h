#ifndef _REQUIRED_MOMENTA_H
#define _REQUIRED_MOMENTA_H

#include<iostream>
#include<vector>
#include<alg/a2a/utils/threemomentum.h>
#include<algorithm>

CPS_START_NAMESPACE

//For generic meson fields of form  A . B  this class contains the momenta assigned to A and B
class MomentumAssignments{
  //Outer index is total momentum, inner index are different assignments of quark momenta that have the same total momentum (optional)
  std::vector<std::vector<ThreeMomentum> >  momA;
  std::vector<std::vector<ThreeMomentum> >  momB;
  std::vector<ThreeMomentum> mom_total;
  int ngp;
  bool combine_same_total_mom; //if true, when momentum-pairs are added that have the same total momentum as an existing entry, the new pair will be added to the existing entry and not a new entry
public:
  
  MomentumAssignments(const int mp=0): ngp(0), combine_same_total_mom(false){
    //We require meson fields with both p and -p for each non-identical momentum direction.     
    for(int i=0;i<3;i++){
      if(GJP.Bc(i) == BND_CND_GPARITY) ngp++;
      if(i>0 && GJP.Bc(i) == BND_CND_GPARITY && GJP.Bc(i-1) != BND_CND_GPARITY){ ERR.General("MomentumAssignments","MomentumAssignments","Expect G-parity directions to be consecutive\n"); }     //(as it is setup here we expect the G-parity directions to be consecutive, i.e.  x, or x and y, or x and y and z)
    }
  }
  inline void combineSameTotalMomentum(const bool val = true){ combine_same_total_mom = val; }
  inline int nGparityDirs() const{ return ngp; }
  inline int nMom() const{ return momA.size(); } //number of total pion momenta
  inline int nAltMom(const int mom_idx) const{ return momA[mom_idx].size(); } //number of alternative momentum assignments that have the same total momentum

  inline ThreeMomentum getTotalMomentum(const int i) const{ return mom_total[i]; }

  inline ThreeMomentum getMomA(const int i, const int alt_idx = 0) const{ return momA[i][alt_idx]; }
  inline ThreeMomentum getMomB(const int i, const int alt_idx = 0) const{ return momB[i][alt_idx]; }

  inline std::pair<ThreeMomentum,ThreeMomentum> getMomAB(const int i, const int alt_idx = 0) const{ return std::pair<ThreeMomentum,ThreeMomentum>( momA[i][alt_idx], momB[i][alt_idx] ); }

  //Add a pair of quark momenta {pA,pB} for a meson of momentum ptot = pA + pB
  void addP(const std::pair<ThreeMomentum,ThreeMomentum> &pAB){
    ThreeMomentum ptot = pAB.first + pAB.second;
    if(combine_same_total_mom)
      for(int i=0;i<mom_total.size();i++){
	if(mom_total[i] == ptot){
	  momA[i].push_back(pAB.first);
	  momB[i].push_back(pAB.second);
	  return;
	}
      }
    mom_total.push_back(ptot);
    momA.push_back(std::vector<ThreeMomentum>(1, pAB.first));
    momB.push_back(std::vector<ThreeMomentum>(1, pAB.second));
  }
  //Add a pair of quark momenta {pA,pB} for a meson of momentum ptot = pA + pB
  //and a pair of quark momenta {-pA,-pB} for a meson of momentum ptot = -(pA + pB)
  //Typically meson fields of equal and opposite total momentum are both required to construct a two-point function hence this convenience function
  inline void addPandMinusP(const std::pair<ThreeMomentum,ThreeMomentum> &pAB){
    addP(pAB);
    addP(std::pair<ThreeMomentum,ThreeMomentum>(-pAB.first,-pAB.second));
  }
  
  //Add a A and B momentum (respectively) from a string in the form "(%d,%d,%d) + (%d%d,%d)"
  //Momenta with the same total momentum will be added to the same mom idx
  inline void addP(const std::string &p){
    std::pair<ThreeMomentum,ThreeMomentum> p2 = ThreeMomentum::parse_str_two_mom(p);
    addP(p2);
  }    
  //Add a A and B momentum (respectively) from a string in the form "(%d,%d,%d) + (%d%d,%d)", plus the negatives of these  
  inline void addPandMinusP(const std::string &p){
    std::pair<ThreeMomentum,ThreeMomentum> p2 = ThreeMomentum::parse_str_two_mom(p);
    addPandMinusP(p2);
  }


  //Add a A and B momentum (respectively) from components
  //Momenta with the same total momentum will be added to the same mom idx
  inline void addP(const int p1x, const int p1y, const int p1z,
		   const int p2x, const int p2y, const int p2z){
    std::pair<ThreeMomentum,ThreeMomentum> p2( ThreeMomentum(p1x,p1y,p1z), ThreeMomentum(p2x,p2y,p2z) );
    addP(p2);
  }    
  //Add a A and B momentum (respectively) from components, plus the negatives of these  
  inline void addPandMinusP(const int p1x, const int p1y, const int p1z,
			    const int p2x, const int p2y, const int p2z){
    std::pair<ThreeMomentum,ThreeMomentum> p2( ThreeMomentum(p1x,p1y,p1z), ThreeMomentum(p2x,p2y,p2z) );
    addPandMinusP(p2);
  }


  //Functions to symmetrize the momentum assignment of the A and B vectors
  inline void reverseABmomentumAssignments(){
    std::swap(momA,momB);
  }
  inline void symmetrizeABmomentumAssignments(){
    for(int p=0;p<nMom();p++){
      int nalt = nAltMom(p);
      for(int a=0;a<nalt;a++){
	momA[p].push_back(momB[p][a]);
	momB[p].push_back(momA[p][a]);
      }
    }
  };

  //Add all momentum assignments from the input
  //Make sure to enable combineSameTotalMomentum if you want meson fields with the same total momenta to be stored as alternative momenta (eg if you want them averaged when the meson fields are generated)
  void addAll(const MomentumAssignments &from){
    for(int i=0;i<from.nMom(); i++)
      for(int j=0;j<from.nAltMom(i);j++)
	this->addP(from.getMomAB(i,j));
  }

  void print(const std::string &name = "") const{
    LOGA2A << "Momentum policy " << name << std::endl;
    for(int p=0;p<nMom();p++){
      LOGA2A << "Total momentum " << getTotalMomentum(p) << " : ";
      for(int a=0;a<nAltMom(p);a++)
	LOGA2ANT << '[' << momA[p][a] << "+" << momB[p][a] << ']';
      LOGA2ANT << std::endl;
    }
  }

};
  



//A class to compute and store the W^dagger V meson field momenta
class RequiredMomentum: public MomentumAssignments{
  const int maxMom;
public:
  //'mp' is assigned to the maxMom member, which is used only by the (optional) addUpToMaxCOMP function
  RequiredMomentum(const int mp=0): maxMom(mp), MomentumAssignments(){ }

  //A meson field has the form (W^dagger V), so the total momentum is the sum of the V momentum and the negative of the W momentum
  //Obtain total momentum using getTotalMomentum or getVmom + getWdagMom
  
  //Get the twist momentum for the W field. Use alt_idx to access the other momentum assignments that will be all be averaged with the improve rotational symmetry
  inline ThreeMomentum getWmom(const int i, const int alt_idx = 0) const{ return ThreeMomentum::negative(this->getMomA(i,alt_idx)); } //negative because p(W) = -p(W^dag)
  inline ThreeMomentum getWdagMom(const int i, const int alt_idx = 0) const{ return this->getMomA(i,alt_idx); }
  
  //Get the twist momentum for the V field. 
  inline ThreeMomentum getVmom(const int i, const int alt_idx = 0) const{ return this->getMomB(i,alt_idx); }
  inline ThreeMomentum getVdagMom(const int i, const int alt_idx = 0) const{ return ThreeMomentum::negative(this->getMomB(i,alt_idx)); }

  //Total momentum of the meson
  inline ThreeMomentum getMesonMomentum(const int i) const{ return this->getTotalMomentum(i); }

  //For periodic BCs only, add all meson momenta with total momentum less than p^2 <= maxMom*maxMom (in units of 2pi/L)
  //All momentum assigned only to one of the two quarks
  void addUpToMaxCOMP(){
    int m=maxMom;
    for(int i=-m; i<=m; i++){
      for(int j=-m; j<=m; j++){
	for(int k=-m; k<=m; k++){
	  if(i*i+j*j+k*k>m*m) continue;
	  std::ostringstream os;
	  os << "(" << i << "," << j << "," << k << ") + (0,0,0)\n";
	  addP(os.str());
	}
      }
    }
  }
};




CPS_END_NAMESPACE

#endif
