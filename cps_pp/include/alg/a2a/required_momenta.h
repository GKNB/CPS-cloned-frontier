#ifndef _REQUIRED_MOMENTA_H
#define _REQUIRED_MOMENTA_H

#include<vector>
#include<alg/a2a/threemomentum.h>
#include<algorithm>

CPS_START_NAMESPACE

//A class to compute and store the W^dagger V meson field momenta
class RequiredMomentum{
protected:
  std::vector<std::vector<ThreeMomentum> >  wdag_mom;
  std::vector<std::vector<ThreeMomentum> >  vmom;
  std::vector<ThreeMomentum> mom_total;
  int ngp;
  const int maxMom;
  bool combine_same_total_mom; //if true, when momentum-pairs are added that have the same total momentum as an existing entry, the new pair will be added to the existing entry and not a new entry
public:
  RequiredMomentum(const int mp=0): ngp(0), maxMom(mp), combine_same_total_mom(false){
    //We require meson fields with both p and -p for each non-identical momentum direction.     
    for(int i=0;i<3;i++){
      if(GJP.Bc(i) == BND_CND_GPARITY) ngp++;
      if(i>0 && GJP.Bc(i) == BND_CND_GPARITY && GJP.Bc(i-1) != BND_CND_GPARITY){ ERR.General("RequiredMomentum","RequiredMomentum","Expect G-parity directions to be consecutive\n"); }     //(as it is setup here we expect the G-parity directions to be consecutive, i.e.  x, or x and y, or x and y and z)
    }
  }
  inline void combineSameTotalMomentum(const bool val = true){ combine_same_total_mom = val; }
  inline int nGparityDirs() const{ return ngp; }
  inline int nMom() const{ return vmom.size(); } //number of total pion momenta
  inline int nAltMom(const int mom_idx) const{ return vmom[mom_idx].size(); } //number of alternative momentum assignments that have the same total momentum
  
  //Get the twist momentum for the W field. Use alt_idx to access the other momentum assignments that will be all be averaged with the improve rotational symmetry
  inline ThreeMomentum getWmom(const int i, const int alt_idx = 0) const{ return ThreeMomentum::negative(wdag_mom[i][alt_idx]); } //negative because p(W) = -p(W^dag)
  inline ThreeMomentum getWdagMom(const int i, const int alt_idx = 0) const{ return wdag_mom[i][alt_idx]; }
  
  //Get the twist momentum for the V field. 
  inline ThreeMomentum getVmom(const int i, const int alt_idx = 0) const{ return vmom[i][alt_idx]; }
  inline ThreeMomentum getVdagMom(const int i, const int alt_idx = 0) const{ return ThreeMomentum::negative(vmom[i][alt_idx]); }

  //Total momentum of the meson
  inline ThreeMomentum getMesonMomentum(const int i) const{ return mom_total[i]; }

  void addP(const std::pair<ThreeMomentum,ThreeMomentum> &p_wdag_v){
    ThreeMomentum ptot = p_wdag_v.first + p_wdag_v.second;
    if(combine_same_total_mom)
      for(int i=0;i<mom_total.size();i++){
	if(mom_total[i] == ptot){
	  wdag_mom[i].push_back(p_wdag_v.first);
	  vmom[i].push_back(p_wdag_v.second);
	  return;
	}
      }
    mom_total.push_back(ptot);
    wdag_mom.push_back(std::vector<ThreeMomentum>(1, p_wdag_v.first));
    vmom.push_back(std::vector<ThreeMomentum>(1, p_wdag_v.second));
  }
  inline void addPandMinusP(const std::pair<ThreeMomentum,ThreeMomentum> &p_wdag_v){
    addP(p_wdag_v);
    addP(std::pair<ThreeMomentum,ThreeMomentum>(-p_wdag_v.first,-p_wdag_v.second));
  }
  
  //Add a W^dag and V momentum (respectively) from a string in the form "(%d,%d,%d) + (%d%d,%d)"
  //Momenta with the same total momentum will be added to the same mom idx
  inline void addP(const std::string &p){
    std::pair<ThreeMomentum,ThreeMomentum> p2 = ThreeMomentum::parse_str_two_mom(p);
    addP(p2);
  }    
  //Add a W^dag and V momentum (respectively) from a string in the form "(%d,%d,%d) + (%d%d,%d)", plus the negatives of these  
  inline void addPandMinusP(const std::string &p){
    std::pair<ThreeMomentum,ThreeMomentum> p2 = ThreeMomentum::parse_str_two_mom(p);
    addPandMinusP(p2);
  }

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

  //Functions to symmetrize the momentum assignment of the W^dag and V vectors
  inline void reverseVWdagMomentumAssignments(){
    std::swap(wdag_mom,vmom);
  }
  inline void symmetrizeVWdagMomentumAssignments(){
    for(int p=0;p<nMom();p++){
      int nalt = nAltMom(p);
      for(int a=0;a<nalt;a++){
	wdag_mom[p].push_back(vmom[p][a]);
	vmom[p].push_back(wdag_mom[p][a]);
      }
    }
  };

};




CPS_END_NAMESPACE

#endif
