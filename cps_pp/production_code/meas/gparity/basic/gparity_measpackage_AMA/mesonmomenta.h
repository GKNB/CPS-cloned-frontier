#ifndef _MESON_MOMENTA_H
#define _MESON_MOMENTA_H

#include <alg/a2a/threemomentum.h>

CPS_START_NAMESPACE

enum QuarkType { Light, Heavy };

//Storage for unique quark momenta needed (for pairs +p and -p, we can trivially reconstruct the flipped momentum using the G-parity conjugate relation without having to do extra inversions)
class QuarkMomenta{
  std::vector<ThreeMomentum> mom;
public:
  inline void add(const ThreeMomentum &val){
    //if(!UniqueID()) std::cout << "QuarkMomenta adding mom " << val.str() << '\n';
    for(int i=0;i<mom.size();i++) 
      if(mom[i] == val || mom[i] == -val){
	//if(!UniqueID()) std::cout << "Skipping as it (or its negative) already exists: " << mom[i].str() << '\n';
	return;
      }

    mom.push_back(val);
  }
  inline int nMom() const{ return mom.size(); }
  inline const ThreeMomentum & getMom(const int pidx) const{ return mom[pidx]; }
};




//For 2pt functions of the form  Tr[ (prop1)^dag A prop2 B ]
//store the momenta of the propagators and compute the total meson momentum
class MesonMomenta{
  typedef std::pair<QuarkType,ThreeMomentum> Mtype;
  std::vector<Mtype> prop1dag_mom;
  std::vector<Mtype> prop2_mom;
  
public:
  inline int nMom() const{ return prop2_mom.size(); }
  
  inline ThreeMomentum getQuarkMom(const int quark_idx, const int mom_idx) const{
    return quark_idx == 0 ? -prop1dag_mom[mom_idx].second : prop2_mom[mom_idx].second; //note the - sign for prop1 due to the dagger swapping the momentum
  }
  inline ThreeMomentum getMesonMom(const int mom_idx) const{ return prop1dag_mom[mom_idx].second + prop2_mom[mom_idx].second; }

  inline QuarkType getQuarkType(const int quark_idx, const int mom_idx) const{
    return quark_idx == 0 ? prop1dag_mom[mom_idx].first : prop2_mom[mom_idx].first; 
  }

  void printAllCombs(const std::string & descr = "") const{
    if(!UniqueID()){
      printf("Momentum combinations %s:\n",descr.c_str());
      for(int i=0;i<nMom();i++){
	std::cout << "(" << (prop1dag_mom[i].first == Light ? "light" : "heavy") << ") " << prop1dag_mom[i].second.str() << " + ";
	std::cout << "(" << (prop2_mom[i].first == Light ? "light" : "heavy") << ") " << prop2_mom[i].second.str() << '\n';
      }
    }
  }
    

  //Add a (prop1)^dag and (prop2) momentum (respectively) from a string in the form "(%d,%d,%d) + (%d%d,%d)"
  void addP(const std::string &p, const QuarkType qtype1,  const QuarkType qtype2){
    std::pair<ThreeMomentum,ThreeMomentum> p2 = ThreeMomentum::parse_str_two_mom(p);
    //if(!UniqueID()) std::cout << "MesonMomenta::addP got mompair '" << p << "' and parsed as " << p2.first.str() << " and " << p2.second.str() << '\n';
    prop1dag_mom.push_back(Mtype(qtype1,p2.first ));
    prop2_mom   .push_back(Mtype(qtype2,p2.second));
  }

  //Add to QuarkMomenta all the required propagator source momenta of a particular quark species (heavy/light)
  void appendQuarkMomenta(const QuarkType qtype,QuarkMomenta &qmom) const{
    for(int i=0;i<prop1dag_mom.size();i++)
      if(prop1dag_mom[i].first == qtype) 
	qmom.add(-prop1dag_mom[i].second); //note minus sign again
    for(int i=0;i<prop2_mom.size();i++)
      if(prop2_mom[i].first == qtype) 
	qmom.add(prop2_mom[i].second); //note no minus sign
  }

};



inline int getNgp(){
  int ngp = 0;
  for(int i=0;i<3;i++){
    if(GJP.Bc(i) == BND_CND_GPARITY) ngp++;
    if(i>0 && GJP.Bc(i) == BND_CND_GPARITY && GJP.Bc(i-1) != BND_CND_GPARITY){ ERR.General("","getNgp","Expect G-parity directions to be consecutive\n"); } //(as it is setup here we expect the G-parity directions to be consecutive, i.e.  x, or x and y, or x and y and z)
  }
  return ngp;
}



class PionMomenta{
public:
  static void setup(MesonMomenta &into, bool include_alternative_mom = true){
    int ngp = getNgp();
    
    if(ngp == 0){
      //p_pi = (0,0,0)
      into.addP("(0,0,0) + (0,0,0)",Light,Light);
    }else if(ngp == 1){
      //p_pi = (2,0,0)     (units of pi/2L)    
      into.addP("(1,0,0) + (1,0,0)",Light,Light); 
      if(include_alternative_mom) into.addP("(-1,0,0) + (3,0,0)",Light,Light); 
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (2,2,0)     (units of pi/2L)  
      into.addP("(1,1,0) + (1,1,0)",Light,Light);
      if(include_alternative_mom) into.addP("(-1,-1,0) + (3,3,0)",Light,Light);

      //Along off-diagonal direction:      
      //p_pi = (-2,2,0)
      into.addP("(1,1,0) + (-3,1,0)",Light,Light); 
      if(include_alternative_mom) into.addP("(-1,-1,0) + (-1,3,0)",Light,Light);
    }else if(ngp == 3){
      //p_pi = (2,2,2)     (units of pi/2L)  
      into.addP("(1,1,1) + (1,1,1)",Light,Light); 
      if(include_alternative_mom) into.addP("(-1,-1,-1) + (3,3,3)",Light,Light);

      //p_pi = (-2,2,2) //only do one off-diagonal as we have symmetry around that axis
      into.addP("(1,1,1) + (-3,1,1)",Light,Light);
      if(include_alternative_mom) into.addP("(-1,-1,-1) + (-1,3,3)",Light,Light);
    }else{
      ERR.General("PionMomenta","setup","ngp cannot be >3\n");
    }
  }
};


//This is the stationary \bar\psi \gamma^5 \psi pseudoscalar flavor-singlet 
class LightFlavorSingletMomenta{
public:
  static void setup(MesonMomenta &into){
    int ngp = getNgp();
    
    if(ngp == 0){
      //p_pi = (0,0,0)
      into.addP("(0,0,0) + (0,0,0)",Light,Light);
    }else if(ngp == 1){
      //p_pi = (0,0,0)     (units of pi/2L)    
      into.addP("(-1,0,0) + (1,0,0)",Light,Light); 
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (0,0,0)     (units of pi/2L)  
      into.addP("(-1,-1,0) + (1,1,0)",Light,Light);
    }else if(ngp == 3){
      //p_pi = (0,0,0)     (units of pi/2L)  
      into.addP("(-1,-1,-1) + (1,1,1)",Light,Light); 
    }else{
      ERR.General("LightFlavorSingletMomenta","setup","ngp cannot be >3\n");
    }
  }
};

//Note the heavy quark propagator (the one that is daggered - cf Eq 191 of the paper) is assigned the - momentum
class KaonMomenta{
public:
  static void setup(MesonMomenta &into){
    int ngp = getNgp();
    
    if(ngp == 0){
      //p_pi = (0,0,0)
      into.addP("(0,0,0) + (0,0,0)",Heavy,Light);
    }else if(ngp == 1){
      //p_pi = (0,0,0)     (units of pi/2L)    
      into.addP("(-1,0,0) + (1,0,0)",Heavy,Light); 
    }else if(ngp == 2){
      //Along G-parity direction:
      //p_pi = (0,0,0)     (units of pi/2L)  
      into.addP("(-1,-1,0) + (1,1,0)",Heavy,Light);
    }else if(ngp == 3){
      //p_pi = (0,0,0)     (units of pi/2L)  
      into.addP("(-1,-1,-1) + (1,1,1)",Heavy,Light); 
    }else{
      ERR.General("KaonMomenta","setup","ngp cannot be >3\n");
    }
  }
};

CPS_END_NAMESPACE

#endif
