#include<array>
#include<map>
#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include<cassert>
#include<set>
#include<list>
#include<fstream>
#include<cstdio>

#include <zlib.h>

#include "mom_type.h"
#include "symmetry_ops.h"
#include "correlator.h"
#include "compute_sets.h"

//Try to find the minimum set of meson fields required
std::set<mom> minimizeMesonFields(const std::vector<std::vector<Correlator> > &sets){
  std::set<mom> pm;

  //Add the correlator from each set that has the most matches to meson fields already inserted
  for(int s=0;s<sets.size();s++){
    int most_matches = -1;
    int most_matches_idx = -1;

    for(int i=0;i<sets[s].size();i++){
      int nmatch = 0;
      if(pm.count(sets[s][i].p1src)) ++nmatch;
      if(pm.count(sets[s][i].p2src)) ++nmatch;
      if(pm.count(sets[s][i].p1snk)) ++nmatch;
      if(pm.count(sets[s][i].p2snk)) ++nmatch;
      
      if(nmatch > most_matches){
        most_matches = nmatch;
	most_matches_idx = i;
      }
    }
    assert(most_matches_idx >= 0);

    pm.insert(sets[s][most_matches_idx].p1src);
    pm.insert(sets[s][most_matches_idx].p2src);
    pm.insert(sets[s][most_matches_idx].p1snk);
    pm.insert(sets[s][most_matches_idx].p2snk);
  }
  
  return pm;
}

//Write to a text file the base (representative) correlators of each set (taken to be the first), one per line, in format
//p1src p2src p1snk p2snk 
//with each momenta written in component form with no parentheses or commas, eg (1,2,3) -> "1 2 3"

//Also write size and crc32 checksum

inline uint32_t crc32_mom(const mom &p, const uint32_t in){ 
  return (uint32_t)crc32(in, (const unsigned char*)p.data(), 3*sizeof(int));
}

inline void writeMomFile(std::ostream &os, const mom &mom){ os << mom[0] << " " << mom[1] << " " << mom[2]; }

void writeBaseCorrelators(std::ofstream &of, const std::vector<std::vector<Correlator> > &sets){
  //Write size
  of << sets.size() << " ";
  
  //Compute and write crc32 checksum
  uint32_t crc = crc32(0L,Z_NULL,0);
  for(int s=0;s<sets.size();s++){
    crc = crc32_mom(sets[s][0].p1src, crc);
    crc = crc32_mom(sets[s][0].p2src, crc);
    crc = crc32_mom(sets[s][0].p1snk, crc);
    crc = crc32_mom(sets[s][0].p2snk, crc);
  }
  
  of << crc << std::endl;

  for(int s=0;s<sets.size();s++){
    writeMomFile(of, sets[s][0].p1src); of << " ";
    writeMomFile(of, sets[s][0].p2src); of << " ";
    writeMomFile(of, sets[s][0].p1snk); of << " ";
    writeMomFile(of, sets[s][0].p2snk); of << "\n";
  }
};

////////////////////FOR MAIN PROGRAM/////////////////
struct ThreeMomentum{
  int p[3];
  int &operator()(const int i){ return p[i]; }
  const int operator()(const int i) const{ return p[i]; }
  bool operator==(const mom &m){ return m[0] == p[0] && m[1] == p[1] && m[2] == p[2]; }
};

inline uint32_t crc32_3mom(const ThreeMomentum &p, const uint32_t in){ 
  return (uint32_t)crc32(in, (const unsigned char*)p.p, 3*sizeof(int));
}


struct CorrelatorMomenta{
  ThreeMomentum pi1_src;
  ThreeMomentum pi2_src;
  ThreeMomentum pi1_snk;
  ThreeMomentum pi2_snk;
};

void parseMom(ThreeMomentum &into, std::istream &f){
  f >> into(0) >> into(1) >> into(2);
}


void parsePiPiMomFile(std::vector<CorrelatorMomenta> &correlators, const std::string &file){
  std::ifstream f(file.c_str());
  assert(f.is_open() && f.good());
  f.exceptions ( std::ifstream::failbit | std::ifstream::badbit );

  std::cout << "Checking file " << file << std::endl;

  while(!f.eof()){
    int size;
    f >> size;
    
    uint32_t cksum_in;
    f >> cksum_in;

    std::cout << "Got a size " << size << " and cksum " << cksum_in << std::endl;

    uint32_t cksum = crc32(0L,Z_NULL,0);
    
    for(int s=0;s<size;s++){
      CorrelatorMomenta c;
      parseMom(c.pi1_src,f);
      parseMom(c.pi2_src,f);
      parseMom(c.pi1_snk,f);
      parseMom(c.pi2_snk,f);

      correlators.push_back(c);

      cksum = crc32_3mom(c.pi1_src,cksum);
      cksum = crc32_3mom(c.pi2_src,cksum);
      cksum = crc32_3mom(c.pi1_snk,cksum);
      cksum = crc32_3mom(c.pi2_snk,cksum);
    }

    if(cksum != cksum_in){
      printf("Set of size %d and checksum %lu failed cksum check, got %lu\n",size,cksum_in,cksum); fflush(stdout);
      exit(-1);
    }
    if(f.peek()==10){
      f.get();
      f.peek(); //triggers eofbit if now at end of file
    }
  }
  f.close();
}





int main(void){
  std::vector<mom> p_orig = { {1,1,1}, {-1,-1,-1},
  			 {1,1,-1}, {-1,-1,1},
  			 {1,-1,1}, {-1,1,-1},
  			 {-1,1,1}, {1,-1,-1} };

  std::vector<mom> p_3_base = { {3,1,1}, {-3,-1,-1},
				{3,1,-1}, {-3,-1,1},
				{3,-1,1}, {-3,1,-1},
				{3,-1,-1}, {-3,1,1} };
  
  //Assemble set of all momenta!
  std::vector<mom> R = p_orig;
  for(int n=0;n<3;n++){
    for(int i=0;i<p_3_base.size();i++)
      R.push_back(cyclicPermute(p_3_base[i],n));
  };
  
  //Symmetry operations that can be used to transform the total momenta provided when building the initial set of correlators that are allowed to be treated as equal provided there is a correlator symmetry between them
  bool allow_parity_ptot = false;
  bool allow_axis_perm_ptot = false;

  //Symmetry operations that can be used to relate correlation functions
  bool allow_aux_diag = true ;
  bool allow_parity_reln = true;
  bool allow_axis_perm_reln = true ;

  //The base set of source total momenta that are considered as distinct (not allowed to be tested for relation by correlator symmetries)
  //std::vector<mom> ptots = { {0,0,0},{2,0,0}, {2,2,0}, {2,2,2} };
  //std::vector<mom> ptots = { {2,2,0}, {2,0,2}, {0,2,2}, {-2,2,0}, {-2,0,2}, {0,-2,2} };
  //std::vector<mom> ptots = {  {2,2,2} };

  std::vector<mom> ptots = { 
    {0,0,0},
    {2,0,0}, {0,2,0},{0,0,2}, 
    {2,2,0}, {2,0,2}, {0,2,2}, {-2,2,0}, {-2,0,2}, {0,-2,2},	    
    {2,2,2}, {-2,2,2}, {2,-2,2}, {2,2,-2}
  };


  std::ofstream fout("pipi_correlators.in");

  for(int pp=0;pp<ptots.size();pp++){
    mom ptot = ptots[pp];
    std::cout << "Total momentum " << ptot << std::endl;
    
    std::list<Correlator> correlators = getCorrelators(R, ptot, allow_parity_ptot, allow_axis_perm_ptot);
    std::cout << "Base set of correlators has size " << correlators.size() << std::endl;

    std::vector<std::vector<Correlator> > sets = computeSets(correlators, allow_parity_reln, allow_axis_perm_reln, allow_aux_diag);

    testSets(sets, correlators, allow_parity_reln, allow_axis_perm_reln, allow_aux_diag);
    for(int s=0;s<sets.size();s++){
      std::cout << sets[s][0] << " " << sets[s].size() << ":\n";
      for(int i=0;i<sets[s].size();i++)
      std::cout << " " << sets[s][i];
      std::cout << std::endl;
    }

    std::cout << "Total sets: " << sets.size() << std::endl;
    
    std::set<mom> mf = minimizeMesonFields(sets);
    
    std::cout << "Requires " << mf.size() << " meson fields with momenta:\n";
    for(auto it = mf.begin(); it != mf.end(); it++)
      std::cout << *it << std::endl;

    for(int e=0;e<sets.size();e++){
      printf("allow[ momPair(threeMomentum({%d,%d,%d}), threeMomentum({%d,%d,%d}) ) ] = %d;\n",
	     sets[e][0].p1src[0],sets[e][0].p1src[1],sets[e][0].p1src[2],
	     sets[e][0].p1snk[0],sets[e][0].p1snk[1],sets[e][0].p1snk[2],
	     sets[e].size());
    }

    for(int e=0;e<sets.size();e++){
      printf("(%d,%d,%d) & (%d,%d,%d) & %d\\\\\n", 
	     sets[e][0].p1src[0],sets[e][0].p1src[1],sets[e][0].p1src[2],
	     sets[e][0].p1snk[0],sets[e][0].p1snk[1],sets[e][0].p1snk[2],
	     sets[e].size());
    }

    writeBaseCorrelators(fout, sets);
  }
  
  fout.close();

  //Check we can read them back
  std::vector<CorrelatorMomenta> correlators;
  parsePiPiMomFile(correlators, "pipi_correlators.in");

  std::cout << "Done" << std::endl;

  return 0;
}
