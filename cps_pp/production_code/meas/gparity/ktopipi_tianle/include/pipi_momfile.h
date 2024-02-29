#ifndef _PIPI_MOMFILE_H_
#define _PIPI_MOMFILE_H_

struct CorrelatorMomenta{
  ThreeMomentum pi1_src;
  ThreeMomentum pi2_src;
  ThreeMomentum pi1_snk;
  ThreeMomentum pi2_snk;
};

inline void parseMom(ThreeMomentum &into, std::istream &f){
  f >> into(0) >> into(1) >> into(2);
}
//Little-endian crc32!
inline uint32_t crc32_3mom(const ThreeMomentum &p, const uint32_t in, hostEndian::EndianType endian){ 
  if(endian == hostEndian::LITTLE){
    return (uint32_t)crc32(in, (const unsigned char*)p.ptr(), 3*sizeof(int));
  }else{
    char little[3*sizeof(int)];
    for(int d=0;d<3;d++){
      int p_d = p(d);
      char const* big = (char const*)&p_d;
      char* to_base = little + d*sizeof(int);
      int v = sizeof(int) - 1;
      for(int b=0;b<sizeof(int);b++) to_base[b] = big[v--];
    }
    return (uint32_t)crc32(in, (unsigned char const*)&little[0], 3*sizeof(int));
  }
}

void parsePiPiMomFile(std::vector<CorrelatorMomenta> &correlators, const std::string &file){
  std::ifstream f(file.c_str());
  if(!f.is_open() || !f.good()) ERR.General("","parsePiPiMomFile","Node %d failed to open file %s\n",UniqueID(),file.c_str());
  f.exceptions ( std::ifstream::failbit | std::ifstream::badbit );

  hostEndian::EndianType endian = hostEndian::get();

  LOGA2A << "parsePiPiMomFile: Checking file " << file << std::endl;

  while(!f.eof()){
    int size;
    f >> size;
    
    uint32_t cksum_in;
    f >> cksum_in;

    LOGA2A << "Got a size " << size << " and cksum " << cksum_in << std::endl;

    uint32_t cksum = crc32(0L,Z_NULL,0);
    
    for(int s=0;s<size;s++){
      CorrelatorMomenta c;
      parseMom(c.pi1_src,f);
      parseMom(c.pi2_src,f);
      parseMom(c.pi1_snk,f);
      parseMom(c.pi2_snk,f);

      cksum = crc32_3mom(c.pi1_src,cksum,endian);
      cksum = crc32_3mom(c.pi2_src,cksum,endian);
      cksum = crc32_3mom(c.pi1_snk,cksum,endian);
      cksum = crc32_3mom(c.pi2_snk,cksum,endian);

      c.pi1_src *= 2; //units of pi/2L
      c.pi1_snk *= 2;
      c.pi2_src *= 2;
      c.pi2_snk *= 2;

      correlators.push_back(c);
    }

    if(cksum != cksum_in)
      ERR.General("","parsePiPiMomFile","Node %d, set of size %d and checksum %lu failed cksum check, got %lu (%lu)\n",UniqueID(),size,cksum_in,cksum); 

    if(f.peek()==10){
      f.get();
      f.peek(); //triggers eofbit if now at end of file
    }
  }
  f.close();
}


//Pick one of the 4 momenta by index
inline const ThreeMomentum & getP(const CorrelatorMomenta &p, const int i){
  switch(i){
  case 0:
    return p.pi1_src;
  case 1:
    return p.pi2_src;
  case 2:
    return p.pi1_snk;
  case 3:
    return p.pi2_snk;
  default:
    assert(0);
  }
}

bool contains(const ThreeMomentum &p, const CorrelatorMomenta &in){
  for(int i=0;i<4;i++) if(getP(in,i) == p) return true;
  return false;
}

  
//Sort the CorrelatorMomenta array by its 4 momenta in the specified order
std::vector<CorrelatorMomenta> sort(const std::vector<CorrelatorMomenta> &correlators, const std::array<int,4> &order){
  std::vector<CorrelatorMomenta> cp(correlators);
  std::sort(cp.begin(),cp.end(),
	    [&order](const CorrelatorMomenta &a, const CorrelatorMomenta &b){
	      ThreeMomentum const* ap[4] = { &getP(a,order[0]), &getP(a,order[1]), &getP(a,order[2]), &getP(a,order[3]) };
	      ThreeMomentum const* bp[4] = { &getP(b,order[0]), &getP(b,order[1]), &getP(b,order[2]), &getP(b,order[3]) };
	      for(int i=0;i<4;i++){
		if( *ap[i] < *bp[i] ) return true;
		else if( *ap[i] > *bp[i] ) return false;
	      }
	      return false; //all equal
	    });
  return cp;
}


//Determine the amount of reuse of momenta assuming execution in order
int computeReuse(const std::vector<CorrelatorMomenta> &correlators){
  int reuse = 0;
  for(int c=1;c<correlators.size();c++){
    const CorrelatorMomenta &cur = correlators[c];
    const CorrelatorMomenta &prev = correlators[c-1];
    for(int p=0;p<4;p++)
      if(getP(cur,p) == getP(prev,p)) ++reuse;
  }
  return reuse;
}


//To maximimize reuse of meson fields we can reorder the correlators
void optimizePiPiMomentumOrdering(std::vector<CorrelatorMomenta> &correlators){
  //We want to sort by the four momenta, but it is not clear what order to sort
  //Instead we try all different orderings and choose the one with the most reuse

  //Enumerate permutations of 0,1,2,3
  int nperm = 4*3*2;
  std::vector<std::array<int,4> > perms(nperm);
  int c=0;
  for(int i=0;i<4;i++)
    for(int j=0;j<4;j++)
      for(int k=0;k<4;k++)
	for(int l=0;l<4;l++)
	  if( i!=j && i!= k && i!=l && j!=k && j!=l && k!=l ){
	    assert(c<nperm);
	    perms[c++] = {i,j,k,l};
	  }

  //Sort according to each different ordering and choose the combination that maximizes the reuse
  std::vector<CorrelatorMomenta> best;
  int max_reuse = -1;

  for(int i=0;i<nperm;i++){
    std::vector<CorrelatorMomenta> sorted = sort(correlators, perms[i]);
    int reuse = computeReuse(sorted);
    LOGA2A << "optimizePiPiMomentumOrdering permutation " << perms[i][0] << "," << perms[i][1] << "," << perms[i][2] << "," << perms[i][3] << " reuse, " << reuse << ", best " << max_reuse << std::endl;
    if(reuse > max_reuse){
      max_reuse = reuse;
      best = std::move(sorted);
    }    
  }
  correlators = std::move(best);
}



#endif
