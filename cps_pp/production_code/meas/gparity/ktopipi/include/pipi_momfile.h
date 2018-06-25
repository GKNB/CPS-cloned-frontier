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

  if(!UniqueID()){ std::cout << "parsePiPiMomFile: Checking file " << file << std::endl; fflush(stdout); }

  while(!f.eof()){
    int size;
    f >> size;
    
    uint32_t cksum_in;
    f >> cksum_in;

    if(!UniqueID()){ std::cout << "Got a size " << size << " and cksum " << cksum_in << std::endl; }

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

#endif
