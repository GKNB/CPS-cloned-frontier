struct Correlator{
  mom p1src;
  mom p2src;

  mom p1snk;
  mom p2snk;

  Correlator(){}
  Correlator(const mom &p1src, const mom &p2src, const mom &p1snk, const mom &p2snk): p1src(p1src), p2src(p2src), p1snk(p1snk), p2snk(p2snk) {  }
  Correlator(const momPair &psrc, const momPair &psnk): p1src(psrc.first), p2src(psrc.second), p1snk(psnk.first), p2snk(psnk.second){}

  bool operator<(const Correlator &b) const{
    return std::tie(p1src, p2src, p1snk, p2snk) < std::tie(b.p1src, b.p2src, b.p1snk, b.p2snk);
  }
};

std::ostream & operator<<(std::ostream &os, const Correlator &c){
  os << "| " << c.p1src << " " << c.p1snk << " " << "[" << c.p1src+c.p2src << "] |";
  return os;
}

inline Correlator applySymmetryOp(const Correlator &c, const int symm){
  return Correlator(applySymmetryOp(c.p1src,symm),
		    applySymmetryOp(c.p2src,symm),
		    applySymmetryOp(c.p1snk,symm),
		    applySymmetryOp(c.p2snk,symm));
}

inline Correlator auxDiag(const Correlator &c){
  //C(p_1^src, p_2^src, p_1^snk, p_2^snk) = C(p_2^snk, p_1^snk, p_2^src, p_1^src)
  return Correlator(c.p2snk, c.p1snk, c.p2src, c.p1src);
}

bool hasReln(Correlator a, Correlator b, const bool allow_parity, const bool allow_axis_perm, const bool allow_aux_diag){
  for(int aux=0;aux<(int)allow_aux_diag + 1;aux++){
    if(aux) b = auxDiag(b);

    //Get all allowed operations for p1src
    std::vector<Operations> ops1st = getReln(a.p1src, b.p1src, allow_parity, allow_axis_perm); //transform for b.first to match a.first
    
    //See if one or more applied to the other momenta work
    for(int i=0;i<ops1st.size();i++){
      if(a.p2src == applyOp(b.p2src, ops1st[i])  &&
	 a.p1snk == applyOp(b.p1snk, ops1st[i])  &&
	 a.p2snk == applyOp(b.p2snk, ops1st[i]) ){

	//if(aux) std::cout << "Found reln involving aux diag!\n";

	return true;
      }
    }
  }
  return false;
}
