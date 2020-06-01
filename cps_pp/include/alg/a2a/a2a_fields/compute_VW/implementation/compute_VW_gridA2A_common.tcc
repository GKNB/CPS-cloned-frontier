//Apply 1/2(1+-g5) to field. In Grid conventions this just zeroes the lower/upper spin components
template<typename FermionField>
void chiralProject(FermionField &out, const FermionField &in, const char sgn){
  int base; //where to start zeroing
  switch(sgn){
  case '+':
    base = 2;
    break;
  case '-':
    base = 0;
    break;
  default:
    assert(0);
  }
  
  out.Checkerboard() = in.Checkerboard();
  conformable(in,out);

  const int Ns = 4;
  Grid::GridBase *grid=in.Grid();

  //decltype(Grid::peekSpin(static_cast<const Grid::Lattice<typename FermionField::vector_object>&>(in),0)) zero_spn(in._grid);
  decltype(Grid::PeekIndex<SpinIndex>(in,0)) zero_spn(in.Grid());
  Grid::zeroit(zero_spn);

  out = in;
  Grid::PokeIndex<SpinIndex>(out, zero_spn, base);
  Grid::PokeIndex<SpinIndex>(out, zero_spn, base+1);
}

//Convert a 5D field to a 4D field, with the upper 2 spin components taken from s-slice 's_u' and the lower 2 from 's_l'
template<typename FermionField>
void DomainWallFiveToFour(FermionField &out, const FermionField &in, int s_u, int s_l){
  assert(out.Grid()->Nd() == 4 && in.Grid()->Nd() == 5);

  FermionField tmp1_4d(out.Grid());
  FermionField tmp2_4d(out.Grid());
  FermionField tmp3_4d(out.Grid());
  ExtractSlice(tmp1_4d,const_cast<FermionField&>(in),s_u, 0); //Note Grid conventions, s-dimension is index 0!
  chiralProject(tmp2_4d, tmp1_4d, '+'); // 1/2(1+g5)  zeroes lower spin components
  
  ExtractSlice(tmp1_4d,const_cast<FermionField&>(in),s_l, 0); 
  chiralProject(tmp3_4d, tmp1_4d, '-'); // 1/2(1-g5)  zeroes upper spin components

  out = tmp2_4d + tmp3_4d;
}
template<typename FermionField>
void DomainWallFourToFive(FermionField &out, const FermionField &in, int s_u, int s_l){
  assert(out.Grid()->Nd() == 5 && in.Grid()->Nd() == 4);

  zeroit(out);
  FermionField tmp1_4d(in.Grid());
  chiralProject(tmp1_4d, in, '+'); // 1/2(1+g5)  zeroes lower spin components
  InsertSlice(tmp1_4d, out,s_u, 0);

  chiralProject(tmp1_4d, in, '-'); // 1/2(1-g5)  zeroes upper spin components
  InsertSlice(tmp1_4d, out,s_l, 0);
}



inline bool isMultiCG(const A2ACGalgorithm al){
  if(al == AlgorithmMixedPrecisionReliableUpdateSplitCG) return true;
  return false;
}
