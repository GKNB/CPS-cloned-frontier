#pragma once
#include<alg/a2a_arg.h>
#include<alg/a2a/lattice/CPSfield/CPSfield_policies.h>
#include<alg/a2a/lattice/CPSfield/CPSfield.h>

CPS_START_NAMESPACE

//If using SIMD, we don't want to vectorize across the time direction
template<typename FieldInputParamType>
struct checkSIMDparams{
  inline static void check(const FieldInputParamType &p){}
};
#ifdef USE_GRID
template<int Dimension>
struct checkSIMDparams<SIMDdims<Dimension> >{
  inline static void check(const SIMDdims<Dimension> &p){
    assert(p[3] == 1);
  }
};
#endif

//Type-labels indicating the allocation strategy
struct ManualAllocStrategy{};
struct AutomaticAllocStrategy{};

//Compute the size of the V,W fields for those types whose fields are all of a consistent type
template<typename VWtype>
inline double VW_Mbyte_size(const A2AArg &_args, const typename VWtype::FieldInputParamType &field_setup_params){
  typedef typename VWtype::DilutionType DilutionType;
  typedef typename VWtype::FermionFieldType FermionFieldType;
  DilutionType dil(_args); const int sz = dil.getNmodes();
  double field_size = double(FermionFieldType::byte_size(field_setup_params))/(1024.*1024.);
  return sz * field_size;
}

//Utility functions for IO
template<typename IOType, typename FieldType>
struct A2Avector_IOconvert{
  inline static void write(std::ostream &file, FP_FORMAT fileformat, CPSfield_checksumType cksumtype, const FieldType &f, IOType &tmp){
    tmp.importField(f);
    tmp.writeParallel(file,fileformat,cksumtype);
  }
  inline static void read(std::istream &file, FieldType &f, IOType &tmp){    
    tmp.readParallel(file);
    f.importField(tmp);    
  }
  
};
template<typename IOType>
struct A2Avector_IOconvert<IOType,IOType>{
  inline static void write(std::ostream &file, FP_FORMAT fileformat, CPSfield_checksumType cksumtype, const IOType &f, IOType &tmp){
    f.writeParallel(file,fileformat,cksumtype);
  }
  inline static void read(std::istream &file, IOType &f, IOType &tmp){
    f.readParallel(file);
  }
};

//Compute the base momentum and Cshift
template< typename FieldType>
FieldType const * getBaseAndShift(int shift[3], const int p[3], FieldType const *base_p, FieldType const *base_m){
  //With G-parity base_p has momentum +1 in each G-parity direction, base_m has momentum -1 in each G-parity direction.
  //Non-Gparity directions are assumed to have momentum 0

  //Units of momentum are 2pi/L for periodic BCs, pi/L for antiperiodic and pi/2L for Gparity
  FieldType const * out = GJP.Gparity() ? NULL : base_p;
  for(int d=0;d<3;d++){
    if(GJP.Bc(d) == BND_CND_GPARITY){
      //Type 1 : f_{p=4b+1}(n) = f_+1(n+b)     // p \in {.. -7 , -3, 1, 5, 9 ..}
      //Type 2 : f_{p=4b-1}(n) = f_-1(n+b)     // p \n  {.. -5, -1, 3, 7 , 11 ..}
      if( (p[d]-1) % 4 == 0 ){
	//Type 1
	int b = (p[d]-1)/4;
	shift[d] = -b;  //shift f_+1 backwards by b
	if(out == NULL) out = base_p;
	else if(out != base_p) ERR.General("","getBaseAndShift","Momentum (%d,%d,%d) appears to be invalid because momenta in different G-parity directions do not reside in the same set\n",p[0],p[1],p[2]);
	
      }else if( (p[d]+1) % 4 == 0 ){
	//Type 2
	int b = (p[d]+1)/4;
	shift[d] = -b;  //shift f_-1 backwards by b
	if(out == NULL) out = base_m;
	else if(out != base_m) ERR.General("","getBaseAndShift","Momentum (%d,%d,%d) appears to be invalid because momenta in different G-parity directions do not reside in the same set\n",p[0],p[1],p[2]);
	
      }else ERR.General("","getBaseAndShift","Momentum (%d,%d,%d) appears to be invalid because one or more components in G-parity directions are not allowed\n",p[0],p[1],p[2]);
    }else{
      //f_b(n) = f_0(n+b)
      //Let the other directions decide on which base to use if some of them are G-parity dirs ; otherwise the pointer defaults to base_p above
      shift[d] = -p[d];
    }
  }
  a2a_printf("getBaseAndShift for p=(%d,%d,%d) determined shift=(%d,%d,%d) from ptr %c\n",p[0],p[1],p[2],shift[0],shift[1],shift[2],out == base_p ? 'p' : 'm');
  assert(out != NULL);
  
  return out;
}

CPS_END_NAMESPACE
