#ifndef _FMATRIX_H
#define _FMATRIX_H

#include<string>
#include<util/gjp.h>
#include<util/qcdio.h>
#include<alg/a2a/utils.h>

CPS_START_NAMESPACE

class StandardAllocPolicy;

template<typename T, typename AllocPolicy = StandardAllocPolicy>
class basicMatrix : public AllocPolicy{
  T* tt;
  int rows, cols;
  int fsize; //number of elements
  
  inline void free(){
    if(tt!=NULL) AllocPolicy::_free(tt);
  }

  inline void alloc(const int _rows, const int _cols){
    rows = _rows; cols = _cols; fsize = rows*cols;
    AllocPolicy::_alloc((void**)&tt, fsize*sizeof(T));
  }

public:
  basicMatrix(): rows(0),cols(0),fsize(0),tt(NULL){ }

  basicMatrix(const int _rows, const int _cols){
    this->alloc(_rows,_cols);
  }
  basicMatrix(const basicMatrix<T> &r){
    this->alloc(r.rows,r.cols);
    for(int i=0;i<fsize;i++) tt[i] = r.tt[i];
  }
  
  T* ptr(){ return tt;}
  T const* ptr() const{ return tt;}
  
  int size() const{ return fsize; }
  
  void resize(const int _rows, const int _cols){
    if(tt == NULL || _rows * _cols != fsize){
      this->free();
      this->alloc(_rows,_cols);
    }
  }

  inline const T & operator()(const int i, const int j) const{ return tt[j + cols*i]; }
  inline T & operator()(const int i, const int j){ return tt[j + cols*i]; }

  inline int nRows() const{ return rows; }
  inline int nCols() const{ return cols; }

  ~basicMatrix(){
    this->free();
  }
};


//A matrix of complex numbers and some useful associated methods
template<typename mf_Complex>
class fMatrix{
  mf_Complex* tt;
  int rows, cols;
  int fsize; //number of elements
  
  void free_matrix(){
    if(tt!=NULL) sfree("fMatrix","~fMatrix","free",tt);
  }

  void alloc_matrix(const int _rows, const int _cols, mf_Complex const* cp = NULL){
    if(_rows != rows || _cols != cols){
      free_matrix();
      rows = _rows; cols = _cols; fsize = rows*cols;
      tt = (mf_Complex*)smalloc("fMatrix", "fMatrix", "alloc" , sizeof(mf_Complex) * fsize);
    }
    if(cp == NULL) zero();
    else for(int i=0;i<fsize;i++) tt[i] = cp[i];
  }

public:
  fMatrix(): rows(0),cols(0),fsize(0),tt(NULL){ }

  fMatrix(const int _rows, const int _cols): rows(0), cols(0), fsize(0),tt(NULL){ 
    alloc_matrix(_rows,_cols);
  }
  fMatrix(const fMatrix<mf_Complex> &r): rows(0), cols(0), fsize(0),tt(NULL){
    alloc_matrix(r.rows,r.cols,r.tt);
  }

  fMatrix(fMatrix<mf_Complex> &&r): rows(r.rows), cols(r.cols), fsize(r.fsize),tt(r.tt){
    r.tt = nullptr;
    r.fsize = r.rows = r.cols = 0;
  }
  
  mf_Complex *ptr(){ return tt;}

  void resize(const int _rows, const int _cols){ alloc_matrix(_rows,_cols); }

  void zero(){ for(int i=0;i<fsize;i++) tt[i] = 0.0; }

  fMatrix & operator*=(const mf_Complex &r){ for(int i=0;i<fsize;i++) tt[i] *= r;  return *this; }
  fMatrix & operator*=(const typename mf_Complex::value_type &r){ for(int i=0;i<fsize*2;i++) ((typename mf_Complex::value_type*)tt)[i] *= r;  return *this; }

  fMatrix & operator+=(const fMatrix<mf_Complex> &r){ for(int i=0;i<fsize;i++) tt[i] += r.tt[i];  return *this; }

  bool equals(const fMatrix<mf_Complex> &r, double tol = 1e-10){
    if(this->nRows() != r.nRows()) return false;
    if(this->nCols() != r.nCols()) return false;
    for(int i=0;i<this->nRows();i++){
      for(int j=0;j<this->nCols();j++){
	const mf_Complex &aa = (*this)(i,j);
	const mf_Complex &bb = r(i,j);	
	if(fabs(aa.real() - bb.real()) > tol)
	  return false;
	if(fabs(aa.imag() - bb.imag()) > tol)
	  return false;
      }
    }
    return true;
  }
  
  fMatrix operator*(const fMatrix<mf_Complex> &r) const{
    assert(this->nCols() == r.nRows());
    fMatrix<mf_Complex> out(this->nRows(), r.nCols());
    out.zero();
    
    for(int i=0;i<this->nRows();i++){
      for(int j=0;j<r.nCols();j++){
	for(int k=0;k<this->nCols();k++){
	  out(i,j) += (*this)(i,k) * r(k,j);
	}
      }
    }
    return out;
  }
  
  inline const mf_Complex & operator()(const int i, const int j) const{ return tt[j + cols*i]; }
  inline mf_Complex & operator()(const int i, const int j){ return tt[j + cols*i]; }

  inline int nRows() const{ return rows; }
  inline int nCols() const{ return cols; }

  void nodeSum(){
    globalSum( (typename mf_Complex::value_type*)tt,2*fsize);
  }

  ~fMatrix(){
    free_matrix();
  }

  //hexfloat option: For reproducibility testing, write the output in hexfloat format rather than truncating the precision
  void write(const std::string &filename, const bool hexfloat = false) const{
    const char* fmt = hexfloat ? "%d %d %a %a\n" : "%d %d %.16e %.16e\n";
    FILE *p;
    if((p = Fopen(filename.c_str(),"w")) == NULL)
      ERR.FileA("fMatrix","write",filename.c_str());
    for(int r=0;r<rows;r++)
      for(int c=0;c<cols;c++)
	Fprintf(p,fmt,r,c, (*this)(r,c).real(), (*this)(r,c).imag());
    Fclose(p);
  }

};

//Rearrange an Lt*Lt matrix from ordering  tsnk, tsrc  to   tsrc,  tsep=tsnk-tsrc
template<typename mf_Complex>
void rearrangeTsrcTsep(fMatrix<mf_Complex> &m){
  int Lt = GJP.Tnodes()*GJP.TnodeSites();
  if(m.nRows()!=Lt || m.nCols()!=Lt) ERR.General("","rearrangeTsrcTsep(fMatrix<mf_Complex> &)","Expect an Lt*Lt matrix\n");

  fMatrix<mf_Complex> tmp(m);
  for(int tsnk=0;tsnk<Lt;tsnk++){
    for(int tsrc=0;tsrc<Lt;tsrc++){
      int tsep = (tsnk-tsrc+Lt) % Lt;
      m(tsrc,tsep) = tmp(tsnk,tsrc);
    }
  }
}



//A vector of complex numbers and some useful associated methods
template<typename mf_Complex>
class fVector{
  mf_Complex* tt;
  int fsize;
  
  void free_mem(){
    if(tt!=NULL) sfree("fVector","~fVector","free",tt);
  }

  void alloc_mem(const int _elems, mf_Complex const* cp = NULL){
    if(_elems != fsize){
      free_mem();
      fsize = _elems;
      tt = (mf_Complex*)smalloc("fVector", "fVector", "alloc" , sizeof(mf_Complex) * fsize);
    }
    if(cp == NULL) zero();
    else for(int i=0;i<fsize;i++) tt[i] = cp[i];
  }

public:
  fVector(): fsize(0),tt(NULL){ }

  fVector(const int _elems): fsize(0),tt(NULL){ 
    alloc_mem(_elems);
  }
  fVector(const fVector<mf_Complex> &r): fsize(0),tt(NULL){
    alloc_mem(r.fsize,r.tt);
  }
  
  fVector(fVector<mf_Complex> &&r): fsize(r.fsize), tt(r.tt){ r.tt=NULL; }

  inline fVector & operator=(const fVector<mf_Complex> &r){
    alloc_mem(r.fsize, r.tt); return *this;
  }
  inline fVector & operator=(fVector<mf_Complex> &&r){
    fsize = r.fsize; tt = r.tt; r.tt = NULL; return *this;
  }

  inline mf_Complex *ptr(){ return tt;}

  inline void resize(const int _elems){ alloc_mem(_elems); }

  inline void zero(){ for(int i=0;i<fsize;i++) tt[i] = mf_Complex(0,0); }

  //Complex multiply and divide by constant
  inline fVector & operator*=(const mf_Complex &r){ for(int i=0;i<fsize;i++) tt[i] *= r;  return *this; }
  inline fVector & operator/=(const mf_Complex &r){ for(int i=0;i<fsize;i++) tt[i] /= r;  return *this; }

  //Floating point multiply and divide by constant
  inline fVector & operator*=(const typename mf_Complex::value_type &r){ for(int i=0;i<fsize*2;i++) ((typename mf_Complex::value_type*)tt)[i] *= r;  return *this; }
  inline fVector & operator/=(const typename mf_Complex::value_type &r){ for(int i=0;i<fsize*2;i++) ((typename mf_Complex::value_type*)tt)[i] /= r;  return *this; }

  //Add and subtract fVectors
  inline fVector & operator+=(const fVector &r){ for(int i=0;i<fsize;i++) tt[i] += r.tt[i]; return *this; }
  inline fVector & operator-=(const fVector &r){ for(int i=0;i<fsize;i++) tt[i] -= r.tt[i]; return *this;  }

  inline fVector operator+(const fVector &r) const{ fVector out(*this); out += r; return out; }
  inline fVector operator-(const fVector &r) const{ fVector out(*this); out -= r; return out; }

  //Element access
  inline const mf_Complex & operator()(const int i) const{ return tt[i]; }
  inline mf_Complex & operator()(const int i){ return tt[i]; }

  inline int size() const{ return fsize; }

  //Global sum of all elements
  void nodeSum(){
    globalSum( (typename mf_Complex::value_type*)tt,2*fsize);
  }

  ~fVector(){
    free_mem();
  }
  
  //hexfloat option: For reproducibility testing, write the output in hexfloat format rather than truncating the precision
  void write(const std::string &filename, const bool hexfloat = false) const{
    const char* fmt = hexfloat ? "%d %a %a\n" : "%d %.16e %.16e\n";
    FILE *p;
    if((p = Fopen(filename.c_str(),"w")) == NULL)
      ERR.FileA("fVector","write",filename.c_str());
    for(int i=0;i<fsize;i++)
      Fprintf(p,fmt,i, tt[i].real(), tt[i].imag());
    Fclose(p);
  }


};

//Average over the rows of the matrix
template<typename mf_Complex>
fVector<mf_Complex> rowAverage(const fMatrix<mf_Complex> &m){
  fVector<mf_Complex> out(m.nCols());
  double nr= m.nRows();
  for(int j=0;j<m.nCols();j++)
    out(j) = m(0,j)/nr;

  for(int i=1;i<m.nRows();i++)
    for(int j=0;j<m.nCols();j++)
      out(j) = out(j) + m(i,j)/nr;

  return out;
}

//Array of complex with optional threading and a policy class controlling allocation to allow this to contain, for example, data types that require aligned memory (SIMD data)
template<typename mf_Complex, typename AllocPolicy = StandardAllocPolicy>
class basicComplexArray: public AllocPolicy{
protected:
  int thread_size; //size of each thread unit
  int nthread;
  int size; //total size
  mf_Complex *con;
public:
  basicComplexArray(): size(0), con(NULL){}
  basicComplexArray(const int &_thread_size, const int &_nthread = 1): size(0), con(NULL){
    resize(_thread_size,_nthread);
  }
  void free_mem(){
    if(con != NULL){ AllocPolicy::_free(con); con = NULL; }
  }
  void resize(const int &_thread_size, const int &_nthread = 1){
    free_mem();
    thread_size = _thread_size; nthread = _nthread;
    size = _thread_size * _nthread;
    this->_alloc((void**)&con, size*sizeof(mf_Complex));
    memset((void*)con, 0, size * sizeof(mf_Complex));
  }
  ~basicComplexArray(){
    free_mem();
  }
  inline const mf_Complex & operator[](const int i) const{ return con[i]; }
  inline mf_Complex & operator[](const int i){ return con[i]; }

  inline mf_Complex & operator()(const int i, const int thread){ return con[i + thread * thread_size]; }
  inline mf_Complex & operator()(const int i, const int thread) const{ return con[i + thread * thread_size]; }

  int nElementsTotal() const{
    return size;
  }
  int nElementsPerThread() const{
    return thread_size;
  }
  int nThreads() const{
    return nthread;
  }
    
  //Sum (reduce) over all threads
  void threadSum(){
    if(nthread == 1) return;
    basicComplexArray<mf_Complex,AllocPolicy> tmp(thread_size,1);
    
#pragma omp parallel for
    for(int i=0;i<thread_size;i++){
      for(int t=0;t<nthread;t++)
	tmp.con[i] += con[i + t*thread_size];
    }
    AllocPolicy::_free(con);
    con = tmp.con;
    nthread = 1;
    size = tmp.size;

    tmp.con = NULL;
  }
  void nodeSum(){
    globalSum(this->con,size);
  }
};



//Same as the above but each thread has its own independent memory allocation. This relieves the requirement for large contiguous memory allocations
template<typename mf_Complex, typename AllocPolicy = StandardAllocPolicy>
class basicComplexArraySplitAlloc: public AllocPolicy{
protected:
  int thread_size; //size of each thread unit
  std::vector<mf_Complex*> con;
public:
  basicComplexArraySplitAlloc(): con(0){}
  basicComplexArraySplitAlloc(const int _thread_size, const int _nthread = 1): con(0){
    resize(_thread_size,_nthread);
  }
  void free_mem(){
    for(int t=0;t<con.size();t++)
      if(con[t] != NULL){ AllocPolicy::_free(con[t]); con[t] = NULL; }
  }
  void resize(const int &_thread_size, const int &_nthread = 1){
    free_mem();
    thread_size = _thread_size;
    con.resize(_nthread);
    for(int t=0;t<con.size();t++){
      this->_alloc((void**)&con[t], _thread_size*sizeof(mf_Complex));
      memset((void*)con[t], 0, _thread_size * sizeof(mf_Complex));
    }
  }
  ~basicComplexArraySplitAlloc(){
    free_mem();
  }
  inline const mf_Complex & operator[](const int i) const{ return con[0][i]; }
  inline mf_Complex & operator[](const int i){ return con[0][i]; }

  inline mf_Complex & operator()(const int i, const int thread){ return con[thread][i]; }
  inline mf_Complex & operator()(const int i, const int thread) const{ return con[thread][i]; }

  int nElementsTotal() const{
    return thread_size*con.size();
  }
  int nElementsPerThread() const{
    return thread_size;
  }
  int nThreads() const{
    return con.size();
  }
    
  //Sum (reduce) over all threads
  void threadSum(){
    if(con.size() == 0 || con.size() == 1) return;

#pragma omp parallel for
    for(int i=0;i<thread_size;i++){
      for(int t=1;t<con.size();t++)
	con[0][i] += con[t][i];
    }
    
    for(int t=1;t<con.size();t++) AllocPolicy::_free(con[t]);
    con.resize(1);
  }
  void nodeSum(){
#if defined(GRID_CUDA) || defined(GRID_HIP)
    //FIXME: Need to check if AMD GPU works for UVM	  
    //Perlmutter MPI calls on UVM are currently broken
    //To workaround we copy to a temp buffer
    size_t bsize = thread_size * sizeof(mf_Complex);
    mf_Complex* p = (mf_Complex*)malloc(bsize);
    for(int t=0;t<con.size();t++){
      memcpy(p, this->con[t], bsize);
      globalSum(p,thread_size);
      memcpy(this->con[t], p, bsize);
    }
    free(p);
#else
    for(int t=0;t<con.size();t++){
      globalSum(this->con[t],thread_size);
    }
#endif
  }
};



CPS_END_NAMESPACE

#endif
