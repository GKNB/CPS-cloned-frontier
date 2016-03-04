#include<config.h>
CPS_START_NAMESPACE
/*!\file
  \brief  Definition of RNG classes.

  $Id: random.h,v 1.28.142.1 2012-11-15 18:17:08 ckelly Exp $
 */
//--------------------------------------------------------------------
//  CVS keywords
//
//  $Author: ckelly $
//  $Date: 2012-11-15 18:17:08 $
//  $Header: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/include/util/random.h,v 1.28.142.1 2012-11-15 18:17:08 ckelly Exp $
//  $Id: random.h,v 1.28.142.1 2012-11-15 18:17:08 ckelly Exp $
//  $Name: not supported by cvs2svn $
//  $Locker:  $
//  $RCSfile: random.h,v $
//  $Revision: 1.28.142.1 $
//  $Source: /home/chulwoo/CPS/repo/CVS/cps_only/cps_pp/include/util/random.h,v $
//  $State: Exp $
//
//--------------------------------------------------------------------


#ifndef INCLUDED_RANDOM_H
#define INCLUDED_RANDOM_H              //!< Prevent multiple inclusion

CPS_END_NAMESPACE
#include <string.h>
#include <util/data_types.h>
#include <util/enum.h>
#include <util/error.h>
#include <util/smalloc.h>
#include <util/verbose.h>
CPS_START_NAMESPACE
//---------------------------------------------------------------
//! A random number generator generating uniform random numbers in (0,1)
/*!
  This uses the Fibonacci RNG routine (ran3) from Numerical Recipes in C
*/
//---------------------------------------------------------------
class RandomGenerator {
  private:


    static int MBIG;
    static IFloat FAC;			// 1.0/MBIG
    static const int state_size = 55;
    int ma[state_size];	// The value 55(range ma[0...54])
    				// is special and should not be
				// modified.
    int inext;  
    int inextp;
    
  public:

    RandomGenerator() {    }
    virtual ~RandomGenerator() {    }

    //CK for testing add copy constructor and equivalence operator
    RandomGenerator(const RandomGenerator &in);
    RandomGenerator & operator=(const RandomGenerator &in);
    bool operator==(const RandomGenerator &in) const;

    //! Gets a random number 
    IFloat Rand();

    //! Seeds the RNG.
    void Reset(int seed);

    //! Stores the RNG state.
    void StoreSeeds(unsigned int *to) const;

    //! Loads the RNG state.
    void RestoreSeeds(const unsigned int *from);

    //! Size of the RNG state.
    int StateSize() const;

    int const* getState() const{ return ma; }
#if 1
    //! Number of Integers in RNG, that should be stored to record status
    virtual int RNGints() const { return state_size + 2; } // ma & inext & inextp

    //! to store this object
    void store(int *buf);

    //! to load from file
    void load(int *buf);
#endif
    
};






//---------------------------------------------------------------
//! A random number generator generating uniform random numbers.
/*!
  The range is defined in the constructor.
  
  This should be inherited by UGrandomGenerator class rather than used
  directly.
*/
//---------------------------------------------------------------
class UniformRandomGenerator: public virtual RandomGenerator
{
  private:
    static IFloat A;
    static IFloat B;

  public:

/*!
  The default range is (-0.5, 0.5).
  \param high_limit the upper bound of the distribution range
  \param low_limit the lower bound of the distribution range
  \note The user must ensure that the lower bound is not greater than the
  upper bound.
*/
    UniformRandomGenerator(IFloat high_limit = 0.5, IFloat low_limit = -0.5):
//	RandomGenerator(), A(low_limit), B(high_limit) {} //what's wrong with this? CK: THEY'RE STATIC MEMBERS!
	RandomGenerator() {}
	~UniformRandomGenerator() {}

//! Sets the interval over which the uniform distribution is defined.
/*!
  The default range is (-0.5, 0.5).
  \param high_limit the upper bound of the distribution range
  \param lower_limit the lower bound of the distribution range
  \post This sets the interval for all instances of this class.
  \note The user must ensure that the lower bound is not greater than the
  upper bound.
*/
    static void SetInterval(IFloat high_limit, IFloat low_limit){
	A = low_limit;
	B = high_limit;
    }
    static void GetInterval(IFloat &hi, IFloat &lo){
      lo = A; hi = B;
    }

    IFloat Rand();
    IFloat Rand(Float high, Float low);
};
 
 


//---------------------------------------------------------------
// Gaussian Random Generator. Generate Exp(-x^2/(2*sigma2))
// 
//! A random number generator generating gaussian random numbers.
/*!
  The mean of the distribution is 0; the variance is defined in the
  constructor. It is based on the Box-Muller algorithm (see "Numerical
  Recipes in C" pp.289).

  This should be inherited by UGrandomGenerator class rather than used
  directly.
*/ 
//---------------------------------------------------------------

class GaussianRandomGenerator : public virtual RandomGenerator
{
  private:
    static IFloat sigma2;
    int iset;			 // flag
    IFloat gset;			 // saved random number

  public:
/*!
  The default variance is 1.0.
  \param s2 the variance of the gaussian distribution.
*/
    GaussianRandomGenerator(IFloat s2 = 1.0):
	RandomGenerator(), iset(0) {SetSigma(s2);}    
    ~GaussianRandomGenerator() {}

    //CK: Added copy constructor and equivalene operator for testing purposes
    GaussianRandomGenerator(const GaussianRandomGenerator &in);
    GaussianRandomGenerator & operator=(const GaussianRandomGenerator &in);
    bool operator==(const GaussianRandomGenerator &in) const;

//! Sets the variance of the distribution.
/*!
  \param s2 the variance of the gaussian distribution.
*/
    static void SetSigma(IFloat s2) {
	sigma2 = s2;
    }

    IFloat Rand(int noexit=0);
    //! Number of Integers in RNG, that should be stored to record status
    int RNGints() const { return RandomGenerator::RNGints()+1; } // iset
    int RNGIFloats() const { return 1; } // gset

    //! to store this object
    void store(int *buf) {
//      VRB.Result("GaussianRandomGenerator","store()","iset=%d",iset);
      if (iset)
        ERR.General("GaussianRandomGenerator","store()","iset !=0, RNG state cannot be saved correctly");
      RandomGenerator::store(buf);
      buf[RandomGenerator::RNGints()] = iset;
    }

    //! to load from file
    void load(int *buf) {
      RandomGenerator::load(buf);
      iset = buf[RandomGenerator::RNGints()];
//      VRB.Result("GaussianRandomGenerator","load","iset=%d",iset);
      if (iset)
        ERR.General("GaussianRandomGenerator","load()","iset !=0, RNG state is not correct");
      RandomGenerator::store(buf);
    }
};
 


//---------------------------------------------------------------
//! The random number generator for a single 2^4 hypercube in the lattice.
/*!
  For each 2^4 hypercube there is a uniform RNG and a gaussian RNG.
  They should be accessed through the LatRanGen class rather than directly.
*/
//  LatRanGen possesses an array of these generators,
//  one for each 2^4 hypercube.
//---------------------------------------------------------------
class UGrandomGenerator:
public UniformRandomGenerator, public GaussianRandomGenerator
{
  public:
    UGrandomGenerator():
	UniformRandomGenerator(),
	GaussianRandomGenerator() {};

    //CK added copy constructor and equivalence operator
    UGrandomGenerator(const UGrandomGenerator&in);
    UGrandomGenerator & operator=(const UGrandomGenerator&in);
    bool operator==(const UGrandomGenerator&in) const;

    //! Get a gaussian random number
    IFloat Grand(int noexit=0) { return GaussianRandomGenerator::Rand(noexit); }

    //! Get a uniform random number.
    IFloat Urand() { return UniformRandomGenerator::Rand(); }
    IFloat Urand(Float high, Float low) 
	{ return UniformRandomGenerator::Rand(high,low); }

//! Sets the interval over which the uniform distribution is defined.
/*!
  The default range is (-0.5, 0.5).
  \param high the upper bound of the distribution range
  \param lower the lower bound of the distribution range
*/
    void SetInterval(IFloat high, IFloat low)
	{UniformRandomGenerator::SetInterval(high, low); }

//! Sets the variance of the distribution.
/*!
  The default variance is 1.0 (and the mean is zero).
  \param s2 the variance of the gaussian distribution.
*/
    void SetSigma(IFloat s2) { GaussianRandomGenerator::SetSigma(s2);  }

    //! Number of Integers in RNG, that should be stored to record status
    int RNGints() const { return GaussianRandomGenerator::RNGints(); } // iset

    //! Stores the RNG state.
    void StoreSeeds(unsigned int *to) const;

};

  

//---------------------------------------------------------------
//! The lattice random number generator.
/*!
  This class contains a uniform and a gaussian RNG for each 2^4 hypercube
  on the lattice.
  To ensure cross-platform reproducibility, these RNGs should be used, not the
  ones defined in the classes from which this inherits.
*/
//---------------------------------------------------------------
class LatRanGen
{
  private:

    static const int default_seed = 112319;
    
    int n_rgen;  // Gives the number of generators (and hypercubes)
    int rgen_pos;// ID of the generator being used
    int can[5];  // Needed for Canonical Assignment of Sites
    int hx[5];   // int (GJP.(X,Y,Z,T,S)nodeSites / 2)
    int is_initialized; // = 0 when LatRanGen is not initialized
                                // = 1 when LatRanGen is initialized
    UGrandomGenerator *ugran;
    int n_rgen_4d ;  // CJ: Gives the number of 4d generators (and hypercubes)
    int rgen_pos_4d;// CJ: ID of the 4D generator being used
    UGrandomGenerator *ugran_4d; // CJ: 4D RNG for gauge field

    char *cname;
    
  public:
    LatRanGen();
    ~LatRanGen();

    //CK added for testing
    LatRanGen(const LatRanGen &in);
    LatRanGen &operator=(const LatRanGen &in);
    bool operator==(const LatRanGen &in) const;

    void Initialize();  

    void Reinitialize(); //added by CK - turns off is_initialised flag, frees memory and runs initialize again
    //below added by CK as I need access to the random number gens themselves.
    UGrandomGenerator & UGrandGen(const int &idx){ return ugran[idx]; }
    UGrandomGenerator & UGrandGen4D(const int &idx){ return ugran_4d[idx]; }

    //! Get a uniform random number.
    IFloat Urand(FermionFieldDimension frm_dim=FOUR_D);
    IFloat Urand(Float high, Float low, FermionFieldDimension frm_dim=FOUR_D);

    //! Get a gaussian random number
    IFloat Grand(FermionFieldDimension frm_dim=FOUR_D);

    //! Get a uniform random number which is the same on all nodes.
    IFloat Lrand(); 

    //! Sets the variance of the distribution.
    void SetSigma(IFloat sigma);

    //! Sets the interval over which the uniform distribution is defined.
    void SetInterval(IFloat high, IFloat low);

    //! Specifies which hypercube RNG to use.
    void AssignGenerator(int x, int y, int z, int t,int s=0, const int &field_idx = 0);
    //! Specifies which hypercube RNG to use.
    void AssignGenerator(const int * coor, const int &field_idx = 0);
    //! Specifies which hypercube RNG to use.
    void AssignGenerator(int i, const int &field_idx = 0);

    //! Size of the RNG state (per hypercube).
    int StateSize() const;


    //! Assign the  state to a selected RNG.
    void SetState(const unsigned int *,
        FermionFieldDimension frm_dim =FIVE_D );

    //! Assign the  state of all RNGs.      
    void SetStates(unsigned int **, 
        FermionFieldDimension frm_dim =FIVE_D );

    //! Get the total number of states
    int NStates(FermionFieldDimension frm_dim =FIVE_D) const;
    
    //! Retrieve the state of a single RNG
    void GetState(unsigned int *, 
        FermionFieldDimension frm_dim = FIVE_D ) const;

    //! Retrieve the state of all RNGs
    void GetStates(unsigned int **,
        FermionFieldDimension frm_dim = FIVE_D ) const;
    
    //! Save the RNGs to a file (due to multi-type class members, not only supports read/write on same platform)
    bool Read(const char* filename, int concur_io_num = 0);
    bool Write(const char* filename, int concur_io_num = 0);

    int GetGeneratorIndex(const FermionFieldDimension &frm_dim = FIVE_D) const{
      if(frm_dim == FOUR_D) return rgen_pos_4d;
      else return rgen_pos;
    }
    
 private:
    bool UseParIO;
    bool io_good;
 public:
    inline void setParallel() { UseParIO = 1; }
    
    inline void setSerial() { 
#if 1
      UseParIO = 0; 
#else
      const char * fname = "setSerial";
      VRB.Flow(cname,fname,"On non-QCDOC platform, setSerial() has no effect!");
#endif
    }

    inline int parIO() const { return UseParIO; }

    inline bool good() const { return io_good; }

    void Shift();

 private:
    int do_log;
    char log_dir[200];
 public:
    void setLogDir(const char * LogDir) {
      do_log = 1;
      strcpy(log_dir,LogDir);
    }

};

/*! An instance of the LatRanGen class, named LRG, should be
  created at the highest scope (outside main). This external declaration
  allows control of and access to the random number generation.
*/
extern LatRanGen LRG;

class LRGState {
  public:

  char *cname;
  unsigned int ** rng4d;
  unsigned int ** rng5d;

  LRGState();
  ~LRGState();
  
  void GetStates();
  void SetStates();

};


#endif


CPS_END_NAMESPACE
