module reset

module load PrgEnv-gnu/8.3.3
module load rocm/5.3.0
module load cray-mpich/8.1.17

module load gmp/6.2.1

# The following does not exist anymore
#module load fftw/3.3.9
module load cray-fftw/3.3.10.3
echo "Should be careful, we might want to use cray-fftw, which support fftw-3.3.10. Need testing!"

# The following does not exist anymore
#module load gperftools/2.9.1

module load gsl/2.7.1
export LD_LIBRARY_PATH=/opt/gcc/mpfr/3.1.4/lib:$LD_LIBRARY_PATH

# Be careful which hdf5 module to use!
# The following one has a bug, similar to https://github.com/ALPSCore/ALPSCore/issues/348. Notice on lustre we don;t need to turn off locking
module load hdf5/1.12.1
#export HDF5_USE_FILE_LOCKING=FALSE

#module load cray-hdf5-parallel/1.12.2.1


echo "To use MPI with hipcc, include the following in linking:"
echo "-L\${MPICH_DIR}/lib -lmpi -I\${MPICH_DIR}/include"

module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1

echo "To use GPU-Aware MPI, depending on the compiler, check the following:"
echo ""
echo "If use cc/CC, check the following two env-var and add add the following flags when compiling"
echo "PE_MPICH_GTL_DIR_amd_gfx90a=-L\${CRAY_MPICH_ROOTDIR}/gtl/lib"
echo "PE_MPICH_GTL_LIBS_amd_gfx90a=-lmpi_gtl_hsa"
echo "-I\${ROCM_PATH}/include"
echo "-L\${ROCM_PATH}/lib -lamdhip64"
echo ""
echo "If use hipcc, be careful about the following three flags when compiling:"
echo "-I\${MPICH_DIR}/include"
echo "-L\${MPICH_DIR}/lib -lmpi -L\${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa"
echo "HIPFLAGS = --amdgpu-target=gfx90a"

