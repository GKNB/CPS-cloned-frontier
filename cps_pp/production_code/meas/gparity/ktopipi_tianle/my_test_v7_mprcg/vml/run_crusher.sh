#!/bin/bash
# Begin LSF Directives
#SBATCH -A lgt104_crusher
#SBATCH -t 0:30:00
#SBATCH -S 0
#SBATCH -J my_test_v7
#SBATCH -o my_test_v7.%J
#SBATCH -e my_test_v7.%J
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH --gpu-bind=map_gpu:0
#SBATCH -c 4
#SBATCH --threads-per-core=1

#SBATCH -x crusher016

#export BIND=""
export BIND="--cpu-bind=verbose,map_ldom:3"

#export MPICH_SHARED_MEM_COLL_OPT=0
#export MPICH_ALLREDUCE_NO_SMP=1
#export MPICH_REDUCE_NO_SMP=1
export MPICH_SMP_SINGLE_COPY_MODE=CMA
#export MPICH_SMP_SINGLE_COPY_MODE=XPMEM
#export MPICH_GPU_NO_ASYNC_MEMCPY
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=16384
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=4

EXE=/ccs/home/tianle/test_diff_module/rocm-5.3.0-mpich-8.1.17/cps-build/production_code/meas/gparity/ktopipi/NOARCH.x
#CMD="srun --gpus-per-node=8 ${BIND} ${EXE} . 0 1 -nthread 4 -old_gparity_cfg -qmp-geom 1 1 1 1 --shm 2048 --accelerator-threads 8 -vMv_inner_blocking 1 -do_split_job 0 /ccs/home/tianle/test_diff_module/rocm-5.3.0-mpich-8.1.17/cps-build/production_code/meas/gparity/ktopipi/my_test_v7_mprcg/vml/checkpoint --device-mem 16384 --shm-mpi 1 --comms-overlap --comms-concurrent"
#CMD="srun --gpus-per-node=8 ${BIND} ${EXE} . 0 1 -nthread 4 -old_gparity_cfg -qmp-geom 1 1 1 1 --shm 2048 --accelerator-threads 8 -vMv_inner_blocking 1 -do_split_job 1 /ccs/home/tianle/test_diff_module/rocm-5.3.0-mpich-8.1.17/cps-build/production_code/meas/gparity/ktopipi/my_test_v7_mprcg/vml/checkpoint --device-mem 16384 --shm-mpi 1 --comms-overlap --comms-concurrent"
CMD="srun --gpus-per-node=8 ${BIND} ${EXE} . 0 1 -nthread 4 -old_gparity_cfg -qmp-geom 1 1 1 1 --shm 2048 --accelerator-threads 8 -vMv_inner_blocking 1 -do_split_job 2 /ccs/home/tianle/test_diff_module/rocm-5.3.0-mpich-8.1.17/cps-build/production_code/meas/gparity/ktopipi/my_test_v7_mprcg/vml/checkpoint --device-mem 16384 --shm-mpi 1 --comms-overlap --comms-concurrent"
#CMD="srun --gpus-per-node=8 ${BIND} ${EXE} . 0 1 -nthread 4 -old_gparity_cfg -qmp-geom 1 1 1 1 --shm 2048 --accelerator-threads 8 -vMv_inner_blocking 1 -do_split_job 3 /ccs/home/tianle/test_diff_module/rocm-5.3.0-mpich-8.1.17/cps-build/production_code/meas/gparity/ktopipi/my_test_v7_mprcg/vml/checkpoint --device-mem 16384 --shm-mpi 1 --comms-overlap --comms-concurrent"
#CMD="srun --gpus-per-node=8 ${BIND} ${EXE} . 0 1 -nthread 4 -old_gparity_cfg -qmp-geom 1 1 1 1 --shm 2048 --accelerator-threads 8 -vMv_inner_blocking 1 -do_split_job 4 /ccs/home/tianle/test_diff_module/rocm-5.3.0-mpich-8.1.17/cps-build/production_code/meas/gparity/ktopipi/my_test_v7_mprcg/vml/checkpoint --device-mem 16384 --shm-mpi 1 --comms-overlap --comms-concurrent"

#CMD="srun --gpus-per-node=8 ${BIND} ${EXE} . 0 1 -nthread 4 -old_gparity_cfg -qmp-geom 1 1 1 1 --shm 2048 --accelerator-threads 8 -vMv_inner_blocking 1 --device-mem 5 --comms-overlap --comms-concurrent --shm-mpi 1"

echo "$CMD"
$CMD
