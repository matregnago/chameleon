#!/bin/bash
# sbatch.sh : submit slurm jobs and wait for completion before exiting
set -x

if [ $# -gt 0 ]
then
    JOB_NAME=$1
fi
JOB_NAME=${JOB_NAME:-chameleon}

# to get kernels execution on both cpus and gpus
export STARPU_SCHED=random

if [[ "${SLURM_CONSTRAINTS}" == "sirocco" ]]; then
    export LD_PRELOAD="/usr/lib64/libcuda.so"
fi

# the build directory comes from gitlab-ci artifacts, the absolute path may be different from the one used in the build stage, so we need to update the LD_LIBRARY_PATH to get the right chameleon libraries
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/build-${VERSION}/compute:$PWD/build-${VERSION}/coreblas/compute:$PWD/build-${VERSION}/gpucublas/compute:$PWD/build-${VERSION}/hqr/src:$PWD/build-${VERSION}/lapack_api:$PWD/build-${VERSION}/runtime/starpu"

# execution commands
sbatch --wait \
       --job-name="$JOB_NAME" \
       --output="$JOB_NAME.out" \
       --nodes=1 \
       --exclusive --ntasks-per-node=1 --threads-per-core=1 \
       --constraint="$SLURM_CONSTRAINTS" \
       --time=01:00:00 \
       $(dirname "$0")/test.sh
# get the error code from the last command: sbatch --wait ...
err=$?

cat $JOB_NAME.out

# exit with error code from the guix command
exit $err
