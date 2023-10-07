#!/bin/bash 
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=PPOexp560
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1-0
#SBATCH --output=PPOexp560.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tbal21@ku.edu.tr

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

## Load Python 3.6.3
echo "Activating Python 3.6.3..."
#module load python/3.6.3
#module load anaconda/3.6
## Load GCC-7.2.1
echo "Activating GCC-7.2.1..."
module load gcc/7.2.1

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Example Job...!"
echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."
# Put Python script command below
export PYTHONPATH="/kuacc/users/tbal21/.conda/envs/stableBaselines/panda-gym/panda_gym/envs/utils"
python3 trainTest.py --expNumber 560 --total_timesteps 30000000 --n_steps 4096 --batch_size 2048 --n_envs 16 --testSamples 1 --evalFreqOnTraining 3000000 --lambdaErr 100.0 --velocityConstant 0.1 --orientationConstant 32 --maeThreshold 0.012 --configName "1.yaml"
# Command 2 for matrix
echo "Running G++ compiler..."
# Put g++ compiler command below

# Command 3 for matrix
echo "Running compiled binary..."
# Put compiled binary command below

