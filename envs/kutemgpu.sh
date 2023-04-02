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
#
# -= Resources =-
#

#SBATCH --job-name=PPOexp216
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=rk02
#SBATCH --partition=kutem_gpu
#SBATCH --qos=kutem
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:1  
#SBATCH --time=1-0
#SBATCH --output=PPOexp216.out
#SBATCH --mail-type=END
#SBATCH --mail-user=tbal21@ku.edu.tr

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

## Load Python 3.8.6
echo "Activating Python 3.8.6..."
module load python/3.8.6
#module load anaconda/2022.05
#source activate stableBaselines

## Load GCC-9.1.0
echo "Activating GCC-9.1.0..."
module load gcc/9.1.0

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
echo "First Script..."
python3 trainTest.py --expNumber 216 --total_timesteps 163840 --n_steps 2048 --batch_size 1024 --n_envs 8 --testSamples 2 --testSampleOnTraining 2 --accelerationConstant 0.00000 --velocityConstant 0.1 --jointLimitLimitLowStartID "W0Low" --jointLimitHighStartID "W0High"
# Command 2 for matrix
echo "Running G++ compiler..."
# Put g++ compiler command below

# Command 3 for matrix
echo "Running compiled binary..."
# Put compiled binary command below

