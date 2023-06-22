#!/bin/bash

#SBATCH --job-name=myJob         # create a short name for your job
#SBATCH --output=slurm-%j.out    # output file, %j will be replaced by job ID
#SBATCH --nodes=1                # number of nodes
#SBATCH --partition=typhon       # specify the partition
#SBATCH --time=7-00:00:00           # time limit: 7 days
#SBATCH --mail-type=END,FAIL     # notifications for job end and fail
#SBATCH --mail-user=zz0962@princeton.edu  # your email

# load the module you need, replace with the actual module name
python $1 $2        # execute the python script`
#nohup python PE_Run_PhenomXPHM.py &
