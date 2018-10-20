#PBS -l walltime=06:00:00
#PBS -o ppo_${PBS_JOBID}-o.txt
#PBS -e ppo_${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR
cd ../ddpg-humanoid
source activate learning-3d
echo $1
mpirun -n $1 -machinefile ../distributed_devcloud/nodes  python -m baselines.ppo1.run_soccer --num-timesteps=800000000 
