  
#PBS -l walltime=08:00:00
#PBS -o ${PBS_JOBID}-o.txt
#PBS -e ${PBS_JOBID}-e.txt
source activate learning-3d-output-saver
cd $PBS_O_WORKDIR
echo $( pwd )
i=$1

SPORT=$[3000 + i]
MPORT=$[3300 + i]
echo "Server Port ${SPORT}"
echo "Monitor Port ${MPORT}"
cd ..
echo $( pwd )
~/start_rcssserver3d.sh ${SPORT} ${MPORT} & 
sleep 2;

~/start_soccer3d_agent.sh ${SPORT} ${MPORT}  &

sleep 3;

cd ~/ddpg-humanoid
python -m baselines.ppo1.run_soccer --num-timesteps=12000000
sleep 15;
