#PBS -l walltime=08:00:00
#PBS -o ${PBS_JOBID}-o.txt
#PBS -e ${PBS_JOBID}-e.txt
cd $PBS_O_WORKDIR
echo $( pwd )
processes=$1
node=$2
echo "${processes} processes."
for ((i=0; i < ${processes}; i++))
do
	SPORT=$[3000 + i + node*processes]
	MPORT=$[3300 + i + node*processes]
	hostname >> nodes
	echo "Server Port ${SPORT}"
	echo "Monitor Port ${MPORT}"
	cd ..
	echo $( pwd )
	~/start_rcssserver3d.sh ${SPORT} ${MPORT} & 
	sleep 2;
	echo "Starting new agent..."
	if [ "$i" -eq  "$[processes - 1]" ]; then
	    echo "Starting last agent"
	    ~/start_soccer3d_agent.sh ${SPORT} ${MPORT}  
	else
	    ~/start_soccer3d_agent.sh ${SPORT} ${MPORT} &
	fi
	cd distributed_devcloud
	echo "finishing iteration ${i}"
done 
echo "finish."
