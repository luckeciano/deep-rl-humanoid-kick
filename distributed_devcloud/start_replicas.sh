#!/bin/bash
nodes=$1

#qsub each of the jobs

for ((i=0; i < ${nodes}; i++))
do 
	echo "Starting replica ${i}..."
	qsub -F "${i}" start_single_agent.sh;
	sleep 2; 
done;
echo "all nodes allocated."
