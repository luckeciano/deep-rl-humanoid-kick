#!/bin/bash
nodes=$1
ppn=$2
workers=$[nodes*ppn]
#qsub each of the jobs

>nodes
for ((i=0; i < ${nodes}; i++))
do 
	qsub -F "${ppn} ${i}" start_worker.sh;
	sleep 35; 
done;
#wait for all nodes to allocate
allocated=0
time=0
while [ $allocated -ne $nodes ]
do
 allocated=`qstat -r | grep "R " | wc -l`
 echo "waiting for allocated nodes.. time:" $(($time/600))"min"
 echo "$allocated already allocated."
 sleep 10 
 let "time=time+10"
done
echo "all nodes allocated."


echo "total of ${workers} workers."

qsub -F "$workers"  start_rl_server.sh
