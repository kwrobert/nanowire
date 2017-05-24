#!/bin/bash
currentHost=$(hostname)

if [ "$currentHost" == 'kyle-ThinkStation-P900' ]; then 
    hosts=('planck' 'dell' 'eastcloud')
elif [ "$currentHost" == 'dell' ]; then
    hosts=('planck' 'lenovo' 'eastcloud')
elif [ "$currentHost" == 'planck' ]; then
    hosts=('dell' 'lenovo' 'eastcloud')
else
    echo "Not on a core host"
    exit;
fi

for host in ${hosts[@]}; do
    ssh $host 'cd $HOME/software/nanowire && git pull'
done
