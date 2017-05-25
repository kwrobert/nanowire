#!/bin/bash
currentHost=$(hostname)
branch=$1
if [ -z "$branch" ]; then
    branch='develop'
fi

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
    echo
    echo "Updating $branch on host $host"
    echo
    ssh $host 'cd $HOME/software/nanowire && git checkout '$branch' && git pull'
done
