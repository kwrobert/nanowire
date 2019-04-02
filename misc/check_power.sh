#!/bin/zsh

for f in ../*/sim.hdf5; do
    str=$(ptdump $f | grep power_absorbed)
    if [ -z "$str" ]; then
        echo "Simulation ""$(dirname $f)"" missing power"
    fi
done
