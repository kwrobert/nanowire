#/bin/zsh

while true; do
    echo "#######################################"
    echo "Syncing data"
    rsync -rvtP --exclude='sims.db' --exclude='sim.hdf5' --exclude='data.hdf5' westcloud:/data/sims/ $nano/simulations/
    echo "Sync complete! Sleeping!"
    sleep 20
done
