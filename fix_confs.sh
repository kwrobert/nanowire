#!/bin/bash

# Replaces the old directory in sim_conf.ini with the new one in the instance that the directory has
# been renamed or moved 

regex="$1"
base=$(pwd)

for dir in $regex; do
    line=$(grep -e "sim_dir" $dir/sim_conf.ini)
    old_path=$(echo "$line" | sed 's/\//\\\//g')
    old_path=${old_path#*= }
    path="$base/""$dir"
    new_path=$(echo "$path" | sed 's/\//\\\//g')
    echo 'sed -i '"s/$old_path/$new_path/g"' $dir/sim_conf.ini'
    #sed -i "s/$old_dir/$dir/g" $dir/sim_conf.ini
    sed -i "s/$old_path/$new_path/g" $dir/sim_conf.ini
done
