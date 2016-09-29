#!/bin/bash

# Replaces the old directory in sim_conf.ini with the new one in the instance that the directory has
# been renamed or moved 

regex="$1"
base=$(pwd)

for dir in $regex; do
    sim_line=$(grep -e "sim_dir" $dir/sim_conf.ini)
    old_sim_path=$(echo "$sim_line" | sed 's/\//\\\//g')
    old_sim_path=${old_sim_path#*= }
    path="$base/""$dir"
    new_path=$(echo "$path" | sed 's/\//\\\//g')
    echo 'sed -i '"s/$old_sim_path/$new_path/g"' $dir/sim_conf.ini'
    #sed -i "s/$old_dir/$dir/g" $dir/sim_conf.ini
    sed -i "s/$old_sim_path/$new_path/g" $dir/sim_conf.ini
    base_line=$(grep -e "basedir" $dir/sim_conf.ini)
    old_base_path=$(echo "$base_line" | sed 's/\//\\\//g')
    old_base_path=${old_base_path#*= }
    path="$base/"
    new_path=$(echo "$path" | sed 's/\//\\\//g')
    echo 'sed -i '"s/basedir = $old_base_path/basedir = $new_path/g"' $dir/sim_conf.ini'
    #sed -i "s/$old_dir/$dir/g" $dir/sim_conf.ini
    sed -i "s/$old_base_path/$new_path/g" $dir/sim_conf.ini
done
