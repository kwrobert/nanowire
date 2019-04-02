#!/bin/zsh

file="$1"
data=$(ptdump $1)
test=$(echo $data | grep '615')
echo "############################"
echo "Sampling number"
echo $test
if [ -z "$test" ]; then
    echo $file
    echo $file >> wrong_sampling.txt
fi
echo "############################"
echo "Z Coords"
test=$(echo $data | grep 'zcoords')
echo $test
if [ -z "$test" ]; then
    echo $file
    echo $file >> wrong_sampling.txt
fi
echo "############################"
echo "X Coords"
test=$(echo $data | grep 'xcoords')
echo $test
if [ -z "$test" ]; then
    echo $file
    echo $file >> wrong_sampling.txt
fi
echo "############################"
echo "Power absorbed"
test=$(echo $data | grep 'power_absorbed')
echo $test
if [ -z "$test" ]; then
    echo $file
    echo $file >> to_postprocess.txt
fi
