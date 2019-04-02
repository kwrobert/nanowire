#!/bin/zsh

total=0
for f in ../*/solution.bin; do
    size=$(du -sh $f | cut -f 1 | tr -d '[:alpha:]')
    total=$((total + size))
done
echo "Total size = $total MB"
