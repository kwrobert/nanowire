#!/bin/bash

ptdump $1 &> /dev/null; ret="$?" 

if [ "$ret" != "0" ]; then
    echo $1
fi
