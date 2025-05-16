#!/bin/bash

for ((n=0; n<=23; n++))
do
    echo $(( 0x$(cat log | grep "layer:$n" -m 1 | awk -F'addr:' '{print $2}' | awk '{print $1}') - 0x70abfeccd040)) | xargs printf "0x%x,\n"
done
