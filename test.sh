#!/bin/sh


for i in $(seq 0 23) 254 255
do
    echo "Number: $i"
    $1 -m $2 --no-mmap -cnv -rl $i  -p "hello" 2>&1 < input-short.txt | tee  datas/short-$i.txt
    $1 -m $2 --no-mmap -cnv -rl $i  -p "hello" 2>&1 < input-long.txt | tee datas/long-$i.txt
done
