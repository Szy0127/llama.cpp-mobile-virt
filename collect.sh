#!/bin/sh

echo "---------------------------------------"
echo "i:discard from layer i and async reload"
echo "254: discard all and sync reload"
echo "255: discard no data"
echo "---------------------------------------"

echo "short"
for i in $(seq 0 23) 254 255
do
average=$(grep "prefill cost:" datas/short-$i.txt | cut -d: -f2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
echo "layer $i: $average"
done

echo "long"

for i in $(seq 0 23) 254 255
do
average=$(grep "prefill cost:" datas/long-$i.txt | cut -d: -f2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
echo "layer $i: $average"
done
