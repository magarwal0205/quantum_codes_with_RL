#!/bin/sh

echo "OK, let's do this!"
for i in 0 1
do
   for j in 0 1
   do
   	  echo "Running for error type = $i, code type = $j, 1 iteration"
   	  python stabilizerplots_new.py $i $j 1
   done
done

for i in 0 1
do
   for j in 0 1
   do
   	  echo "Running for error type = $i, code type = $j, 100 iterations"
   	  python stabilizerplots_new.py $i $j 100
   done
done

for i in 0 1
do
   for j in 0 1
   do
   	  echo "Running for error type = $i, code type = $j, 1000 iterations"
   	  python stabilizerplots_new.py $i $j 1000
   done
done