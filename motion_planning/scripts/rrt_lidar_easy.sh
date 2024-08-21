#!/bin/bash

# Fixed params in this trial
IDX=(0 250 500 750 1000)
SEED=(123 231 312)

#################### below ########################
#export CUDA_VISIBLE_DEVICES=1,2,3
for j in 0 1 2; do
  for i in {0..3}; do
  python "../evaluation/eval_rrt_all.py" --robot_name panda --level hard --method_name rrt_docbf \
          --start_idx ${IDX[$i]} --end_idx ${IDX[1+$i]} --seed ${SEED[$j]} --max_node 500 --devices $i &
  done
done
wait
