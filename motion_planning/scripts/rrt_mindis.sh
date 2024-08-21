#!/bin/bash

# Fixed params in this trial
IDX=(0 250 500 750 1000)
SEED=(123 231 312)

#################### below ########################
export CUDA_VISIBLE_DEVICES=0,1,3
for i in 0 1 2; do
  for j in 0 1 2 3; do
    python "../evaluation/eval_rrt_all.py" --robot_name magician --level easy --method_name rrt_mindis \
            --start_idx ${IDX[$j]} --end_idx ${IDX[1+$j]} --seed ${SEED[$i]} --max_node 300 --devices $i &
  done
done
wait
