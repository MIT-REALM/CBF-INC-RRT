##!/bin/bash

# Fixed params in this trial
IDX=(0 250 500 750 1000)
SEED=(123 231 312)

#################### below ########################
for i in 0 1 2; do
  for j in 0 1 2 3; do
    python "../evaluation/eval_rrt_all.py" --robot_name magician --level hard --method_name rrt_dd \
            --start_idx ${IDX[$j]} --end_idx ${IDX[1+$j]} --seed ${SEED[$i]} --max_node 200 &
  done
done
wait

#for i in 0 1 2; do
#  for j in 0 1 2 3; do
#    python "../evaluation/eval_rrt_all.py" --robot_name yumi --level easy --method_name rrt_vanilla \
#            --start_idx ${IDX[$j]} --end_idx ${IDX[1+$j]} --RRT_STEP 120 --goal_biasing 0.3 --seed ${SEED[$i]} --max_node 300 &
#  done
#done
#wait
