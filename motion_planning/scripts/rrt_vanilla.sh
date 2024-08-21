#!/bin/bash
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "../../")

#T_MAX_LIST=("400" "500" "750")
GOAL_BIAS_LIST=("0.1" "0.15" "0.2")
RRT_STEPSIZE_LIST=("0.7")
T_MAX_LIST=("500")
#GOAL_BIAS_LIST=("0.03")
#RRT_STEPSIZE_LIST=("0.7")

cd "${ROOT}"

for TMAX in "${T_MAX_LIST[@]}"; do
    for GOAL_BIAS in "${GOAL_BIAS_LIST[@]}"; do
        for RRT_STEPSIZE in "${RRT_STEPSIZE_LIST[@]}"; do
            python "motion_planning/evaluation/eval_rrt_vanilla.py" \
                --root_dir "${ROOT}"  --t_max $TMAX  --goal_biasing $GOAL_BIAS --RRT_STEPSIZE $RRT_STEPSIZE
        done
        wait
    done
done