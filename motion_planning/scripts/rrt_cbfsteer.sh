#!/bin/bash
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "../../")

#T_MAX_LIST=("400" "500" "750" "1000")
#GOAL_BIAS_LIST=("0.05" "0.10" "0.20" "0.30" "0.40" "0.50")
#RRT_STEP_LIST=("10" "15" "20" "25")
T_MAX_LIST=("500")
GOAL_BIAS_LIST=("0.20")
RRT_STEP_LIST=("10" )

cd "${ROOT}"

for TMAX in "${T_MAX_LIST[@]}"; do
    for GOAL_BIAS in "${GOAL_BIAS_LIST[@]}"; do
        for RRT_STEP in "${RRT_STEP_LIST[@]}"; do
            python "motion_planning/evaluation/eval_rrt_cbfsteer.py" \
                --root_dir "${ROOT}"  --t_max $TMAX  --goal_biasing $GOAL_BIAS --RRT_STEP $RRT_STEP
        done
        wait
    done
done