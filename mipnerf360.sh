#!/bin/bash

# training settings
record_running_output=true
iterations=40000
warmup=false
progressive=true
# data path, scenes and output path
DATA_BASE_PATH="data/mipnerf360"
OUTPUT_BASE_PATH="outputs/mipnerf360"
scenes=(
  "bicycle"
  "bonsai"
  "counter"
  "flowers"
  "garden"
  "kitchen"
  "room"
  "stump"
  "treehill"
)

# create command list
command_list=()
for scene in ${scenes[@]}; do
    # input and output paths
    scene_output_path=${OUTPUT_BASE_PATH}/${scene}
    data_path=${DATA_BASE_PATH}/${scene}
    mkdir -p ${scene_output_path}

    ### train command ###
    extra_args=" --iterations ${iterations} --eval --port 0 --data_device cuda -r -1 --gpu -1 --fork 2 --ratio 1 --appearance_dim 0 --visible_threshold -1 --base_layer 10 --dist2level round --update_ratio 0.2 --init_level -1 --dist_ratio 0.999 --levels -1 --extra_ratio 0.25 --extra_up 0.01"
    [ "$progressive" = true ] && extra_args+=" --progressive "
    [ "$warmup" = true ] && extra_args+=" --warmup "
    [ "$record_running_output" = true ] && extra_args+=" &>> ${scene_output_path}/running.txt "
    command="python train.py -s ${data_path} -m ${scene_output_path} ${extra_args}"
    echo execute command: $command
    command_list+=("$command")

    ### render command ###
    # extra_args=" --video --skip_train "
    # [ "$record_running_output" = true ] && extra_args+=" &>> ${scene_output_path}/running.txt"
    # command="python render.py -m ${scene_output_path} ${extra_args}"
    # echo execute command: $command
    # command_list+=("$command")
    
done

# parallel execution
n_jobs=4
delay_time=20
parallel --jobs ${n_jobs} --delay ${delay_time} ::: "${command_list[@]}"