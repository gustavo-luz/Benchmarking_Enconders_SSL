#!/bin/bash
# chmod +x run_exp_example.sh
# ray start --head --port=6380

# activate venv
# set -e

# source ../../minerva_ssl_env/bin/activate

# VENV_PY="../../minerva_ssl_env/bin/python3.11"


# ### start exp ####

export ROOT_DIR="$(pwd)"

export EXPERIMENT_NAME="tfc_resnetse5_run1"

export BASE_CONFIGS_PATH="${ROOT_DIR}/base_configs"
export MY_EXPERIMENT_DIR="${ROOT_DIR}/experiments/${EXPERIMENT_NAME}"
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  

# # # node 03 get local node
python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 1 --use-ray --ray-address 192.168.15.4:6380

# # node 15
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # node 04
# # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # to add additional node eg node 17 help node 16
# # # # # ray start --address='192.168.1.216:6379'
# # # # # to help node03
# # # # # ray start --address='192.168.1.203:6379'

python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing
