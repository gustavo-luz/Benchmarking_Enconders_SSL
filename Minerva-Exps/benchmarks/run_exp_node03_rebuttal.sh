#!/bin/bash
# chmod +x run_exp_node02.sh
# ray start --head


# ### start exp ####rebuttal_times_tnc_lr4_seed43

export EXPERIMENT_NAME="rebuttal_times_tnc_lr4_seed43_harcnn"

export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


python generate_evaluations.py \
    --db_file ${EXECUTIONS_PATH} \
    --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  

# # # node 03
python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379

# # node 15
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # node 04
# # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # to add additional node eg node 17 help node 16
# # # # # ray start --address='192.168.1.216:6379'
# # # # # to help node03
# # # # # ray start --address='192.168.1.203:6379'

python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # ## end exp ####

# # ### start exp ####rebuttal_times_tnc_lr4_seed43

# export EXPERIMENT_NAME="rebuttal_times_tnc_lr4_seed43_rnn"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379

# # # node 15
# # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # # node 04
# # # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # # to add additional node eg node 17 help node 16
# # # # # # ray start --address='192.168.1.216:6379'
# # # # # # to help node03
# # # # # # ray start --address='192.168.1.203:6379'

# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tnc_lr4_seed43_resnetse5"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tnc_lr4_seed43_cnnpff"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tnc_lr3_seed43_ts2vec"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tnc_lr4_seed43_transformer"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

########################################## fim tnc inicio tfc ##########################################


# ### start exp ####rebuttal_times_tnc_lr4_seed43

export EXPERIMENT_NAME="rebuttal_times_tfc_lr4_seed43_harcnn"

export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


python generate_evaluations.py \
    --db_file ${EXECUTIONS_PATH} \
    --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  

# # # node 03
python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379

# # node 15
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # node 04
# # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # to add additional node eg node 17 help node 16
# # # # # ray start --address='192.168.1.216:6379'
# # # # # to help node03
# # # # # ray start --address='192.168.1.203:6379'

python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # ## end exp ####

# # ### start exp ####rebuttal_times_tnc_lr4_seed43

# export EXPERIMENT_NAME="rebuttal_times_tfc_lr4_seed43_rnn"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379

# # # node 15
# # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # # node 04
# # # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # # to add additional node eg node 17 help node 16
# # # # # # ray start --address='192.168.1.216:6379'
# # # # # # to help node03
# # # # # # ray start --address='192.168.1.203:6379'

# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tfc_lr4_seed43_resnetse5"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tfc_lr4_seed43_cnnpff"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tfc_lr3_seed43_ts2vec"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_tfc_lr4_seed43_transformer"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####



########################################## fim tfc inicio supervised ##########################################


# # # ### start exp ####rebuttal_times_tnc_lr4_seed43

# export EXPERIMENT_NAME="rebuttal_times_supervised_lr4_seed43_rnn"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379

# # # node 15
# # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # # node 04
# # # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # # to add additional node eg node 17 help node 16
# # # # # # ray start --address='192.168.1.216:6379'
# # # # # # to help node03
# # # # # # ray start --address='192.168.1.203:6379'

# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_supervised_lr4_seed43_resnetse5"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_supervised_lr4_seed43_cnnpff"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_supervised_lr3_seed43_ts2vec"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# # python generate_evaluations.py \
# #     --db_file ${EXECUTIONS_PATH} \
# #     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# # python execution_planner.py  \
# #     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
# #     --base_configs_path ${BASE_CONFIGS_PATH}   \
# #     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
# #     --db_file ${EXECUTIONS_PATH}  \
# #     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
# #     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
# #     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_supervised_lr4_seed43_transformer"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

# # ### start exp ####

export EXPERIMENT_NAME="rebuttal_times_supervised_lr4_seed43_harcnn"

export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


python generate_evaluations.py \
    --db_file ${EXECUTIONS_PATH} \
    --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  

# # # node 03
python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # ## end exp ####



########################################## fim supervised inicio diet ##########################################

# # ### start exp ####rebuttal_times_tnc_lr4_seed43

# export EXPERIMENT_NAME="rebuttal_times_diet_lr4_seed43_rnn"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379

# # # node 15
# # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # # node 04
# # # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # # to add additional node eg node 17 help node 16
# # # # # # ray start --address='192.168.1.216:6379'
# # # # # # to help node03
# # # # # # ray start --address='192.168.1.203:6379'

# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

# # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_diet_lr4_seed43_resnetse5"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_diet_lr4_seed43_cnnpff"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_diet_lr3_seed43_ts2vec"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

# # ### start exp ####

export EXPERIMENT_NAME="rebuttal_times_diet_lr4_seed43_harcnn"

export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


python generate_evaluations.py \
    --db_file ${EXECUTIONS_PATH} \
    --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  

# # # node 03
python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_diet_lr4_seed43_transformer"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####




# ########################################## fim diet inicio lfr ##########################################

# # # ### start exp ####rebuttal_times_tnc_lr4_seed43

# export EXPERIMENT_NAME="rebuttal_times_lfr_lr4_seed43_rnn"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379

# # # node 15
# # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.202:6379

# # # # # # node 04
# # # # # # python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.204:6379

# # # # # # to add additional node eg node 17 help node 16
# # # # # # ray start --address='192.168.1.216:6379'
# # # # # # to help node03
# # # # # # ray start --address='192.168.1.203:6379'

# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####

# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_lfr_lr4_seed43_resnetse5"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_lfr_lr4_seed43_cnnpff"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_lfr_lr3_seed43_ts2vec"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####



# # ### start exp ####

export EXPERIMENT_NAME="rebuttal_times_lfr_lr4_seed43_harcnn"

export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


python generate_evaluations.py \
    --db_file ${EXECUTIONS_PATH} \
    --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  

# # # node 03
python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # ## end exp ####


# # # ### start exp ####

# export EXPERIMENT_NAME="rebuttal_times_lfr_lr4_seed43_transformer"

# export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
# export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
# export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


# python generate_evaluations.py \
#     --db_file ${EXECUTIONS_PATH} \
#     --output_file ${MY_EXPERIMENT_DIR}/configs/generated_evaluations.csv

# python execution_planner.py  \
#     --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
#     --base_configs_path ${BASE_CONFIGS_PATH}   \
#     --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
#     --db_file ${EXECUTIONS_PATH}  \
#     --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
#     --log_dir ${MY_EXPERIMENT_DIR}/logs  \
#     --seed 42  

# # # # node 03
# python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.203:6379


# python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv --include_timing

# # # # # ## end exp ####




