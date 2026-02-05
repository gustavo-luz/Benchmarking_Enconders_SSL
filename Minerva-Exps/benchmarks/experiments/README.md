
all experiments are at ...

explain here the structure
### experiments for the paper - right place

at each experiment there are the config files


---------------- 
### tips to run easier

export EXPERIMENT_NAME="supervised_exp_ts2vec_fixed"

export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/${EXPERIMENT_NAME}/" 
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    


python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  
    

python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.216:6379 


python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_${EXPERIMENT_NAME}.csv


ray start --head

ray start --address=192.168.1.216:6379



----------------

Structure:

tests_gustavo contains old experiments that are good for now to be kept
experiments with paper prefix are the ones for the journal of the best backbone
TODO- organize this
