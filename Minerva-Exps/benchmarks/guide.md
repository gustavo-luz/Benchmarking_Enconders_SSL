short guide

cd benchmarks

export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/resnet_tnc_tfc_cpc_exp/" 
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"    

python summarizer.py ${EXECUTIONS_PATH}

python summarizer.py ${EXECUTIONS_PATH} --output_csv saved_metrics_resnet_tnc_tfc_cpc_exp.csv

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  
    
    
other."execution/id" = "finetune_tfc_ts2vec" AND this."data/dataset" = other."data/dataset"  AND this."data/name" = other."data/name" AND other."model/override_id" = this."model/override_id"


tmux 

ray start --head

python submit_it.py ${MY_EXPERIMENT_DIR}/configs/generated_executions.csv -w 2 --use-ray --ray-address 192.168.1.216:6379


control b + d - detach

pra voltar:

tmux list-sessions
tmux attach -t 0

exit pra sair


