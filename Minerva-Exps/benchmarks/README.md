# Benchmarks Reproducibility Suite

This framework is designed to facilitate the reproducibility of experiments across different models, datasets, and pipelines. It provides fine control over configuration files, deterministic execution plan generation, and efficient use of compute resources with execution precedence and overrides.

## Why is this used?

The process consists of two phases:
1. **Generating an execution plan** (`execution_planner.py`): This phase generates a detailed execution plan with precise configurations for models, datasets, and pipelines based on the user-defined settings and overrides.
2. **Executing distributedly/parallel** (`submit_it.py`): This phase runs the execution plan across a distributed system or in parallel, ensuring efficient use of computational resources.

### Key Features:
- **Fine control of configuration files**: Each configuration is hashed to ensure data integrity and reproducibility.
- **Deterministic execution plan generation**: As long as the configuration files remain unchanged, the generated execution plan will always be the same.
- **Topological execution**: The execution order follows the precedence rules defined in the experiment, forming a tree-like graph where some tasks depend on others.
- **Avoid re-running already completed executions**: The system tracks executions that have already been run.
- **Overrides**: Overrides allow for quick adjustments and extensions to the base configurations without modifying the base files themselves.



## How Do I Create My Experiment?

If you're new to this framework, don’t worry—setting up an experiment is straightforward. In just a few steps, you can define, configure, and run your experiments. Here’s how to get started:

### Step-by-Step Instructions

1. **Create a New Experiment Directory**:
   First, you need to create a folder where your experiment will live. This folder will store your experiment's configurations, logs, and settings.

   Open a terminal and run the following commands:
   
   ```bash
   mkdir my_experiment
   cd my_experiment
   mkdir configs logs
   mkdir configs/overrides
   ```

   - **`configs/`**: This folder will hold your experiment setup (CSV files and overrides).
   - **`logs/`**: This folder will store logs generated while your experiment runs.
   - **`overrides/`**: This is for any special settings that will override the default configurations (more on this later).

2. **Create the Experiment Definition (`experiments.csv`)**:
   In the `configs/` folder, create a file called `experiments.csv`. This file will list out the specific experiments you want to run—like training a model, fine-tuning it, or evaluating it.

   **What is an experiment?**
   An experiment is a combination of:
   - A model (what you’re trying to train)
   - A dataset (the data you’re using to train the model)
   - A pipeline (how the training process should happen)

   Here’s an example of what your `experiments.csv` might look like:

   ```csv
   execution/id,model/config,model/name,data/data_module,data/view,data/dataset,data/partition,data/name,data/override_id,pipeline/task,pipeline/name,backbone/load_from_id,ckpt/resume
   pretrain_tfc,train,tfc,multimodal_df,daghar_standardized_balanced,*,train,*,perc_100,har,train,,
   finetune_tfc,finetune,tfc,multimodal_df,daghar_standardized_balanced,*,train,*,*,har,train,pretrain_tfc,
   evaluate_tfc,finetune,tfc,multimodal_df,daghar_standardized_balanced,*,test,*,perc_100,har,evaluate,finetune_tfc,true
   ```

   - **`execution/id`**: This is the name of the experiment.
   - **`model/config`**: The model configuration (e.g., training or fine-tuning).
   - **`data/data_module`**: The type of dataset you’re using.
   - **`pipeline/task`**: What process you’re running (e.g., training or evaluation).
   - **Wildcards (`*`)**: These let you refer to multiple datasets or settings without having to list each one.

3. **Add Overrides (Optional)**:
   Sometimes, you may want to quickly change some settings for specific experiments, like using only a small part of the dataset or changing the batch size. That’s where **overrides** come in.

   **Overrides** are extra settings you can apply on top of your default configurations to quickly change behavior without rewriting everything.

   Create an override file for your data module in `configs/overrides/`:

   Example `data_modules.csv` (in `configs/overrides/`):
   ```csv
   override_id,data_module,view,dataset,partition,name,overrides
   perc_1,multimodal_df,daghar_standardized_balanced,*,train,*,"data_percentage=0.01 batch_size=64"
   perc_50,multimodal_df,daghar_standardized_balanced,*,train,*,"data_percentage=0.50 batch_size=64"
   perc_100,multimodal_df,daghar_standardized_balanced,*,*,*,"data_percentage=1.0 batch_size=64"
   ```

   In this example:
   - **`perc_1`**: Only uses 1% of the data for training.
   - **`perc_100`**: Uses 100% of the data for training.
   - **`batch_size=64`**: Sets the batch size for training to 64.

4. **Use Predefined Configurations**:
   The framework comes with a set of base configuration files that describe how data is loaded, how models are structured, and how pipelines run. These base configurations are located in the `base_configs/` folder that comes with the framework (you don’t need to create this yourself). The framework automatically uses these configurations for your experiments.

   **What are base configurations?**
   - **Data Modules**: Define how datasets are loaded and processed.
   - **Models**: Define what model architecture to use and its parameters.
   - **Pipelines**: Define how to train or evaluate models.

5. **Generate Your Execution Plan**:
   Once your `experiments.csv` and overrides (if any) are ready, you need to generate an **execution plan**. This plan tells the system exactly what commands to run for your experiment.

   Run the following command to generate the plan:

   ```bash
   python execution_planner.py \
       --executions_path ./experiments/example/configs/experiments.csv \
       --base_configs_path ./base_configs/ \
       --overrides_path ./experiments/example/configs/overrides/ \
       --db_file ./experiments/example/configs/generated_executions.csv \
       --log_dir ./experiments/example/logs/ \
       --seed 42 \
       --version_name final
   ```

   This command will generate a new file called `generated_executions.csv` inside `./experiments/example/configs`. This file contains all the individual tasks that need to be run based on your experiment definitions.



With these simple steps, you’ve now set up your experiment! The framework will take care of the rest by running your experiment according to the configurations you provided. If you're ready to learn more about how the base configurations work and how they fit into your experiment, continue to the next section on **Creating an Experiment**.


### Example 

Check out [experiments/example directory](./experiments/example) to see a complete setup with base configurations, overrides, and an experiment definition. You can use this as a template to create your own experiments.

## Creating an Experiment

The core idea is to define an experiment using base configurations, optionally apply overrides, and then generate an execution plan. Let’s go step by step:

### Base Configurations

The `base_configs/` directory contains the core configuration files for data modules, models, and pipelines. Each configuration file is responsible for specifying the behavior of a specific part of the experiment. These files are referenced in the experiment setup, and unique identifiers (UIDs) are generated to ensure reproducibility and traceability. Below, we explore the structure, UID generation, and provide example YAML configurations for each type.

#### Directory Structure

```plaintext
base_configs/
├── data_modules
│   └── multimodal_df
│       └── daghar_standardized_balanced
│           ├── kuhar
│           │   ├── test
│           │   │   └── config_0.yaml
│           │   └── train
│           │       └── config_0.yaml
│           └── motionsense
│               ├── test
│               │   └── config_0.yaml
│               └── train
│                   └── config_0.yaml
├── models
│   ├── finetune
│   │   ├── tfc.yaml
│   │   ├── tfc_ts2vec.yaml
│   └── train
│       ├── tfc.yaml
│       ├── tfc_ts2vec.yaml
└── pipelines
    └── har
        ├── evaluate.yaml
        └── train.yaml
```

Each configuration is referenced in the `experiments.csv` and dynamically combined during the execution. Let’s break down the structure by data modules, models, and pipelines, and show how UIDs are generated and configurations are used.



#### 1. Data Modules

Data modules specify how the datasets are structured, processed, and used in the experiment. Each data module is defined hierarchically, as shown below:

**Structure of a data module:**
```plaintext
data_modules/
└── <data_module>/
    └── <view>/
        └── <dataset>/
            └── <partition>/
                └── <name>.yaml
```

##### Nomenclature:
- `<data_module>`: The main data module name. Example: `multimodal_df`.
- `<view>`: The specific view of the data module. Example: `daghar_standardized_balanced`.
- `<dataset>`: The dataset being used. Example: `kuhar`, `motionsense`.
- `<partition>`: The data partition, such as `train` or `test`.
- `<name>`: The specific configuration file name. Example: `config_0.yaml`.

##### Example Path:
```plaintext
base_configs/data_modules/multimodal_df/daghar_standardized_balanced/kuhar/train/config_0.yaml
```

##### YAML Example:
```yaml
class_path: minerva.data.data_modules.har.MultiModalHARSeriesDataModule
init_args:
  data_path: "/workspaces/HIAAC-KR-Dev-Container/shared_data/daghar/standardized_view/KuHar/"
  feature_prefixes: ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
  label: "standard activity code"
  features_as_channels: True
  cast_to: "float32"
```
This YAML configuration defines the data module for the `KuHar` dataset. It specifies where the data is located (`data_path`), the features to be used (`feature_prefixes`), and how the data should be processed (`features_as_channels`, `cast_to`).

##### UID Generation:
The UID for this configuration is generated by combining the following:
- `<data_module>` (e.g., `multimodal_df`)
- `<view>` (e.g., `daghar_standardized_balanced`)
- `<dataset>` (e.g., `kuhar`)
- `<partition>` (e.g., `train`)
- `<name>` (e.g., `config_0.yaml`)

The file contents are also hashed to ensure uniqueness.

**Example UID**: `dfee44fdd17e`



#### 2. Models

Model configurations specify the architecture and hyperparameters used during training or fine-tuning. Each model is stored in a subdirectory according to the type of operation (e.g., training, fine-tuning).

**Structure of a model configuration:**
```plaintext
models/
└── <config>/
    └── <name>.yaml
```

##### Nomenclature:
- `<config>`: Defines the task or phase (e.g., `train` or `finetune`).
- `<name>`: The name of the model configuration. Example: `tfc`, `tfc_ts2vec`.

##### Example Path:
```plaintext
base_configs/models/train/tfc.yaml
```

##### YAML Example:
```yaml
class_path: minerva.models.ssl.tfc.TFC_Model
init_args:
  input_channels: 6
  batch_size: 64
  TS_length: 60
  num_classes: 6
  single_encoding_size: 128
  backbone:
    class_path: minerva.models.nets.tfc.TFC_Backbone
    init_args:
      input_channels: 6
      TS_length: 60
      single_encoding_size: 128
  pred_head: null
```
This YAML file defines the `TFC_Model` with specific parameters such as `input_channels`, `batch_size`, `TS_length`, and `num_classes`. The `backbone` section specifies the architecture of the underlying neural network.

##### UID Generation:
The UID for this configuration is generated by hashing:
- `<config>` (e.g., `train`)
- `<name>` (e.g., `tfc`)

Along with the file contents.

**Example UID**: `160c03b688f3`



#### 3. Pipelines

Pipelines define how models are trained or evaluated on the data. Each pipeline configuration specifies the trainer settings and training strategies, such as the number of epochs or devices to use.

**Structure of a pipeline configuration:**
```plaintext
pipelines/
└── <task>/
    └── <name>.yaml
```

##### Nomenclature:
- `<task>`: Defines the task or process (e.g., `har` for Human Activity Recognition).
- `<name>`: The specific operation for that task (e.g., `train` or `evaluate`).

##### Example Path:
```plaintext
base_configs/pipelines/har/train.yaml
```

##### YAML Example:
```yaml
model: null                 # (???) It will be overwritten dynamically (model/file_path, from CSV)
trainer:
  class_path: lightning.Trainer
  init_args:
    accelerator: gpu        # The accelerator to use.
    devices: 1              # Number of GPUs to use (when using GPU).
    strategy: auto          # Strategy to use for distributed training.
    max_epochs: 10          # Number of epochs to train.
    logger:
      class_path: lightning.pytorch.loggers.CSVLogger
      init_args:
        save_dir: null      # (???) Overwritten dynamically (log_dir, from schema)
        name: null          # (???) Overwritten dynamically (execution/uid, from CSV)
        version: null       # (???) Overwritten dynamically (version_name, from schema)
    callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss   # Metric to monitor.
        mode: min           # Optimize the metric to minimize.
        save_last: true     # Save the last checkpoint.
        save_top_k: 1       # Save the best checkpoint.
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val_loss   # Metric to monitor.
        patience: 30        # Number of epochs with no improvement before stopping.
        mode: min           # Optimize the metric to minimize.
    deterministic: warn
run:
  task: fit                 # The task to run.
  data: null                # (???) Overwritten dynamically (data/file_path)
  ckpt_path: null           # (???) Overwritten dynamically (if ck

pt/resume is True)
```
This YAML file defines the training pipeline using the `lightning.Trainer`. The parameters such as `max_epochs`, `accelerator`, `devices`, and `logger` are all specified here. Some fields (`model`, `data`, `ckpt_path`) will be dynamically overwritten based on the execution plan.

##### UID Generation:
The UID for the pipeline configuration is generated by hashing:
- `<task>` (e.g., `har`)
- `<name>` (e.g., `train`)

Along with the file contents.

**Example UID**: `2c2e39042a14`



### Examples of Overriding

Overrides allow you to dynamically change parameters in the base configurations without modifying the original YAML files. These overrides are applied through CSV files, which can adjust data modules, models, or pipelines.

#### Data Module Overrides (`data_modules.csv`)

You can override specific parameters in the data module, such as the percentage of data used or batch size.

**Example:**
```csv
override_id,data_module,view,dataset,partition,name,overrides
perc_1,multimodal_df,daghar_standardized_balanced,*,train,*,"data_percentage=0.01 batch_size=64"
```
- **override_id**: `perc_1` (used in the `experiments.csv` file).
- **overrides**: Specifies that only 1% of the data should be used (`data_percentage=0.01`) with a batch size of 64.

**Resulting Command Override:**
```bash
--data.data_percentage=0.01 --data.batch_size=64
```

#### Model Overrides (`models.csv`)

Model overrides allow you to adjust parameters such as the hidden size or the number of layers.

**Example:**
```csv
override_id,config,name,overrides
custom_layers,train,tfc,"model.hidden_size=512 model.num_layers=3"
```
- **override_id**: `custom_layers` (used in the `experiments.csv` file).
- **overrides**: Sets the hidden size to 512 and the number of layers to 3.

**Resulting Command Override:**
```bash
--model.hidden_size=512 --model.num_layers=3
```

#### Pipeline Overrides (`pipelines.csv`)

Pipeline overrides can adjust the number of epochs or the GPUs used during training.

**Example:**
```csv
override_id,task,name,overrides
train_fast,har,train,"trainer.max_epochs=50 trainer.gpus=2"
```
- **override_id**: `train_fast` (used in the `experiments.csv` file).
- **overrides**: Sets the training to run for 50 epochs and use 2 GPUs.

**Resulting Command Override:**
```bash
--trainer.max_epochs=50 --trainer.gpus=2
```




## Defining Experiments (`experiments.csv`)

The `experiments.csv` file is the core of the experiment definition. It defines how different configurations—such as data modules, models, and pipelines—come together to create a full execution plan. By specifying combinations of these configurations and leveraging wildcards, you can easily define complex experiments with minimal effort.

Each row in the `experiments.csv` corresponds to one or more executions, depending on the use of wildcards. Additionally, you can define dependencies between experiments, forming a tree-like structure where some tasks depend on others (e.g., a model fine-tunes on a pre-trained model).



### Example of `experiments.csv`

Here is an example `experiments.csv` file:

```csv
execution/id,model/config,model/name,data/data_module,data/view,data/dataset,data/partition,data/name,data/override_id,pipeline/task,pipeline/name,backbone/load_from_id,ckpt/resume
pretrain_tfc,train,tfc,multimodal_df,daghar_standardized_balanced,*,train,*,perc_100,har,train,,
finetune_tfc,finetune,tfc,multimodal_df,daghar_standardized_balanced,*,train,*,*,har,train,pretrain_tfc,
evaluate_tfc,finetune,tfc,multimodal_df,daghar_standardized_balanced,*,test,*,perc_100,har,evaluate,finetune_tfc,true
```

Let’s break down each field and its purpose:

##### Fields in `experiments.csv`:

- **execution/id**: A unique name for each execution. This will be used to refer to the execution in logs and for defining dependencies.
  - Example: `pretrain_tfc`, `finetune_tfc`, `evaluate_tfc`.

- **model/config**: The configuration of the model. This refers to the subfolder inside `base_configs/models/`.
  - Example: `train`, `finetune`.

- **model/name**: The specific model being used. This corresponds to the YAML file name inside the model configuration folder.
  - Example: `tfc`, `tfc_ts2vec`.

- **data/data_module**: Refers to the data module being used. This corresponds to the folder inside `base_configs/data_modules/`.
  - Example: `multimodal_df`.

- **data/view**: The view of the data module, such as a specific preprocessing or data transformation applied to the dataset.
  - Example: `daghar_standardized_balanced`.

- **data/dataset**: The specific dataset being used for the experiment. Wildcards (`*`) can be used to refer to multiple datasets.
  - Example: `kuhar`, `motionsense`.

- **data/partition**: The data partition, such as `train`, `test`, or `validation`.
  - Example: `train`, `test`.

- **data/name**: The name of the configuration file in the data module. Wildcards can be used to refer to multiple configurations.
  - Example: `config_0`, `config_1`.

- **data/override_id**: Refers to an override defined in the `overrides/data_modules.csv` file. This allows you to dynamically change parameters such as data percentage or batch size.
  - Example: `perc_100`, `perc_50`, `perc_1`.

- **pipeline/task**: Refers to the task or process being performed, such as `har` (Human Activity Recognition) or another task.
  - Example: `har`.

- **pipeline/name**: The specific pipeline configuration. This corresponds to the YAML file inside the `base_configs/pipelines/<task>/` folder.
  - Example: `train`, `evaluate`.

- **backbone/load_from_id**: Specifies the execution ID that this task depends on. This creates a tree of dependencies where one execution must complete before another starts.
  - Example: `pretrain_tfc`, `finetune_tfc`.

- **ckpt/resume**: A boolean (true/false) that indicates whether this task should resume from a previous checkpoint. This is useful when continuing training from a saved model.
  - Example: `true`, `false`.


### Example Breakdown

Let’s walk through the example `experiments.csv` file:

```csv
execution/id,model/config,model/name,data/data_module,data/view,data/dataset,data/partition,data/name,data/override_id,pipeline/task,pipeline/name,backbone/load_from_id,ckpt/resume
pretrain_tfc,train,tfc,multimodal_df,daghar_standardized_balanced,*,train,*,perc_100,har,train,,
finetune_tfc,finetune,tfc,multimodal_df,daghar_standardized_balanced,*,train,*,*,har,train,pretrain_tfc,
evaluate_tfc,finetune,tfc,multimodal_df,daghar_standardized_balanced,*,test,*,perc_100,har,evaluate,finetune_tfc,true
```

1. **Pretraining TFC model (`pretrain_tfc`)**:
   - **Model**: The model `tfc` is trained from scratch using the `train` configuration.
   - **Data**: The data module is `multimodal_df` with the view `daghar_standardized_balanced`. It uses all datasets (`*` wildcard), the `train` partition, and the `perc_100` override (100% of data).
   - **Pipeline**: The `train` pipeline for the `har` task is used.
   - **Dependency**: None. This is an independent task that doesn't rely on a previous execution.

2. **Fine-tuning TFC model (`finetune_tfc`)**:
   - **Model**: The `tfc` model is fine-tuned using the `finetune` configuration.
   - **Data**: The same data module and view (`multimodal_df`, `daghar_standardized_balanced`) is used for all datasets (`*`), but no specific override is applied (`*` for `data/override_id`).
   - **Pipeline**: The same `train` pipeline for the `har` task is used.
   - **Dependency**: This task depends on `pretrain_tfc`, meaning it will load the backbone from the `pretrain_tfc` checkpoint once it's completed.

3. **Evaluating TFC model (`evaluate_tfc`)**:
   - **Model**: The fine-tuned `tfc` model is evaluated using the `finetune` configuration.
   - **Data**: The evaluation is performed on the `test` partition with the `perc_100` override (100% of data).
   - **Pipeline**: The `evaluate` pipeline for the `har` task is used.
  

 - **Dependency**: This task depends on `finetune_tfc` and will resume from its checkpoint (`ckpt/resume` is set to `true`).


### Wildcards in `experiments.csv`

Wildcards (`*`) allow you to define a single experiment that can expand into multiple executions by dynamically matching configurations across datasets, partitions, and other parameters.

- **`*` in `data/dataset`**: Refers to all datasets under the specified view. For example, in the above case, both the `kuhar` and `motionsense` datasets will be used.
- **`*` in `data/name`**: Refers to all configuration files in the specified partition. This allows you to apply experiments across multiple configurations without manually specifying each one.
- **`*` in `data/override_id`**: Expands to all possible overrides defined in `overrides/data_modules.csv`.

### How Experiments are Expanded

The framework processes wildcards and expands the `experiments.csv` into multiple executions. For instance, if the `data/dataset` uses a `*` wildcard, and two datasets (`kuhar` and `motionsense`) exist, the framework will create separate executions for each dataset:

| execution/id | model/config | model/name | data/data_module | data/view | data/dataset | data/partition | data/name | data/override_id | pipeline/task | pipeline/name | backbone/load_from_id | ckpt/resume |
|--------------|--------------|------------|------------------|-----------|--------------|----------------|-----------|------------------|---------------|---------------|----------------------|-------------|
| pretrain_tfc | train         | tfc        | multimodal_df     | daghar_standardized_balanced | kuhar | train          | config_0  | perc_100         | har           | train         |                      | false       |
| pretrain_tfc | train         | tfc        | multimodal_df     | daghar_standardized_balanced | motionsense | train          | config_0  | perc_100         | har           | train         |                      | false       |

This means a single line in `experiments.csv` can dynamically expand into multiple tasks that cover all datasets or configurations.


## Generating an Execution Plan

Once the `experiments.csv` is defined, the next step is to generate the execution plan using `execution_planner.py`. This script reads the base configurations, overrides, and experiment definitions, then expands the experiments into concrete execution commands. These commands can later be executed in parallel or distributed systems.

#### Step-by-Step: Generating the Configuration

1. **Prepare Your Experiment Directory**
   
   Ensure your experiment directory contains:
   
   - **Base configurations**: The `base_configs/` directory should have the YAML files for data modules, models, and pipelines.
   - **Overrides**: If you're using overrides, place the CSV files in the `overrides/` folder inside your experiment directory.
   - **Experiments CSV**: The `experiments.csv` should define the experiments, wildcards, and dependencies.

   Example experiment directory structure:
   ```plaintext
   my_experiment/
   ├── configs
   │   ├── experiments.csv
   │   └── overrides
   │       ├── data_modules.csv
   │       ├── models.csv
   │       └── pipelines.csv
   └── logs
   ```

2. **Run `execution_planner.py`**

   Use the following command to generate the execution plan:
   
   ```bash
   python execution_planner.py \
       --executions_path my_experiment/configs/experiments.csv \
       --base_configs_path base_configs/ \
       --overrides_path my_experiment/configs/overrides/ \
       --db_file my_experiment/configs/generated_executions.csv \
       --log_dir my_experiment/logs \
       --seed 42 \
       --version_name final
   ```

   - **`--executions_path`**: Path to the `experiments.csv` file.
   - **`--base_configs_path`**: Path to the `base_configs/` directory where all YAML configurations are stored.
   - **`--overrides_path`**: Path to the directory containing the override CSV files (optional).
   - **`--db_file`**: Path to save the generated executions (output CSV).
   - **`--log_dir`**: Directory where logs will be stored.
   - **`--seed`**: Random seed to ensure deterministic execution.
   - **`--version_name`**: Version name used for the logging and checkpointing system.

#### Example Output: `generated_executions.csv`

After running `execution_planner.py`, a new CSV file, `generated_executions.csv`, will be created. This file contains all the expanded executions based on your input `experiments.csv`, overrides, and base configurations.

An example of a generated execution plan might look like this:

```csv
execution/id,execution/uid,execution/model_uid,execution/pipeline_uid,execution/data_uid,execution/num_deps,execution/dependency_chain,execution/bash_command,execution/ckpt_path,execution/backbone_path,execution/status,execution/root_dir,backbone/load_from_uid,ckpt/resume
pretrain_tfc,98a6394300bf,160c03b688f3,2c2e39042a14,dfee44fdd17e,0,,python -m minerva.pipelines.lightning_pipeline --config '/path/to/pipelines/har/train.yaml' --seed 42 --trainer.logger.save_dir 'logs' --trainer.logger.name '98a6394300bf' --model '/path/to/models/train/tfc.yaml' --data '/path/to/data_modules/multimodal_df/daghar_standardized_balanced/kuhar/train/config_0.yaml',/logs/98a6394300bf/final/checkpoints/last.ckpt,,unknown,/logs/98a6394300bf/final,,False
```

Each row represents a specific execution, with fields describing the model, pipeline, data module, and their respective UIDs. It also contains the full `bash_command` that can be used to run the execution.

#### Key Fields in `generated_executions.csv`:
- **`execution/id`**: The ID of the execution, matching what was specified in `experiments.csv`.
- **`execution/uid`**: The unique identifier for the execution, generated by hashing the combination of model, data, and pipeline UIDs.
- **`execution/bash_command`**: The full command to execute the task. This command is ready to be executed in a terminal.
- **`execution/ckpt_path`**: Path to the checkpoint, where the model’s state will be saved.
- **`execution/status`**: The current status of the execution (e.g., `unknown`, `running`, `completed`).


## Running Distributed Executions

After generating the execution plan with `execution_planner.py`, the next step is to execute the tasks across multiple nodes in a distributed manner. This is done using **Ray**, a high-performance distributed execution framework. The script `submit_it.py` handles job submission and execution.

Follow these steps to run the distributed execution:


### 1. Start a Ray Head Node

Before you can run distributed jobs, you need to start a **Ray Head Node**. This step is only required once and should be done on one node, typically the master node. Other worker nodes will connect to this head node.

To start the Ray head node, run the following command on the master node:

```bash
ray start --head --port=6379
```

Make sure to note the IP address and port number (`192.168.1.201:6379` in this case) because the worker nodes will use this to connect.

> **Important**: You must use a **shared filesystem (NFS)** across all nodes to ensure proper synchronization of execution status, logs, and checkpoints.


### 2. Submit Executions with `submit_it.py`

Once the Ray head node is up, you can submit jobs to the cluster using `submit_it.py`. This script will distribute the tasks defined in `generated_executions.csv` to available nodes, ensuring each task is executed once and preventing re-running of completed executions.

Use the following command to submit jobs:

```bash
python submit_it.py experiments/example/configs/generated_executions.csv -w 4 --use-ray --ray-address 192.168.1.201:6379
```

Here’s what each argument means:
- `test_exps/configs/generated_executions.csv`: Path to the generated execution plan.
- `-w 4`: Number of workers per node (this is the number of parallel processes that will run on each worker node).
- `--use-ray`: Tells the script to use Ray for distributed execution.
- `--ray-address 192.168.1.201:6379`: Specifies the address of the Ray head node.



### 3. Track Progress via CSV

As each task is executed, the status of the executions will be updated in the `generated_executions.csv` file. The **status column** will reflect the current state of each task, such as:
- `unknown`: Execution not yet started.
- `running`: Execution is currently in progress.
- `completed`: Execution has finished successfully.
- `failed`: Execution has encountered an error.

You can check this file at any time to see which executions are pending, running, or completed.



### 4. Monitor Progress with Ray Dashboard

Ray provides a web-based dashboard that you can use to monitor the progress of distributed jobs. By default, the dashboard runs on `localhost:826

5` on the Ray head node.

- **To access the Ray dashboard**: Open a browser and go to `localhost:8265` on the Ray head node. You’ll be able to see job statuses, node resources, and logs for each execution.

If you’re working on a remote system (e.g., using VSCode or SSH), you can use port forwarding to access the dashboard:
- In VSCode, click the **port forwarding** button at the bottom left and forward port `8265`. This will allow you to open the dashboard on your local machine.



### Summary

By following these steps, you can easily distribute the execution of your experiments across multiple nodes. The Ray framework, combined with the `submit_it.py` script, ensures that tasks are efficiently distributed, tracked, and managed. Using the Ray dashboard provides real-time visibility into the progress of your jobs, making it easy to monitor and troubleshoot distributed executions.