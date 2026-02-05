################################################################################
# INTERACTIVE MODE
################################################################################

import argparse
import traceback
import logging
import readline
from pathlib import Path
from typing import Dict

import pandas as pd
import pandasql

from minerva.utils.typing import PathLike
import yaml
from execution_planner import (
    DataModuleDatabase,
    PipelineDatabase,
    ModelDatabase,
    ExecutionDatabase,
    LightningPipelineExecutionGenerator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


options = {
    "print": "md",
}

class DummyDataframe:
    def get_df(self):
        return pd.DataFrame()

class PandasSQLDatabase:
    def __init__(
        self,
        tables: Dict[str, pd.DataFrame],
        raise_on_error: bool = True,
        verbose: bool = False,
    ) -> None:
        self._tables = tables
        self._raise_on_error = raise_on_error
        self._verbose = verbose

    def query(self, sql_cmd: str):
        try:
            result_df = pandasql.sqldf(sql_cmd, self._tables)
            if self._verbose:
                if result_df is None:
                    print(f"Query: {sql_cmd} returned None")
                else:
                    print(f"Query: {sql_cmd} returned {len(result_df)} rows")
                    print(result_df.head(n=3))
            return result_df
        except Exception as e:
            print(f"SQL Error: {e}")
            if self._raise_on_error:
                raise e
            else:
                return None


def print_dataframe(df):
    global options
    print_type = options["print"]

    if df is None:
        print("Empty dataframe")
        return

    if print_type in ("md", "markdown"):
        print(df.to_markdown(index=True))
    elif print_type in ("html"):
        print(df.to_html(index=False))
    elif print_type in ("csv"):
        print(df.to_csv(index=False))
    elif print_type in ("json"):
        print(df.to_json(orient="records", lines=True))
    elif print_type in ("jsonc"):
        print(df.to_json(orient="columns", lines=True))
    elif print_type in ("yaml"):
        print(yaml.dump(df.to_dict(orient="records"), default_flow_style=False))
    elif print_type in ("dict"):
        for row in df.to_dict(orient="records"):
            print(row)
    else:
        raise ValueError(f"Unknown print type: {options['print.type']}")

    print(f"-- Query returned {len(df)} rows --")


def clear_screen():
    """Clears the terminal screen."""
    print("\033[H\033[J")


def show_dataframe(df, indices=None):
    """Displays the dataframe based on the indices provided or displays all rows if indices is None."""
    if indices is None:
        print_dataframe(df)
    else:
        try:
            selected_indices = []
            for part in indices.split(" "):
                if "-" in part:
                    start, end = part.split("-")
                    selected_indices.extend(range(int(start), int(end) + 1))
                else:
                    selected_indices.append(int(part))
            result_df = df.iloc[selected_indices]
            print_dataframe(result_df)
        except Exception as e:
            print(f"Error: {e}")


def handle_dm_command(data_modules, cmd):
    """Handles the /dm command to show data modules."""
    if cmd == "/dm":
        show_dataframe(data_modules)
    else:
        indices = cmd.replace("/dm", "").strip()
        show_dataframe(data_modules, indices)


def handle_pl_command(pipelines, cmd):
    """Handles the /pl command to show pipelines."""
    if cmd == "/pl":
        show_dataframe(pipelines)
    else:
        indices = cmd.replace("/pl", "").strip()
        show_dataframe(pipelines, indices)


def handle_md_command(models, cmd):
    """Handles the /md command to show models."""
    if cmd == "/md":
        show_dataframe(models)
    else:
        indices = cmd.replace("/md", "").strip()
        show_dataframe(models, indices)


def handle_set_option(cmd):
    """Handles the /o command to set options."""
    global options
    key, value = cmd.replace("/o", "").strip().split("=")
    key = key.strip()
    value = value.strip()
    if key not in options:
        print(f"Error: Unknown option '{key}'. No value set.")
        return

    options[key] = value
    print(f"Set option: {key} = {value}")


def handle_print_option():
    """Handles the /p command to print options."""
    global options
    print("Current options:")
    for key, value in options.items():
        print(f"\t{key} = {value}")


def handle_exec_command(executions_df, cmd):
    """Handles the /exec command to show executions."""
    if cmd == "/ex":
        show_dataframe(executions_df)
    else:
        indices = cmd.replace("/ex", "").strip()
        show_dataframe(executions_df, indices)


def execute_sql_query(cmd, tables, verbose=False):
    """Executes the SQL query using pandasql and handles errors."""
    try:
        db = PandasSQLDatabase(
            tables=tables, verbose=verbose, raise_on_error=True
        )
        result_df = db.query(cmd)
        print_dataframe(result_df)
    except Exception as e:
        print(f"SQL Error: {e}")


def show_database_info(data_modules, pipelines, models, executions):
    """Shows the number of rows in each table."""

    def print_df_info(df, name):
        print("-" * 80)
        print(name)
        print(f"\tColumns: {', '.join(df.columns)}")
        print(f"\tThere are {len(df)} rows")

    print_df_info(data_modules, "data_modules")
    print_df_info(pipelines, "pipelines")
    print_df_info(models, "models")
    print_df_info(executions, "executions")
    print("-" * 80)


def print_help():
    print("You can manage the configurations interactively.")
    print(
        " 4 tables are available: data_modules (dm), pipelines (pl), models (md), executions (ex). You can query them using SQL syntax."
    )
    print(" Use /q to quit the interactive mode.")
    print(" Use /h to show help.")
    print(" Use /c to clear the screen.")
    print(" Use /s to show database information.")
    print(" Use /o to set variable value. For example /o print.type=md")
    print(" Use /p to print the current options.")
    print(
        " Use /dm to show data modules (shortcut for SELECT * FROM data_modules)."
    )
    print(" Use /pl to show pipelines (shortcut for SELECT * FROM pipelines).")
    print(" Use /md to show models (shortcut for SELECT * FROM models).")
    print(
        " Use /ex to show executions (shortcut for SELECT * FROM executions)."
    )
    print(
        "    Optionally, you can pass numbers (or ranges, separated by spaces) to show specific rows."
    )


def handle_command(cmd, data_modules, pipelines, models, executions):
    """Handles the entered command based on user input."""
    global options
    dm = data_modules  # Alias for data_modules
    pl = pipelines  # Alias for pipelines
    md = models  # Alias for models
    ex = executions  # Alias for executions

    if cmd == "/c" or cmd == "/clear":
        clear_screen()
    elif cmd == "/s" or cmd == "/show":
        show_database_info(data_modules, pipelines, models, executions)
    elif cmd == "/h" or cmd == "/help":
        print_help()
    elif cmd.startswith("/o"):
        handle_set_option(cmd)
    elif cmd == "/p":
        handle_print_option()
    elif cmd.startswith("/dm"):
        handle_dm_command(data_modules, cmd)
    elif cmd.startswith("/pl"):
        handle_pl_command(pipelines, cmd)
    elif cmd.startswith("/md"):
        handle_md_command(models, cmd)
    elif cmd.startswith("/ex"):
        handle_exec_command(executions, cmd)
    elif cmd.startswith("/"):
        print(f"Error: Unrecognized command '{cmd}'. Please try again.")
    else:
        execute_sql_query(cmd, locals())


def interactive_mode(dm_db, pipeline_db, model_db, executions_db):
    """Main interactive mode function where user inputs are processed."""
    data_modules, pipelines, models, executions = None, None, None, None

    def refresh_dbs():
        print("Refreshing the databases...")

        print("\tScanning data modules... ", end="")
        data_modules = dm_db.get_df()
        print(f"Found {len(data_modules)} data modules.")

        print("\tScanning pipelines... ", end="")
        pipelines = pipeline_db.get_df()
        print(f"Found {len(pipelines)} pipelines.")

        print("\tScanning models... ", end="")
        models = model_db.get_df()
        print(f"Found {len(models)} models.")

        print("\tScanning executions... ", end="")
        executions = executions_db.get_df()
        print(f"Found {len(executions)} executions.")
        print()

        return data_modules, pipelines, models, executions

    # Show the help message
    clear_screen()
    print("Welcome to the interactive mode!")
    print("-" * 80)
    print_help()
    print("-" * 80)
    handle_print_option()
    print("-" * 80)
    # Refresh the databases
    data_modules, pipelines, models, executions = refresh_dbs()
    print("-" * 80)

    # Start the interactive mode!
    while True:
        try:
            cmd = input("> ")

            if cmd == "/r":
                print("Refreshing the databases...")
                data_modules, pipelines, models, executions = refresh_dbs()
                continue

        except EOFError:  # This catches Ctrl+D
            print("/quit\n")
            break
        except KeyboardInterrupt:  # This catches Ctrl+C
            print("\n")
            continue

        cmd = cmd.strip()
        if not cmd:
            # print("Please enter a query.")
            continue

        readline.add_history(cmd)

        if cmd == "/q" or cmd == "/quit":
            print("Quitting interactive mode.")
            break

        try:
            handle_command(cmd, data_modules, pipelines, models, executions)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            print("Please try again.")
            continue


def main(
    executions_path: PathLike,
    base_configs_path: PathLike,
    overrides_path: PathLike,
    log_dir: PathLike,
    seed: int = 42,
    version_name: str = "final",
):
    base_configs_path = Path(base_configs_path)
    executions_path = Path(executions_path)

    # Create the databases paths
    dm_path = base_configs_path / "data_modules"
    pl_path = base_configs_path / "pipelines"
    model_path = base_configs_path / "models"

    # Check overrides path
    if overrides_path:
        overrides_path = Path(overrides_path)
        dm_overrides_path = overrides_path / "data_modules.csv"
        pl_overrides_path = overrides_path / "pipelines.csv"
        model_overrides_path = overrides_path / "models.csv"
    else:
        dm_overrides_path = None
        pl_overrides_path = None
        model_overrides_path = None

    # Create the databases
    dm_db = DataModuleDatabase(
        base_path=dm_path, overrides_path=dm_overrides_path
    )
    pl_db = PipelineDatabase(
        base_path=pl_path, overrides_path=pl_overrides_path
    )
    model_db = ModelDatabase(
        base_path=model_path, overrides_path=model_overrides_path
    )
    
    try:
        exp_db = ExecutionDatabase(
            data_module_database=dm_db,
            pipeline_database=pl_db,
            model_database=model_db,
            executions_path=executions_path,
        )
        
        # Generate the execution commands
        generator = ExecutionGenerator(
            exp_db,
            output_log_dir=log_dir,
            seed=seed,
            version_name=version_name,
        )
    except Exception as e:
        print(f"No executions found. Entering interactive mode without executions")
        generator = DummyDataframe()



    print("Entering interactive mode...")
    interactive_mode(dm_db, pl_db, model_db, generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule executions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--executions_path",
        type=str,
        required=True,
        help="Path to the executions CSV file.",
    )
    parser.add_argument(
        "--base_configs_path",
        type=str,
        required=True,
        help="Path to the base configurations.",
    )
    parser.add_argument(
        "--overrides_path",
        type=str,
        help="Path to the overrides CSV files.",
        required=False,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="experiments/logs",
        help="Directory to save logs.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for generation."
    )
    parser.add_argument(
        "--version_name",
        type=str,
        default="final",
        help="Version name for execution logs.",
    )

    args = parser.parse_args()

    main(
        executions_path=args.executions_path,
        base_configs_path=args.base_configs_path,
        overrides_path=args.overrides_path,
        log_dir=args.log_dir,
        seed=args.seed,
        version_name=args.version_name,
    )

# Example command
"""
-
"""
