#!/usr/bin/env python3
import argparse
import json
import shutil
import os


def get_run_arg(run: list, arg_name: str):
    for i, arg in enumerate(run):
        if arg.startswith(arg_name):
            if "=" in arg:
                key, value = arg.split("=")
                if key == arg_name:
                    return value
            else:
                if arg == arg_name:
                    return run[i + 1]
    return ""


def remove_key(run: list, arg_name: str):
    new_run = []
    skip_next = False
    for i, arg in enumerate(run):
        if skip_next:
            skip_next = False
            continue

        if arg.startswith(arg_name):
            if "=" in arg:
                continue
            else:
                skip_next = True
        else:
            new_run.append(arg)

    return new_run


def main(
    template_file: str,
    config_file: str,
    container_name: str = "",
    shared_data_path: str = "/home/borin/shared_data/",
    shared_runs: str = "/home/borin/shared_runs/",
):
    with open(template_file, "r") as f:
        config = json.load(f)

    # Run Args
    run_args = config["runArgs"]

    # Container Name
    container_name = get_run_arg(run_args, "--name")
    # Default container name
    user_id = os.getlogin()
    container_name = container_name or f"hiaac-m4-{user_id}"
    container_name_input = input(
        f"Enter the name for you container (default: {container_name}): "
    )
    container_name = (
        container_name_input if container_name_input else container_name
    )
    if container_name:
        run_args = remove_key(run_args, "--name")
        run_args.append("--name")
        run_args.append(container_name)

    # Mounts
    run_args = remove_key(run_args, "-v")
    run_args = remove_key(run_args, "--volume")
    shared_data_path_input = input(
        f"Enter the path to the shared data folder (default: {shared_data_path}): "
    )

    shared_data_path = (
        shared_data_path_input if shared_data_path_input else shared_data_path
    )
    if shared_data_path:
        if not os.path.isdir(shared_data_path):
            raise ValueError(
                f"Invalid shared data path [{shared_data_path}]."
            )
        run_args.append("-v")
        run_args.append(f"{shared_data_path}:/workspaces/shared/data/")

    shared_runs_input = input(
        f"Enter the path to the shared runs folder (default: {shared_runs}): "
    )
    shared_runs = shared_runs_input if shared_runs_input else shared_runs
    if shared_runs:
        if not os.path.isdir(shared_runs):
            raise ValueError(f"Invalid shared runs path [{shared_runs}].")
        run_args.append("-v")
        run_args.append(f"{shared_runs}:/workspaces/shared/runs/")

    config["runArgs"] = run_args

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    print(
        f"Configuration file generated: {config_file} (from: {template_file})"
    )
    print("The configuration file is:")
    print("-" * 80)
    print(json.dumps(config, indent=4, sort_keys=True))
    print("-" * 80)
    print("Bye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dev container configuration")
    parser.add_argument(
        "--template",
        type=str,
        help="Configuration template file path",
        default=".devcontainer/devcontainer.template.json",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path",
        default=".devcontainer/devcontainer.json",
    )
    args = parser.parse_args()

    main(args.template, args.config)
