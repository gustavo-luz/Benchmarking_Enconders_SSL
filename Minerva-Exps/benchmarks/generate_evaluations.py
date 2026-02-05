#!/usr/bin/env python3
"""
generate_evaluations.py

This script generates evaluation executions based on existing training executions
in the database. It reads the training executions and creates corresponding
evaluation executions with the same model/config but with test data partitions.
"""

import pandas as pd
import argparse
from pathlib import Path
import hashlib
from typing import List, Dict, Any
import sys


def hash_str(s: str, prefix: str = "id_") -> str:
    """Computes the MD5 hash of a string and returns the first 12 characters."""
    return prefix + hashlib.md5(s.encode()).hexdigest()[:12]


def generate_evaluation_executions(
    db_file: Path,
    output_file: Path,
    pipeline_task: str = "evaluate",
    pipeline_name: str = "evaluate",
    data_partition: str = "test",
    data_override_id: str = "multimodal_perc_100",
    ckpt_resume: bool = True,
) -> pd.DataFrame:
    """
    Generate evaluation executions from existing training executions.

    Parameters
    ----------
    db_file : Path
        Path to the existing executions database CSV file
    output_file : Path
        Path where to save the generated evaluation executions
    pipeline_task : str
        Pipeline task for evaluation (default: "evaluate")
    pipeline_name : str
        Pipeline name for evaluation (default: "evaluate")
    data_partition : str
        Data partition for evaluation (default: "test")
    data_override_id : str
        Data override ID for evaluation (default: "multimodal_perc_100")
    ckpt_resume : bool
        Whether to resume from checkpoint (default: True)

    Returns
    -------
    pd.DataFrame
        DataFrame containing the generated evaluation executions
    """

    # Read the existing database
    print(f"Reading existing database from: {db_file}")
    existing_df = pd.read_csv(db_file)

    # Filter only training executions (you might want to adjust this condition)
    train_executions = existing_df[
        (existing_df["pipeline/task"] == "train")
        | (existing_df["execution/id"].str.startswith("train_"))
    ].copy()

    print(f"Found {len(train_executions)} training executions")

    if len(train_executions) == 0:
        print("No training executions found!")
        return pd.DataFrame()

    # Generate evaluation executions
    evaluation_executions = []

    for _, train_row in train_executions.iterrows():
        # Create evaluation execution ID
        eval_id = train_row["execution/id"].replace("train_", "evaluate_")
        if not eval_id.startswith("evaluate_"):
            eval_id = f"evaluate_{train_row['execution/id']}"

        # Generate new UID for evaluation execution
        eval_uid = hash_str(
            f"{eval_id}/{train_row['model/uid']}/{data_partition}/{data_override_id}"
        )

        # Create evaluation row based on training row
        eval_row = {
            "execution/id": eval_id,
            "execution/uid": eval_uid,
            "execution/run_name": train_row.get("execution/run_name", ""),
            "backbone/load_from_uid": train_row[
                "execution/uid"
            ],  # Point to training execution
            "execution/status": "pending",
            "execution/num_deps": train_row.get("execution/num_deps", 0) + 1,
            "ckpt/resume": ckpt_resume,
            # Model info (same as training)
            "model/uid": train_row["model/uid"],
            "model/config": train_row["model/config"],
            "model/name": train_row["model/name"],
            "model/override_id": train_row["model/override_id"],
            "model/file_path": train_row["model/file_path"],
            "model/file_hash": train_row["model/file_hash"],
            # Data info (modified for evaluation)
            "data/uid": hash_str(
                f"{train_row['data/data_module']}/{train_row['data/view']}/{train_row['data/dataset']}/{data_partition}/{train_row['data/name']}/{data_override_id}"
            ),
            "data/data_module": train_row["data/data_module"],
            "data/view": train_row["data/view"],
            "data/dataset": train_row["data/dataset"],
            "data/partition": data_partition,  # Changed to test
            "data/name": train_row["data/name"],
            "data/override_id": data_override_id,  # Changed for evaluation
            "data/file_path": train_row["data/file_path"].replace(
                "/train/", "/test/"
            ),  # Adjust path if needed
            "data/file_hash": train_row["data/file_hash"],  # This might need adjustment
            # Pipeline info (changed for evaluation)
            "pipeline/uid": hash_str(f"{pipeline_task}/{pipeline_name}/no_override"),
            "pipeline/task": pipeline_task,
            "pipeline/name": pipeline_name,
            "pipeline/override_id": train_row["pipeline/override_id"],
            "pipeline/file_path": train_row["pipeline/file_path"].replace(
                "train", "evaluate"
            ),  # Adjust path
            "pipeline/file_hash": train_row[
                "pipeline/file_hash"
            ],  # This might need adjustment
            # Execution details (will be populated by execution planner)
            "execution/root_dir": "",
            "execution/bash_command": "",
        }

        evaluation_executions.append(eval_row)

    # Create DataFrame
    eval_df = pd.DataFrame(evaluation_executions)

    # Remove duplicates
    eval_df = eval_df.drop_duplicates(subset=["execution/uid"])

    # Save to file
    if output_file.suffix == ".csv":
        eval_df.to_csv(output_file, index=False)
    else:
        eval_df.to_csv(output_file.with_suffix(".csv"), index=False)

    print(f"Generated {len(eval_df)} evaluation executions")
    print(f"Saved to: {output_file}")

    return eval_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation executions from existing training executions"
    )
    parser.add_argument(
        "--db_file",
        type=str,
        required=True,
        help="Path to the existing executions database CSV file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path where to save the generated evaluation executions",
    )
    parser.add_argument(
        "--pipeline_task",
        type=str,
        default="evaluate",
        help="Pipeline task for evaluation (default: evaluate)",
    )
    parser.add_argument(
        "--pipeline_name",
        type=str,
        default="evaluate",
        help="Pipeline name for evaluation (default: evaluate)",
    )
    parser.add_argument(
        "--data_partition",
        type=str,
        default="test",
        help="Data partition for evaluation (default: test)",
    )
    parser.add_argument(
        "--data_override_id",
        type=str,
        default="multimodal_perc_100",
        help="Data override ID for evaluation (default: multimodal_perc_100)",
    )
    parser.add_argument(
        "--ckpt_resume",
        action="store_true",
        default=True,
        help="Whether to resume from checkpoint (default: True)",
    )

    args = parser.parse_args()

    db_file = Path(args.db_file)
    output_file = Path(args.output_file)

    if not db_file.exists():
        print(f"Error: Database file {db_file} does not exist!")
        sys.exit(1)

    # Generate evaluation executions
    eval_df = generate_evaluation_executions(
        db_file=db_file,
        output_file=output_file,
        pipeline_task=args.pipeline_task,
        pipeline_name=args.pipeline_name,
        data_partition=args.data_partition,
        data_override_id=args.data_override_id,
        ckpt_resume=args.ckpt_resume,
    )

    if len(eval_df) > 0:
        print("\nGenerated evaluation executions:")
        print(
            eval_df[
                ["execution/id", "execution/uid", "backbone/load_from_uid"]
            ].to_string(index=False)
        )
    else:
        print("No evaluation executions were generated!")


if __name__ == "__main__":
    main()
