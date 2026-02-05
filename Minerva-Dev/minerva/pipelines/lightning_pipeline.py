from collections import defaultdict
from typing import Any, Dict, Literal, Optional

import lightning as L

import torch
import yaml
from torchmetrics import Metric

from minerva.pipelines.base import Pipeline
from minerva.data.data_module_tools import get_full_data_split, get_split_dataloader
from minerva.utils.typing import PathLike
from minerva.analysis.model_analysis import _ModelAnalysis
import json

from minerva.analysis.metrics.balanced_accuracy import BalancedAccuracy
import time
import csv
from pathlib import Path


def _append_timing_csv(csv_path: Path, row: dict):
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def predict_batch(classification_metrics, regression_metrics):
    def predict_batch_fn(self, batch, batch_idx, dataloader_idx):
        X, y = batch
        y_hat = self.forward(X)

        if classification_metrics is not None:
            y_hat = torch.argmax(y_hat, dim=1)

        return y_hat, y

    return predict_batch_fn


class SimpleLightningPipeline(Pipeline):
    """Simple pipeline to train, test, predict and evaluate models using Pytorch
    Lightning. This class is intended to be seamlessly integrated with
    jsonargparse CLI.
    """

    def __init__(
        self,
        model: L.LightningModule,
        trainer: L.Trainer,
        log_dir: Optional[PathLike] = None,
        save_run_status: bool = True,
        classification_metrics: Optional[Dict[str, Metric]] = None,
        regression_metrics: Optional[Dict[str, Metric]] = None,
        model_analysis: Optional[Dict[str, _ModelAnalysis]] = None,
        apply_metrics_per_sample: bool = False,
        seed: Optional[int] = None,
    ):
        """Train/test/predict/evaluate a Pytorch Lightning model.

        It provides 4 tasks: fit, test, predict and evaluate. The fit task
        trains the model, the test task evaluates the model on the test set, the
        predict task generates predictions for the predict set and the evaluate
        task evaluates the model on the predict set and returns the metrics.

        The evaluate task can calculate classification and regression metrics,
        which is passed as arguments. The metrics are calculated per sample if
        `apply_metrics_per_sample` is True (that generate a metric for each),
        otherwise the metrics are calculated for the whole dataset (single
        metric). The last option is the default.

        Parameters
        ----------
        model : L.LightningModule
            The LightningModule to be used.
        trainer : L.Trainer
            The Lightning Trainer to be used.
        log_dir : PathLike, optional
            The default logging directory where all related pipeline files
            should be saved. By default None (uses current working directory)
        save_run_status : bool, optional
            If True, save the status of each run in a YAML file. This file will
            be saved in the working directory with the name
            `run_{pipeline_id}.yaml`. By default True.
        classification_metrics : Dict[str, Metric], optional
            The classification metrics to be used in the evaluate task. This
            dictionary should have the metric name as key and the
            `torchmetrics.Metric`-like object as value. The metric should be
            able to receive two tensors (y_true, y_pred) and return a tensor
            with the metric value. If None, no classification metrics will be
            calculated. Different from regression, the torch.argmax will be
            applied to the predictions before calculating the metrics.
            By default None.
        regression_metrics : Dict[str, Metric], optional
            The regression metrics to be used in the evaluate task. This
            dictionary should have the metric name as key and the
            `torchmetrics.Metric`-like object as value. The metric should be
            able to receive two tensors (y_true, y_pred) and return a tensor
            with the metric value. If None, no regression metrics will be
            calculated. By default None.
        model_analysis: Dict[str, _ModelAnalysis], optional
            The model analysis to be performed after the model is trained. This
            dictionary should have the analysis name as key and the
            `_ModelAnalysis`-like object as value. The analysis should be able
            to receive the model and the data and return a result. If None, no
            model analysis will be performed. By default None.
        apply_metrics_per_sample : bool, optional
            Apply the metrics per sample. If True, the metrics will be
            calculated for each sample and the results will be a list of
            metrics. If False, the metrics will be calculated for the whole
            dataset and the results will be a single metric (single-element
            list). By default False
        seed : int, optional
            The seed to be used in the pipeline. By default None.
        """
        if log_dir is None and trainer.log_dir is not None:
            log_dir = trainer.log_dir

        super().__init__(
            log_dir=log_dir,
            ignore=[
                "model",
                "trainer",
                "classification_metrics",
                "regression_metrics",
            ],
            cache_result=True,
            save_run_status=save_run_status,
            seed=seed,
        )
        self._model = model
        self._trainer = trainer
        self._data = None
        self._model_analysis = model_analysis
        self._classification_metrics = classification_metrics
        self._regression_metrics = regression_metrics
        self._apply_metrics_per_sample = apply_metrics_per_sample

    # Public read-only properties
    @property
    def model(self) -> L.LightningModule:
        """The LightningModule used in the pipeline.

        Returns
        -------
        L.LightningModule
            The model used in the pipeline.
        """
        return self._model

    @property
    def trainer(self) -> L.Trainer:
        """The Lightning Trainer used in the pipeline.

        Returns
        -------
        L.Trainer
            The trainer used in the pipeline.
        """
        return self._trainer

    @property
    def data(self) -> Optional[L.LightningDataModule]:
        """The LightningDataModule used in the last run of the pipeline.

        Returns
        -------
        L.LightningDataModule
            The data used in the last run of the pipeline.
        """
        return self._data

    def _calculate_metrics(
        self, metrics: Dict[str, Metric], y_hat: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, Any]:
        """Calculate the metrics for the given predictions and targets.

        Parameters
        ----------
        metrics : Dict[str, Metric]
            The metrics to be calculated. The dictionary should have the metric
            name as key and the `torchmetrics.Metric`-like object as value.
        y_hat : torch.Tensor
            The predictions tensor.
        y : torch.Tensor
            The targets tensor.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the metric name as key and the list of metric
            values as value. The list will have a single element if
            `apply_metrics_per_sample` is False, otherwise it will have a value.
        """
        results = {}
        if self._apply_metrics_per_sample:
            y, y_hat = y.split(1), y_hat.split(1)
        else:
            y, y_hat = y.unsqueeze(0), y_hat.unsqueeze(0)

        for metric_name, metric in metrics.items():
            final_results = []
            for i, (y_i, y_hat_i) in enumerate(zip(y, y_hat)):
                res = metric(y_i, y_hat_i).float().item()
                final_results.append(res)
            results[metric_name] = final_results
            print(f"Metric {metric_name}: {final_results}")

        return results

    # Private methods
    def _fit(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike] = None):
        """Fit the model using the given data.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `train_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.
        """
        return self._trainer.fit(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _test(self, data: L.LightningDataModule, ckpt_path: Optional[PathLike] = None):
        """Test the model using the given data.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `test_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.
        """
        return self._trainer.test(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )

    def _predict(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike] = None,
    ) -> torch.Tensor:
        """Predict using the given data.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `predict_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.

        Returns
        -------
        torch.Tensor
            The predictions tensor.
        """
        return self._trainer.predict(
            model=self._model, datamodule=data, ckpt_path=ckpt_path
        )  # type: ignore

    def _evaluate(
        self,
        data: L.LightningDataModule,
        ckpt_path: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model and calculate regression and/or classification
        metrics.

        Parameters
        ----------
        data : L.LightningDataModule
            The data module to be used. The data module should have the
            `predict_dataloader` method implemented.
        ckpt_path : PathLike
            The checkpoint path to be used. If None, no checkpoint will be used.

        Returns
        -------
        Dict[str, Dict[str, Any]
            A dictionary with metrics.
        """
        metrics = defaultdict(dict)

        # Get the predictions and targets
        _, y = get_full_data_split(data, "predict")
        y = torch.tensor(y, device="cpu")

        print(f"🔍 True labels shape: {y.shape}")
        print(f"🔍 Unique true classes: {torch.unique(y)}")
        print(f"🔍 True class distribution: {torch.bincount(y)}")
        y_hat = self.trainer.predict(self._model, datamodule=data, ckpt_path=ckpt_path)
        y_hat = torch.cat(y_hat).detach().cpu()  # type: ignore
        print("\n\n saving y and yhat \n\n")
        y_hat_raw = y_hat.tolist()
        print(f"🔍 Raw predictions shape: {y_hat.shape}")
        # Convert to list so it's JSON serializable
        y_list = y.tolist()

        # Check if the shapes are the same
        if len(y_hat) != len(y):
            raise ValueError(
                f"Shapes are different: y_hat shape: {y_hat.shape}; y shape: {y.shape}. Is `limit_predict_batches` set?"
            )

        # Argmax and calculate metrics
        if self._classification_metrics is not None:
            print(f"Running classification metrics...")
            y_hat_logits = y_hat.softmax(dim=1).tolist()
            y_hat_probs = y_hat.softmax(dim=1)

            y_hat = torch.argmax(y_hat, dim=1)

            # ===============================
            # Misclassification analysis # ADDED MISCLASSIFICATION
            # ===============================
            y_true = y.cpu()
            y_pred = y_hat.cpu()

            misclassified_mask = y_true != y_pred

            total_samples = len(y_true)
            total_misclassified = misclassified_mask.sum().item()
            total_misclassification_rate = total_misclassified / total_samples

            # Per-class misclassification
            per_class_stats = {}

            classes = torch.unique(y_true)
            for c in classes:
                class_mask = y_true == c
                class_total = class_mask.sum().item()
                class_mis = ((y_pred != y_true) & class_mask).sum().item()

                per_class_stats[int(c)] = {
                    "total": class_total,
                    "misclassified": class_mis,
                    "misclassification_rate": (
                        class_mis / class_total if class_total > 0 else 0.0
                    ),
                }

            print("\n📊 Misclassification summary")
            print(
                f"Total: {total_misclassified}/{total_samples} "
                f"({total_misclassification_rate:.4%})"
            )
            for c, stats in per_class_stats.items():
                print(
                    f"Class {c}: {stats['misclassified']}/{stats['total']} "
                    f"({stats['misclassification_rate']:.2%})"
                )
            # END MISCLASSIFICATION
            # other debus

            y_hat_classes = torch.argmax(y_hat_probs, dim=1)
            y_hat_probs_list = y_hat_probs.tolist()  # Probabilities
            print(f"🔍 Predicted classes shape: {y_hat_classes.shape}")
            print(f"🔍 Unique predicted classes: {torch.unique(y_hat_classes)}")
            print(f"🔍 Predicted class distribution: {torch.bincount(y_hat_classes)}")
            if len(torch.unique(y_hat_classes)) == 1:
                print(
                    f"⚠️  WARNING: All predictions are the same class: {y_hat_classes[0]}"
                )
            # end other debugs
            y_hat_list = y_hat.tolist()  # original logits
            with open(self._log_dir / "predictions.json", "w") as f:
                json.dump(
                    {
                        "y": y_list,
                        "y_hat": y_hat_list,
                        "y_hat_raw": y_hat_raw,
                        "y_hat_logits_probs": y_hat_logits,
                    },
                    f,
                    indent=2,
                )
            metrics["classification"] = self._calculate_metrics(
                self._classification_metrics, y_hat, y
            )
            # Calculate accuracy manually for verification
            correct = (torch.tensor(y_hat_list) == torch.tensor(y_list)).sum().item()
            accuracy = correct / len(y_list)
            print(f"🔍 Manual accuracy: {accuracy:.4f} ({correct}/{len(y_list)})")

            balanced_accuracy_metric = BalancedAccuracy(
                num_classes=6, task="multiclass", adjusted=False
            )
            print(balanced_accuracy_metric)
            # y_pred = torch.tensor(y_pred)
            # y_true = torch.tensor(y_true)
            bal_acc = balanced_accuracy_metric(y_hat, y)
            print(f"\n\n🔍 Balanced Accuracy: {bal_acc:.4f}\n\n")
        # Just calculate metrics (without argmax)
        elif self._regression_metrics is not None:
            print(f"Running regression metrics...")
            metrics["regression"] = self._calculate_metrics(
                self._regression_metrics, y_hat, y
            )

        else:
            pass

        # Run model analysis
        if self._model_analysis is not None:
            print(f"Running model analysis...")
            metrics["analysis"] = {}
            for analysis_name, analysis in self._model_analysis.items():
                analysis.path = self._log_dir
                metrics["analysis"][analysis_name] = analysis.compute(self._model, data)

        # Save metrics
        metrics = dict(metrics)
        # ADDED MISCLASSIFICATION
        metrics["misclassification"] = {
            "total": {
                "samples": total_samples,
                "misclassified": total_misclassified,
                "rate": total_misclassification_rate,
            },
            "per_class": per_class_stats,
        }

        error_pairs = [
            {"y": int(t), "y_pred": int(p)}
            for t, p in zip(y_true[misclassified_mask], y_pred[misclassified_mask])
        ]

        with open(self._log_dir / "misclassifications.json", "w") as f:
            json.dump(error_pairs, f, indent=2)
        # END  MISCLASSIFICATION

        # Save metrics to a YAML file
        if self._save_run_status:
            yaml_path = self._log_dir / f"metrics_{self.pipeline_id}.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(metrics, f)
                print(f"Metrics saved to {yaml_path}")

        return metrics

    def _run(
        self,
        data: L.LightningDataModule,
        task: Literal["fit", "test", "predict", "evaluate"],
        ckpt_path: Optional[PathLike] = None,
    ):
        self._data = data

        start = time.perf_counter()

        if task == "fit":
            result = self._fit(data, ckpt_path)
        elif task == "test":
            result = self._test(data, ckpt_path)
        elif task == "predict":
            result = self._predict(data, ckpt_path)
        elif task == "evaluate":
            result = self._evaluate(data, ckpt_path)
        else:
            raise ValueError(f"Unknown task: {task}")

        elapsed = time.perf_counter() - start

        timing_row = {
            "pipeline_id": self.pipeline_id,
            "log_dir": str(self._log_dir),
            "task": task,
            "elapsed_seconds": round(elapsed, 4),
            "model": self.model.__class__.__name__,
        }

        csv_path = self._log_dir / f"timings_{task}.csv"
        _append_timing_csv(csv_path, timing_row)

        print(f"⏱️ {task} took {elapsed:.2f}s → saved to {csv_path}")

        return result

    # # Default run method (entry point)
    # def _run(
    #     self,
    #     data: L.LightningDataModule,
    #     task: Literal["fit", "test", "predict", "evaluate"],
    #     ckpt_path: Optional[PathLike] = None,
    # ):
    #     """
    #     Run the specified task on the given data.

    #     Parameters
    #     ----------
    #     data : L.LightningDataModule
    #         The LightningDataModule object containing the data for the task.
    #     task : Literal["fit", "test", "predict", "evaluate"], optional
    #         The task to be performed. Valid options are "fit", "test",
    #         "predict", and "evaluate".
    #     ckpt_path : PathLike, optional
    #         The path to the checkpoint file to be used for resuming training or
    #         performing inference. Defaults to None.

    #     Returns
    #     -------
    #     Any
    #         The result of the specified task.

    #     Raises
    #     ------
    #     ValueError
    #         If an unknown task is provided.
    #     """
    #     self._data = data

    #     if task == "fit":
    #         return self._fit(data, ckpt_path)
    #     elif task == "test":
    #         return self._test(data, ckpt_path)
    #     elif task == "predict":
    #         return self._predict(data, ckpt_path)
    #     elif task == "evaluate":
    #         return self._evaluate(data, ckpt_path)
    #     else:
    #         raise ValueError(f"Unknown task: {task}")


def cli_main():
    from jsonargparse import CLI

    CLI(SimpleLightningPipeline, as_positional=False)  # , parser_mode="omegaconf")
    print("✨ 🍰 ✨")


if __name__ == "__main__":
    cli_main()
